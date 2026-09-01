"""LangChain 기반 신약개발 보조 에이전트.

LLM(Gemini)이 질문 내용에 따라 스스로 판단해서
  1) literature_search  — PubMed 실시간 문헌 검색
  2) compound_lookup    — 화합물명 → SMILES
  3) molecular_properties — SMILES → 물성/Lipinski's Rule of Five
세 tool 중 필요한 것을 호출(tool calling)하도록 구성한다.

예: "이매티닙의 분자량이랑 Rule of Five 통과하는지 알려주고, 관련 최신 연구도 찾아줘"
    → compound_lookup("Imatinib") → molecular_properties(smiles) → literature_search(...)
    순으로 에이전트가 스스로 체이닝해서 답한다.

구현 메모: LangChain 1.x부터 구버전 `AgentExecutor`/`create_tool_calling_agent`
(langchain.agents) API가 사라지고, LangGraph 기반 `create_agent`로 대체되었다
(직접 확인: `from langchain.agents import AgentExecutor` → ImportError,
`langchain.agents`에는 `create_agent`만 존재). pubmed-rag-qa 프로젝트 때도
겪었던 것과 같은 종류의 메이저 버전 breaking change라 미리 확인하고
새 API로 작성함.
"""
from __future__ import annotations

import os

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.compound_lookup import CompoundLookupError, format_for_agent as format_compound
from tools.compound_lookup import lookup_compound
from tools.literature_search import LiteratureSearchError, format_for_agent as format_literature
from tools.literature_search import search_literature
from tools.molecular_properties import InvalidSMILESError, format_for_agent as format_properties
from tools.molecular_properties import calculate_properties

SYSTEM_PROMPT = """당신은 신약개발 연구자를 돕는 AI 에이전트입니다.
아래 세 가지 tool을 상황에 맞게 조합해서 사용하세요.

- literature_search: 최신 연구 동향, 특정 기전/약물에 대한 문헌 근거가 필요할 때
- compound_lookup: 화합물 '이름'만 알고 SMILES 구조가 필요할 때
- molecular_properties: SMILES가 주어졌거나 compound_lookup으로 얻었을 때, 물성/Lipinski 규칙 확인용

절대 tool 없이 화합물의 물성이나 논문 내용을 지어내서 답하지 마세요.
근거가 부족하면 "확인된 정보가 없습니다"라고 솔직히 답하세요."""


@tool
def literature_search_tool(query: str) -> str:
    """PubMed에서 query와 관련된 최신 논문을 실시간으로 검색해 제목과 초록을 반환한다."""
    try:
        articles = search_literature(query)
    except LiteratureSearchError as e:
        return f"검색 실패: {e}"
    return format_literature(articles)


@tool
def compound_lookup_tool(compound_name: str) -> str:
    """화합물명(예: Imatinib, Aspirin)으로 PubChem에서 SMILES 구조를 조회한다."""
    try:
        result = lookup_compound(compound_name)
    except CompoundLookupError as e:
        return f"조회 실패: {e}"
    return format_compound(result)


@tool
def molecular_properties_tool(smiles: str) -> str:
    """SMILES 문자열로부터 분자량, LogP, TPSA 등 물성과 Lipinski's Rule of Five 통과 여부를 계산한다."""
    try:
        props = calculate_properties(smiles)
    except InvalidSMILESError as e:
        return f"계산 실패: {e}"
    return format_properties(props)


TOOLS = [literature_search_tool, compound_lookup_tool, molecular_properties_tool]


def build_agent(model_name: str = "gemini-3.5-flash-lite"):
    """LangGraph 기반 tool-calling 에이전트를 생성한다. GOOGLE_API_KEY 필요."""
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY 환경변수가 설정되어 있지 않습니다.")

    llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
    return create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)


def _extract_answer_text(content) -> str:
    """LangChain 메시지의 content에서 사람이 읽을 답변 텍스트만 뽑아낸다.

    Gemini(langchain_google_genai)는 content를 단순 str이 아니라
    [{"type": "text", "text": ..., "extras": {...}}] 형태의 블록 리스트로 반환하는
    경우가 있다. isinstance(content, str) 분기만으로는 이 경우 str(list)가 그대로
    answer에 들어가 "[{'type': 'text', ...}]" 같은 지저분한 값이 나간다.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        if texts:
            return "\n".join(texts)
    # 예상 밖 구조(향후 langchain/모델 쪽 포맷 변경 등)라도 예외를 던지지 않고
    # 최소한 뭐라도 반환한다 — PubChem 응답 키 변경 때 겪은 것과 같은 종류의
    # breaking change가 다시 나도 500이 빈 채로 새어나가지 않도록.
    return str(content)


def run_agent_query(question: str, model_name: str = "gemini-3.5-flash-lite") -> dict:
    """에이전트를 1회 실행하고, main.py의 AgentQueryResponse에 맞는 형태로 결과를 정리한다.

    반환값: {"answer": str, "tool_calls": [{"tool": str, "input": str, "output": str}, ...]}
    """
    graph = build_agent(model_name=model_name)
    result = graph.invoke({"messages": [("human", question)]})
    messages = result["messages"]

    # AIMessage.tool_calls로 "어떤 tool을 어떤 인자로 불렀는지"를,
    # 그 뒤에 오는 ToolMessage(tool_call_id로 매칭)로 "그 tool이 뭘 반환했는지"를 복원한다.
    tool_call_inputs: dict[str, dict] = {}
    for msg in messages:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls:
                tool_call_inputs[call["id"]] = {"tool": call["name"], "input": str(call["args"])}

    tool_calls = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            call_info = tool_call_inputs.get(msg.tool_call_id, {"tool": msg.name, "input": ""})
            tool_calls.append(
                {"tool": call_info["tool"], "input": call_info["input"], "output": str(msg.content)}
            )

    final_message = messages[-1]
    answer = _extract_answer_text(final_message.content)

    return {"answer": answer, "tool_calls": tool_calls}
