"""FastAPI 백엔드 진입점.

엔드포인트:
  GET  /health                    - 헬스체크
  POST /molecule/properties       - SMILES → 물성 계산 (RDKit, LLM 불필요)
  POST /compound/lookup           - 화합물명 → SMILES 조회 (PubChem, LLM 불필요)
  POST /literature/search         - PubMed 문헌 검색 (Biopython Entrez, LLM 불필요)
  POST /agent/query               - 자연어 질문 → LangChain 에이전트가 tool 조합해 답변 (LLM 필요)

/molecule/properties, /compound/lookup, /literature/search 3개는 에이전트 없이도
그 자체로 쓸 수 있는 독립 API로 설계했다 — 에이전트 데모가 안 되는 상황(API 키
문제 등)에서도 "Python·FastAPI 기반 백엔드 API 설계·개발" 역량 자체는 별도로
시연 가능하도록.
"""
from __future__ import annotations

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from schemas import (
    AgentQueryRequest,
    AgentQueryResponse,
    CompoundLookupRequest,
    CompoundLookupResponse,
    LiteratureSearchRequest,
    LiteratureSearchResponse,
    MoleculePropertiesRequest,
    MoleculePropertiesResponse,
    PubMedArticleModel,
    ToolCallTrace,
)
from tools.compound_lookup import CompoundLookupError, lookup_compound
from tools.literature_search import LiteratureSearchError, search_literature
from tools.molecular_properties import InvalidSMILESError, calculate_properties

load_dotenv()

app = FastAPI(
    title="Drug Discovery Assistant API",
    description=(
        "AI 신약개발 보조 에이전트 — 문헌 검색, 화합물 조회, "
        "cheminformatics 물성 계산을 FastAPI + LangChain으로 통합."
    ),
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/molecule/properties", response_model=MoleculePropertiesResponse)
def molecule_properties(req: MoleculePropertiesRequest) -> MoleculePropertiesResponse:
    try:
        props = calculate_properties(req.smiles)
    except InvalidSMILESError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    return MoleculePropertiesResponse(
        smiles=props.smiles,
        molecular_weight=props.molecular_weight,
        logp=props.logp,
        h_bond_donors=props.h_bond_donors,
        h_bond_acceptors=props.h_bond_acceptors,
        tpsa=props.tpsa,
        rotatable_bonds=props.rotatable_bonds,
        lipinski_violations=props.lipinski_violations,
        passes_lipinski_ro5=props.passes_lipinski_ro5,
        violation_details=props.violation_details,
    )


@app.post("/compound/lookup", response_model=CompoundLookupResponse)
def compound_lookup(req: CompoundLookupRequest) -> CompoundLookupResponse:
    try:
        result = lookup_compound(req.name)
    except CompoundLookupError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return CompoundLookupResponse(
        name=result.name, cid=result.cid, smiles=result.smiles, found=result.found
    )


@app.post("/literature/search", response_model=LiteratureSearchResponse)
def literature_search(req: LiteratureSearchRequest) -> LiteratureSearchResponse:
    try:
        articles = search_literature(req.query, max_results=req.max_results)
    except LiteratureSearchError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return LiteratureSearchResponse(
        query=req.query,
        articles=[
            PubMedArticleModel(pmid=a.pmid, title=a.title, abstract=a.abstract)
            for a in articles
        ],
    )


@app.post("/agent/query", response_model=AgentQueryResponse)
def agent_query(req: AgentQueryRequest) -> AgentQueryResponse:
    # 지연 import: GOOGLE_API_KEY 없이도 /health, /molecule/properties 등은
    # 정상 동작해야 하므로 에이전트 관련 무거운 import를 이 엔드포인트 호출 시점으로 미룬다.
    from agent import run_agent_query

    try:
        result = run_agent_query(req.question)
    except RuntimeError as e:
        # GOOGLE_API_KEY 미설정 등 설정 문제 (agent.build_agent에서 명시적으로 발생시킴)
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        # LangGraph 내부에서 tool의 예외를 그대로 재-raise하는 경우가 있어(예: tool 함수가
        # 처리하지 못한 예외), 여기서도 한 번 더 넓게 막아 클라이언트에 빈 500 대신 원인이
        # 담긴 응답을 준다. 실제로 compound_lookup의 KeyError가 이 경로로 재현된 적 있음.
        raise HTTPException(status_code=502, detail=f"에이전트 실행 중 오류: {e}") from e

    tool_calls = [
        ToolCallTrace(tool=tc["tool"], input=tc["input"], output=tc["output"])
        for tc in result["tool_calls"]
    ]

    return AgentQueryResponse(question=req.question, answer=result["answer"], tool_calls=tool_calls)


# 데모용 프론트엔드 (static/index.html) — curl 없이 브라우저에서 바로 써볼 수 있게
app.mount("/", StaticFiles(directory="static", html=True), name="static")
