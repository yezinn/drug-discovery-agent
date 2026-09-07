# Drug Discovery Assistant API

FastAPI + LangChain(tool-calling agent) + RDKit 기반 신약개발 보조 백엔드.
LLM이 필요에 따라 PubMed 문헌 검색, PubChem 화합물 조회, RDKit 분자 물성 계산 tool을 조합해서 답한다.

## Architecture

```
Client → FastAPI (main.py)
           ├── POST /molecule/properties  → tools/molecular_properties.py (RDKit, LLM 불필요)
           ├── POST /compound/lookup      → tools/compound_lookup.py (PubChem REST, LLM 불필요)
           ├── POST /literature/search    → tools/literature_search.py (PubMed E-utilities, LLM 불필요)
           └── POST /agent/query          → agent.py (LangGraph tool-calling agent, Gemini)
                                              ├─ literature_search_tool → tools/literature_search.py
                                              ├─ compound_lookup_tool   → tools/compound_lookup.py
                                              └─ molecular_properties_tool → tools/molecular_properties.py
```

`/molecule/properties`, `/compound/lookup`, `/literature/search` 3개는 에이전트/LLM 없이 그 자체로
동작하는 독립 REST API다. `/agent/query`는 자연어 질문을 받아 LLM이 위 3개 tool 중 필요한 것을
스스로 선택·체이닝해서 답한다 (내부적으로 같은 tool 함수를 재사용).

## Setup

**Python 버전**: 3.11~3.12 권장. `biopython`, `pydantic-core` 등 일부 의존성이 아직 최신 Python(3.14 등)용
사전 빌드 wheel을 제공하지 않아, 시스템 기본 `python3`가 너무 최신이면 소스 빌드 중 실패할 수 있다
(C API 변경, Rust 바인딩 미지원 등). conda가 있으면 버전을 명시적으로 고정하는 게 가장 안전하다:

```bash
conda create -n drug-discovery-agent python=3.12 -y
conda activate drug-discovery-agent
pip install -r requirements.txt
cp .env.example .env          # GOOGLE_API_KEY, ENTREZ_EMAIL 채우기
python -m uvicorn main:app --reload   # 'uvicorn ...'만 쓰면 PATH상 다른 Python의 uvicorn이 잡힐 수 있음
```

conda 없이 표준 `venv`로 하고 싶다면, 시스템에 3.11/3.12가 따로 설치되어 있는지 확인 후
(`which python3.12`) 그 인터프리터로 venv를 만든다 — `python3 -m venv venv`처럼 버전을 지정하지 않으면
시스템 기본 `python3`(너무 최신일 수 있음)로 만들어진다:

```bash
python3.12 -m venv venv
source venv/bin/activate      # Windows는 venv\Scripts\activate
pip install -r requirements.txt
```

작업 끝나면 `deactivate`(venv) 또는 `conda deactivate`(conda)로 빠져나올 수 있다. `venv/`는
`.gitignore`에 포함되어 있어 커밋되지 않는다.

- `GOOGLE_API_KEY`: https://aistudio.google.com/apikey (무료 티어)
- `ENTREZ_EMAIL`: PubMed E-utilities 요청 시 필요 (가입 불필요, 이메일 형식만 맞으면 됨)

### Docker

```bash
docker build -t drug-discovery-agent .
docker run -p 8000:8000 --env-file .env drug-discovery-agent
```

## API

| Method | Path | 설명 | LLM 필요 |
|---|---|---|---|
| GET | `/health` | 헬스체크 | 아니오 |
| POST | `/molecule/properties` | `{"smiles": "..."}` → MW, LogP, TPSA, Lipinski's Rule of Five 통과 여부 | 아니오 |
| POST | `/compound/lookup` | `{"name": "Imatinib"}` → PubChem CID, SMILES | 아니오 |
| POST | `/literature/search` | `{"query": "...", "max_results": 5}` → PubMed 논문 목록(PMID/제목/초록) | 아니오 |
| POST | `/agent/query` | `{"question": "..."}` → 에이전트가 tool 조합해서 답변 + tool 호출 trace | 예 |

예시 (실제 로컬 실행 결과 기준, 2026-09-01 검증):

```bash
curl -X POST localhost:8000/molecule/properties \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}'

curl -X POST localhost:8000/compound/lookup \
  -H "Content-Type: application/json" \
  -d '{"name": "Imatinib"}'
# → {"name":"Imatinib","cid":5291,"smiles":"CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)NC4=NC=CC(=N4)C5=CN=CC=C5","found":true}

curl -X POST localhost:8000/literature/search \
  -H "Content-Type: application/json" \
  -d '{"query": "imatinib resistance mechanism"}'
# → PubMed 상위 5개 논문(PMID, 제목, 초록) 반환

curl -X POST localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "이매티닙 SMILES 구조 찾아서 Rule of Five 통과하는지 알려줘"}'
# → 에이전트가 compound_lookup_tool → molecular_properties_tool 순으로 자동 체이닝해서
#   SMILES/MW/LogP/HBD/HBA/RO5 통과 여부를 마크다운 형식으로 답변 (tool_calls 필드로 호출 순서 확인 가능)

# 3개 tool 모두 필요한 질문 (문헌 검색 + 화합물 조회 + 물성 계산):
curl -X POST localhost:8000/agent/query \
  -H "Content-Type: application/json" \
  -d '{"question": "이매티닙 SMILES 찾아서 분자 물성이랑 Rule of Five 통과 여부 알려주고, 이매티닙 내성 관련 최신 연구도 찾아줘"}'
# → tool_calls: compound_lookup_tool → literature_search_tool → molecular_properties_tool
#   (SMILES 조회와 무관한 literature_search를 물성 계산보다 먼저 호출 — 의존성 없는 tool의
#   실행 순서는 고정되어 있지 않고 LLM이 그때그때 정한다는 걸 보여주는 예)
```

`| python3 -m json.tool`을 붙이면 응답 JSON을 들여쓰기해서 보기 편하다 (예:
`curl -s ... | python3 -m json.tool`). 브라우저에서 직접 호출해보고 싶다면 서버 실행 중
`http://localhost:8000/docs`(Swagger UI)에서 각 엔드포인트의 "Try it out" 버튼으로 curl 없이
바로 테스트할 수 있다.

curl이나 Swagger UI보다 간단하게 써보고 싶다면 서버 실행 중 `http://localhost:8000`(루트 경로)에
접속하면 된다 — `static/index.html`로 구현된 데모 페이지로, 4개 탭(물성 계산/화합물 조회/문헌
검색/AI 에이전트)에서 폼 입력만으로 바로 결과를 확인할 수 있다.

## Testing

```bash
pytest tests/ -v   # 28개 전부 통과 (2026-09-01 로컬 검증 기준)
```

전부 네트워크/API 키 없이 로컬에서 바로 실행 가능하다. 실제 외부 API(Gemini·PubChem·NCBI) 호출은
`unittest.mock`으로 모킹해서 응답 파싱 로직만 검증한다:

- `tests/test_molecular_properties.py`: Aspirin·Erythromycin의 PubChem 공개 물성값과 대조해 RDKit
  계산 결과 검증
- `tests/test_compound_lookup.py`: PubChem PUG REST 응답을 모킹. PubChem이 `CanonicalSMILES` 키를
  폐기하고 `SMILES`로 바꾼 실제 겪은 breaking change의 회귀 테스트 포함
- `tests/test_literature_search.py`: NCBI Entrez 응답을 모킹. 정상 파싱, 빈 검색 결과, 응답 구조가
  예상과 다른 항목 스킵, `ENTREZ_EMAIL` 미설정 에러 등 검증
- `tests/test_agent.py`: 에이전트 응답의 `answer` 텍스트 추출 로직(`_extract_answer_text`) 검증.
  Gemini가 content를 블록 리스트로 반환해 응답이 지저분해지던 버그의 회귀 테스트
- `tests/test_api.py`: FastAPI 엔드포인트 자체(헬스체크, 422/500 에러 처리 등)

실제 외부 API를 살아있는 상태로 호출하는 통합 테스트는 별도로 없고, 이 문서의 curl 예시로 수동
검증했다 (2026-09-01, 로컬 conda 환경 Python 3.12).

## Design notes

- **LangChain 1.x agent API**: 구버전(`langchain.agents.AgentExecutor` + `create_tool_calling_agent`)이
  1.x에서 LangGraph 기반 `create_agent`로 대체되었다. `langchain.agents`에 실제로 어떤 이름이
  노출되는지 직접 import해서 확인 후 새 API로 작성함. `create_agent(...).invoke({"messages": [...]})`는
  `messages` 리스트를 반환하며, `AIMessage.tool_calls`와 `ToolMessage.tool_call_id`를 매칭해서
  tool 호출 trace를 복원한다.
- **literature_search는 벡터DB를 쓰지 않는다**: 매 요청마다 새로 인덱스를 구축하는 대신, PubMed
  검색 결과 상위 N개 초록을 그대로 LLM 컨텍스트에 넣어 판단하게 한다. 단발성 질의에 벡터DB는
  과설계라고 판단.
- **Cheminformatics 범위를 의도적으로 제한**: 분자 물성 계산(RDKit descriptor, Lipinski's Rule of
  Five)까지만 다루고, 도킹 시뮬레이션이나 3D 구조 예측은 포함하지 않았다.
- **네트워크 제약과 로컬 재검증**: 초기 스캐폴딩은 외부 네트워크가 제한된 환경에서 작성되어
  `compound_lookup`(PubChem), `literature_search`(NCBI E-utilities), `/agent/query`(Gemini API)는
  실제 호출 테스트 없이 로직만으로 작성됐었다. 로컬 재검증 과정에서 실제로 두 가지 버그를 발견해
  수정함: (1) PubChem이 `CanonicalSMILES` property를 폐기하고 `SMILES`로 통합해서 생긴 `KeyError`
  → 여러 property 이름을 순서대로 fallback하도록 수정, (2) Gemini가 메시지 content를 단순 문자열이
  아니라 블록 리스트로 반환해 `/agent/query`의 `answer` 필드가 지저분하게 나가던 문제 → 텍스트
  블록만 추출하는 헬퍼 추가. 둘 다 클라우드 작업공간에서는 재현이 안 되고 로컬에서만 재현된 문제였다.

## Limitations

- Lipinski's Rule of Five는 경구 흡수 가능성에 대한 경험칙일 뿐, 실제 약물 개발 판단 기준으로는
  불충분하다 (Ro5를 벗어나는 승인 약물도 다수 존재).
- `literature_search`는 최신 문헌 상위 N개를 그대로 컨텍스트에 넣을 뿐, 관련도 재순위화나
  hallucination 방지 장치(Grounding Guard)가 없다 — 별도 프로젝트인 pubmed-rag-qa의 Grounding
  Guard/평가 파이프라인과는 목적이 다르다.
