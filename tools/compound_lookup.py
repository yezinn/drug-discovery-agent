"""PubChem PUG REST API로 화합물명 → SMILES 변환.

구조기반 계산의 입력값(SMILES)을 화합물 '이름'만으로도 얻을 수 있게 해서
molecular_properties 툴과 자연스럽게 이어지도록 하는 보조 tool.

주의: 이 프로젝트가 실행되는 클라우드 작업공간은 PubChem 등 일반 웹 접근이
네트워크 정책상 막혀 있어 여기서는 실제 호출 테스트를 하지 못했다. 로컬 환경
(예: 예진님 맥북)에서는 정상적으로 인터넷에 접근되므로 문제없이 동작해야 하며,
merge 후 로컬에서 반드시 재검증 필요 (README 참고).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class CompoundLookupError(Exception):
    """PubChem 조회 실패(네트워크 오류, 이름 없음 등) 시 발생."""


@dataclass
class CompoundLookupResult:
    name: str
    cid: Optional[int]
    smiles: Optional[str]
    found: bool


def lookup_compound(name: str, timeout: int = 10) -> CompoundLookupResult:
    """화합물명으로 PubChem CID와 canonical SMILES를 조회한다.

    두 번의 요청이 필요하다: (1) 이름 → CID, (2) CID → SMILES.
    이름이 PubChem에 없으면 found=False로 반환(예외를 던지지 않음) —
    에이전트가 "결과 없음"을 자연스럽게 답할 수 있도록.

    함수 전체를 하나의 try/except로 감싼다 — 이전 버전은 네트워크 요청(`requests.get`)만
    잡고 `raise_for_status()`/`.json()`/딕셔너리 접근은 밖에 있어서, PubChem이 4xx·5xx를
    반환하거나 예상과 다른 JSON 구조를 주면 예외가 그대로 새어나가 FastAPI가 500(빈 응답
    본문)을 뱉었다. 실제로 로컬 실행에서 이 경로로 500이 재현됨 — 원인 그대로 여기 기록.
    """
    name = name.strip()
    if not name:
        raise CompoundLookupError("빈 화합물명입니다.")

    cid_url = f"{PUBCHEM_BASE}/compound/name/{requests.utils.quote(name)}/cids/JSON"
    # PubChem이 "CanonicalSMILES"/"IsomericSMILES" property 이름을 폐기하고 "SMILES"로
    # 통합했다 (실제로 로컬 실행에서 KeyError: 'CanonicalSMILES'로 재현됨). 여러 property를
    # 한 번에 요청해서, 어느 이름으로 오든 파싱 쪽에서 순서대로 fallback한다.
    smiles_url_template = f"{PUBCHEM_BASE}/compound/cid/{{cid}}/property/SMILES,ConnectivitySMILES,IsomericSMILES,CanonicalSMILES/JSON"
    SMILES_KEYS = ("SMILES", "ConnectivitySMILES", "IsomericSMILES", "CanonicalSMILES")

    try:
        cid_resp = requests.get(cid_url, timeout=timeout)

        if cid_resp.status_code == 404:
            return CompoundLookupResult(name=name, cid=None, smiles=None, found=False)
        cid_resp.raise_for_status()

        cids = cid_resp.json().get("IdentifierList", {}).get("CID", [])
        if not cids:
            return CompoundLookupResult(name=name, cid=None, smiles=None, found=False)
        cid = cids[0]

        smiles_resp = requests.get(smiles_url_template.format(cid=cid), timeout=timeout)
        smiles_resp.raise_for_status()

        props = smiles_resp.json().get("PropertyTable", {}).get("Properties", [])
        smiles = None
        if props:
            for key in SMILES_KEYS:
                if key in props[0]:
                    smiles = props[0][key]
                    break

        return CompoundLookupResult(name=name, cid=cid, smiles=smiles, found=smiles is not None)

    except requests.RequestException as e:
        raise CompoundLookupError(f"PubChem 요청 중 네트워크/HTTP 오류: {e}") from e
    except (ValueError, KeyError, IndexError, TypeError) as e:
        # ValueError는 requests의 JSONDecodeError(응답 본문이 JSON이 아닐 때)를 포함한다.
        raise CompoundLookupError(f"PubChem 응답 파싱 중 오류(예상과 다른 형식): {e}") from e


def format_for_agent(result: CompoundLookupResult) -> str:
    if not result.found:
        return f"'{result.name}'을(를) PubChem에서 찾지 못했습니다."
    return f"{result.name} (PubChem CID {result.cid}): SMILES = {result.smiles}"
