"""compound_lookup.py 단위 테스트. 실제 PubChem 네트워크 호출 없이 requests.get을 모킹한다.

로컬 실행에서 실제로 재현된 버그(KeyError: 'CanonicalSMILES' — PubChem이 해당 property 이름을
폐기하고 'SMILES'로 바꾼 게 원인)를 회귀 테스트로 고정해둔다.
"""
from unittest.mock import Mock, patch

import pytest

from tools.compound_lookup import CompoundLookupError, lookup_compound


def _mock_response(status_code: int, json_data: dict):
    resp = Mock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.side_effect = None
    return resp


@patch("tools.compound_lookup.requests.get")
def test_lookup_compound_with_smiles_key(mock_get):
    """PubChem이 현재 실제로 쓰는 'SMILES' 키로 응답하는 정상 케이스."""
    cid_resp = _mock_response(200, {"IdentifierList": {"CID": [5291]}})
    smiles_resp = _mock_response(
        200, {"PropertyTable": {"Properties": [{"CID": 5291, "SMILES": "CC(=O)Nc1ccc(O)cc1"}]}}
    )
    mock_get.side_effect = [cid_resp, smiles_resp]

    result = lookup_compound("Acetaminophen")

    assert result.found is True
    assert result.cid == 5291
    assert result.smiles == "CC(=O)Nc1ccc(O)cc1"


@patch("tools.compound_lookup.requests.get")
def test_lookup_compound_falls_back_to_canonical_smiles_key(mock_get):
    """옛 property 이름('CanonicalSMILES')만 오는 경우에도 깨지지 않고 파싱해야 한다."""
    cid_resp = _mock_response(200, {"IdentifierList": {"CID": [2244]}})
    smiles_resp = _mock_response(
        200,
        {"PropertyTable": {"Properties": [{"CID": 2244, "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(=O)O"}]}},
    )
    mock_get.side_effect = [cid_resp, smiles_resp]

    result = lookup_compound("Aspirin")

    assert result.found is True
    assert result.smiles == "CC(=O)OC1=CC=CC=C1C(=O)O"


@patch("tools.compound_lookup.requests.get")
def test_lookup_compound_missing_smiles_key_raises_clean_error(mock_get):
    """SMILES 계열 키가 응답에 하나도 없으면 KeyError로 서버가 죽는 대신
    found=False로 정상 반환해야 한다 (실제로 재현됐던 500 버그의 회귀 테스트)."""
    cid_resp = _mock_response(200, {"IdentifierList": {"CID": [999]}})
    smiles_resp = _mock_response(200, {"PropertyTable": {"Properties": [{"CID": 999}]}})
    mock_get.side_effect = [cid_resp, smiles_resp]

    result = lookup_compound("SomeCompound")

    assert result.found is False
    assert result.smiles is None


@patch("tools.compound_lookup.requests.get")
def test_lookup_compound_not_found(mock_get):
    mock_get.return_value = _mock_response(404, {})

    result = lookup_compound("ThisCompoundDoesNotExist12345")

    assert result.found is False
    assert result.cid is None


@patch("tools.compound_lookup.requests.get")
def test_lookup_compound_malformed_json_raises_compound_lookup_error(mock_get):
    """응답 본문이 JSON이 아니면(예: 프록시가 HTML 에러 페이지를 끼워넣는 경우)
    ValueError가 그대로 새지 않고 CompoundLookupError로 변환되어야 한다."""
    bad_resp = Mock()
    bad_resp.status_code = 200
    bad_resp.raise_for_status.side_effect = None
    bad_resp.json.side_effect = ValueError("Expecting value: line 1 column 1 (char 0)")
    mock_get.return_value = bad_resp

    with pytest.raises(CompoundLookupError):
        lookup_compound("Whatever")


def test_lookup_compound_empty_name_raises():
    with pytest.raises(CompoundLookupError):
        lookup_compound("   ")
