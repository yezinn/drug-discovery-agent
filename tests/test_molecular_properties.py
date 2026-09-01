"""molecular_properties.py 단위 테스트.

Aspirin(경구 약물, RO5 통과)과 Erythromycin(대형 마크로라이드 항생제, RO5 위반)으로
경계 케이스를 검증한다. 알려진 약물의 공개 물성값과 대조해 계산이 맞는지 확인하는 것이
목적 — RDKit을 "가져다 쓰기만" 한 게 아니라 결과를 직접 검증했다는 근거를 남기기 위함.
"""
import pytest

from tools.molecular_properties import (
    InvalidSMILESError,
    calculate_properties,
)

ASPIRIN_SMILES = "CC(=O)OC1=CC=CC=C1C(=O)O"
# PubChem CID 12560, canonical SMILES
ERYTHROMYCIN_SMILES = (
    "CCC1OC(=O)C(C)C(OC2CC(C)(OC)C(O)C(C)O2)C(C)C(OC3OC(C)CC(N(C)C)C3O)"
    "C(C)(O)CC(C)C(=O)C(C)C(O)C1(C)O"
)


def test_aspirin_passes_lipinski():
    props = calculate_properties(ASPIRIN_SMILES)
    # PubChem 공개값 기준: MW 180.16, LogP 1.31
    assert 179 <= props.molecular_weight <= 181
    assert props.lipinski_violations == 0
    assert props.passes_lipinski_ro5 is True


def test_erythromycin_violates_lipinski():
    props = calculate_properties(ERYTHROMYCIN_SMILES)
    # PubChem 공개값 기준: MW 733.93 — MW, HBA 두 항목에서 RO5 위반
    assert 730 <= props.molecular_weight <= 737
    assert props.lipinski_violations >= 2
    assert props.passes_lipinski_ro5 is False


def test_invalid_smiles_raises():
    with pytest.raises(InvalidSMILESError):
        calculate_properties("not-a-valid-smiles!!!")


def test_empty_smiles_raises():
    with pytest.raises(InvalidSMILESError):
        calculate_properties("")
