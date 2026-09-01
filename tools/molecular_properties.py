"""RDKit 기반 분자 물성 계산 (Cheminformatics 기초 tool).

SMILES 문자열을 입력받아 신약개발에서 흔히 쓰이는 기본 물성 지표와
Lipinski's Rule of Five(경구 흡수 가능성 경험칙) 통과 여부를 계산한다.

의도적으로 도킹(docking)이나 3D 구조 예측처럼 계산 비용이 크고
깊은 전문성이 필요한 영역은 다루지 않는다 — 이 프로젝트에서 이 부분은
"기초적인 cheminformatics 툴 활용 경험"을 보여주는 용도로 한정한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors


class InvalidSMILESError(ValueError):
    """RDKit이 파싱하지 못하는 SMILES 문자열일 때 발생."""


@dataclass
class MoleculeProperties:
    smiles: str
    molecular_weight: float
    logp: float
    h_bond_donors: int
    h_bond_acceptors: int
    tpsa: float
    rotatable_bonds: int
    lipinski_violations: int
    passes_lipinski_ro5: bool
    violation_details: list[str] = field(default_factory=list)


def calculate_properties(smiles: str) -> MoleculeProperties:
    """SMILES로부터 분자 물성을 계산한다.

    Raises:
        InvalidSMILESError: RDKit이 분자를 파싱할 수 없는 경우.
    """
    if not smiles or not smiles.strip():
        raise InvalidSMILESError("빈 SMILES 문자열입니다.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSMILESError(f"RDKit이 파싱할 수 없는 SMILES입니다: {smiles!r}")

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    rot_bonds = Descriptors.NumRotatableBonds(mol)

    violations: list[str] = []
    if mw > 500:
        violations.append(f"분자량 {mw:.1f} > 500")
    if logp > 5:
        violations.append(f"LogP {logp:.2f} > 5")
    if hbd > 5:
        violations.append(f"수소 결합 공여체 {hbd} > 5")
    if hba > 10:
        violations.append(f"수소 결합 수용체 {hba} > 10")

    return MoleculeProperties(
        smiles=smiles,
        molecular_weight=round(mw, 2),
        logp=round(logp, 2),
        h_bond_donors=hbd,
        h_bond_acceptors=hba,
        tpsa=round(tpsa, 2),
        rotatable_bonds=rot_bonds,
        lipinski_violations=len(violations),
        passes_lipinski_ro5=len(violations) <= 1,  # Lipinski 원 규칙: 위반 1개까지는 허용
        violation_details=violations,
    )


def format_for_agent(props: MoleculeProperties) -> str:
    """LLM 에이전트가 tool 응답으로 바로 쓸 수 있는 자연어 요약."""
    status = "통과" if props.passes_lipinski_ro5 else "위반"
    lines = [
        f"SMILES: {props.smiles}",
        f"분자량(MW): {props.molecular_weight} g/mol",
        f"LogP: {props.logp}",
        f"수소결합 공여체(HBD): {props.h_bond_donors}",
        f"수소결합 수용체(HBA): {props.h_bond_acceptors}",
        f"TPSA: {props.tpsa} Å²",
        f"회전 가능 결합 수: {props.rotatable_bonds}",
        f"Lipinski's Rule of Five: {status} (위반 {props.lipinski_violations}개)",
    ]
    if props.violation_details:
        lines.append("위반 상세: " + "; ".join(props.violation_details))
    return "\n".join(lines)
