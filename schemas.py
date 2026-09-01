"""API 요청/응답 스키마 정의."""
from typing import Optional

from pydantic import BaseModel, Field


class MoleculePropertiesRequest(BaseModel):
    smiles: str = Field(..., description="분석할 분자의 SMILES 문자열", examples=["CC(=O)OC1=CC=CC=C1C(=O)O"])


class MoleculePropertiesResponse(BaseModel):
    smiles: str
    molecular_weight: float
    logp: float
    h_bond_donors: int
    h_bond_acceptors: int
    tpsa: float
    rotatable_bonds: int
    lipinski_violations: int
    passes_lipinski_ro5: bool
    violation_details: list[str]


class CompoundLookupRequest(BaseModel):
    name: str = Field(..., description="화합물명 (예: Aspirin, Imatinib)")


class CompoundLookupResponse(BaseModel):
    name: str
    cid: Optional[int] = None
    smiles: Optional[str] = None
    found: bool


class LiteratureSearchRequest(BaseModel):
    query: str = Field(..., description="PubMed 검색어", examples=["imatinib resistance mechanism"])
    max_results: int = Field(5, ge=1, le=20, description="가져올 최대 논문 수 (1~20)")


class PubMedArticleModel(BaseModel):
    pmid: str
    title: str
    abstract: str


class LiteratureSearchResponse(BaseModel):
    query: str
    articles: list[PubMedArticleModel]


class AgentQueryRequest(BaseModel):
    question: str = Field(..., description="에이전트에게 묻는 자연어 질문")


class ToolCallTrace(BaseModel):
    tool: str
    input: str
    output: str


class AgentQueryResponse(BaseModel):
    question: str
    answer: str
    tool_calls: list[ToolCallTrace]
