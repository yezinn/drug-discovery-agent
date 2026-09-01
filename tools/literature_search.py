"""PubMed 실시간 문헌 검색 tool (Biopython Entrez 기반).

기존 pubmed-rag-qa 프로젝트는 "미리 수집한 고정 코퍼스 + 벡터DB"로 답하는
오프라인 RAG였다면, 이 tool은 에이전트가 대화 중 필요하다고 판단할 때마다
그때그때 PubMed를 실시간으로 검색하는 용도다. 검색 결과를 임베딩/재정렬하지
않고 상위 N개 초록을 그대로 LLM에게 넘겨 요약·판단하게 한다 — 단일 질의에
벡터DB를 새로 구축하는 건 과설계라고 판단해 의도적으로 단순화한 설계.

주의: NCBI E-utilities 호출은 클라우드 작업공간 네트워크 정책상 여기서
실제 호출 테스트를 하지 못했다 (pubmed-rag-qa-project-plan.md §6에 기록된
것과 동일한 제약). 로컬에서 재검증 필요.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from Bio import Entrez


class LiteratureSearchError(Exception):
    """PubMed 검색/조회 실패 시 발생."""


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str


def _ensure_entrez_email() -> None:
    email = os.environ.get("ENTREZ_EMAIL")
    if not email:
        raise LiteratureSearchError(
            "ENTREZ_EMAIL 환경변수가 설정되어 있지 않습니다. .env 참고."
        )
    Entrez.email = email


def search_literature(query: str, max_results: int = 5) -> list[PubMedArticle]:
    """PubMed에서 query로 검색해 상위 max_results개 논문의 제목/초록을 가져온다."""
    _ensure_entrez_email()

    try:
        with Entrez.esearch(db="pubmed", term=query, retmax=max_results) as handle:
            search_result = Entrez.read(handle)
    except Exception as e:  # noqa: BLE001 - Entrez가 다양한 예외를 던짐
        raise LiteratureSearchError(f"PubMed 검색(esearch) 중 오류: {e}") from e

    pmids = search_result.get("IdList", [])
    if not pmids:
        return []

    try:
        with Entrez.efetch(db="pubmed", id=pmids, rettype="abstract", retmode="xml") as handle:
            fetch_result = Entrez.read(handle)
    except Exception as e:  # noqa: BLE001
        raise LiteratureSearchError(f"PubMed 초록 조회(efetch) 중 오류: {e}") from e

    articles: list[PubMedArticle] = []
    for pubmed_article in fetch_result.get("PubmedArticle", []):
        try:
            medline = pubmed_article["MedlineCitation"]
            pmid = str(medline["PMID"])
            article_data = medline["Article"]
            title = str(article_data.get("ArticleTitle", ""))
            abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(part) for part in abstract_parts)
        except (KeyError, TypeError):
            continue
        articles.append(PubMedArticle(pmid=pmid, title=title, abstract=abstract))

    return articles


def format_for_agent(articles: list[PubMedArticle]) -> str:
    if not articles:
        return "검색 결과가 없습니다."
    blocks = []
    for a in articles:
        snippet = a.abstract[:500] + ("..." if len(a.abstract) > 500 else "")
        blocks.append(f"[PMID {a.pmid}] {a.title}\n{snippet}")
    return "\n\n".join(blocks)
