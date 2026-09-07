"""literature_search.py 단위 테스트. 실제 PubMed(NCBI Entrez) 네트워크 호출 없이
Entrez.esearch/efetch/read를 모킹한다.

/literature/search 엔드포인트를 신규 추가(2026-09-01)하면서 함께 추가한 테스트 —
compound_lookup 때(PubChem 응답 키 변경으로 KeyError → 500)와 같은 종류의 문제가
이 tool에서도 조용히 서버를 죽이지 않는지 확인하는 게 목적이다.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from tools.literature_search import (
    LiteratureSearchError,
    format_for_agent,
    search_literature,
)


def _mock_handle():
    """Entrez.esearch/efetch가 반환하는 'with ... as handle' 컨텍스트 매니저를 흉내낸다."""
    handle = MagicMock()
    handle.__enter__.return_value = handle
    handle.__exit__.return_value = False
    return handle


@patch.dict(os.environ, {"ENTREZ_EMAIL": "test@example.com"})
@patch("tools.literature_search.Entrez.read")
@patch("tools.literature_search.Entrez.efetch")
@patch("tools.literature_search.Entrez.esearch")
def test_search_literature_returns_parsed_articles(mock_esearch, mock_efetch, mock_read):
    mock_esearch.return_value = _mock_handle()
    mock_efetch.return_value = _mock_handle()
    mock_read.side_effect = [
        {"IdList": ["42663066"]},
        {
            "PubmedArticle": [
                {
                    "MedlineCitation": {
                        "PMID": "42663066",
                        "Article": {
                            "ArticleTitle": "Imatinib resistance mechanisms in CML",
                            "Abstract": {"AbstractText": ["BCR-ABL 관련 내성 기전을 다룬다."]},
                        },
                    }
                }
            ]
        },
    ]

    articles = search_literature("imatinib resistance mechanism")

    assert len(articles) == 1
    assert articles[0].pmid == "42663066"
    assert articles[0].title == "Imatinib resistance mechanisms in CML"
    assert "내성 기전" in articles[0].abstract


@patch.dict(os.environ, {"ENTREZ_EMAIL": "test@example.com"})
@patch("tools.literature_search.Entrez.read")
@patch("tools.literature_search.Entrez.efetch")
@patch("tools.literature_search.Entrez.esearch")
def test_search_literature_multiple_abstract_parts_are_joined(mock_esearch, mock_efetch, mock_read):
    """AbstractText가 여러 단락으로 쪼개져 오는 경우(실제 PubMed에서 흔함)도 하나로 합쳐져야 한다."""
    mock_esearch.return_value = _mock_handle()
    mock_efetch.return_value = _mock_handle()
    mock_read.side_effect = [
        {"IdList": ["1"]},
        {
            "PubmedArticle": [
                {
                    "MedlineCitation": {
                        "PMID": "1",
                        "Article": {
                            "ArticleTitle": "Title",
                            "Abstract": {"AbstractText": ["Background.", "Methods.", "Results."]},
                        },
                    }
                }
            ]
        },
    ]

    articles = search_literature("query")

    assert articles[0].abstract == "Background. Methods. Results."


@patch.dict(os.environ, {"ENTREZ_EMAIL": "test@example.com"})
@patch("tools.literature_search.Entrez.read")
@patch("tools.literature_search.Entrez.efetch")
@patch("tools.literature_search.Entrez.esearch")
def test_search_literature_strips_inline_markup_tags(mock_esearch, mock_efetch, mock_read):
    """PubMed 원본 XML에 섞여 오는 <i>, <sup> 같은 인라인 마크업 태그가 제거되어야 한다
    (2026-09-07 실제 발견: "PI3<i>K</i>/110β" 형태로 제목에 태그가 그대로 노출되던 문제)."""
    mock_esearch.return_value = _mock_handle()
    mock_efetch.return_value = _mock_handle()
    mock_read.side_effect = [
        {"IdList": ["40294240"]},
        {
            "PubmedArticle": [
                {
                    "MedlineCitation": {
                        "PMID": "40294240",
                        "Article": {
                            "ArticleTitle": "Novel Selective PI3<i>K</i>/110β PROTAC Degraders",
                            "Abstract": {
                                "AbstractText": ["The p110<sup>β</sup> isoform plays a key role."]
                            },
                        },
                    }
                }
            ]
        },
    ]

    articles = search_literature("query")

    assert articles[0].title == "Novel Selective PI3K/110β PROTAC Degraders"
    assert articles[0].abstract == "The p110β isoform plays a key role."


@patch.dict(os.environ, {"ENTREZ_EMAIL": "test@example.com"})
@patch("tools.literature_search.Entrez.read")
@patch("tools.literature_search.Entrez.esearch")
def test_search_literature_no_results_returns_empty_list(mock_esearch, mock_read):
    """검색 결과가 0건이면 efetch는 아예 호출하지 않고 빈 리스트를 반환해야 한다."""
    mock_esearch.return_value = _mock_handle()
    mock_read.return_value = {"IdList": []}

    articles = search_literature("존재하지 않을 법한 질의")

    assert articles == []


@patch.dict(os.environ, {"ENTREZ_EMAIL": "test@example.com"})
@patch("tools.literature_search.Entrez.read")
@patch("tools.literature_search.Entrez.efetch")
@patch("tools.literature_search.Entrez.esearch")
def test_search_literature_malformed_article_is_skipped_not_raised(mock_esearch, mock_efetch, mock_read):
    """응답 구조가 예상과 다른 항목(키 누락 등)이 섞여 있어도 KeyError로 죽지 않고
    그 항목만 건너뛰고 나머지는 정상 반환해야 한다 (compound_lookup 버그와 같은 종류의 회귀 방지)."""
    mock_esearch.return_value = _mock_handle()
    mock_efetch.return_value = _mock_handle()
    mock_read.side_effect = [
        {"IdList": ["1", "2"]},
        {
            "PubmedArticle": [
                {"MedlineCitation": {"PMID": "1"}},  # Article 키 자체가 없는 비정상 항목
                {
                    "MedlineCitation": {
                        "PMID": "2",
                        "Article": {
                            "ArticleTitle": "정상 논문",
                            "Abstract": {"AbstractText": ["정상 초록."]},
                        },
                    }
                },
            ]
        },
    ]

    articles = search_literature("query")

    assert len(articles) == 1
    assert articles[0].pmid == "2"
    assert articles[0].title == "정상 논문"


@patch.dict(os.environ, {"ENTREZ_EMAIL": "test@example.com"})
@patch("tools.literature_search.Entrez.esearch")
def test_search_literature_esearch_error_wrapped(mock_esearch):
    mock_esearch.side_effect = Exception("network unreachable")

    with pytest.raises(LiteratureSearchError):
        search_literature("query")


@patch.dict(os.environ, {}, clear=True)
def test_search_literature_missing_entrez_email_raises():
    with pytest.raises(LiteratureSearchError):
        search_literature("query")


def test_format_for_agent_empty_list():
    assert format_for_agent([]) == "검색 결과가 없습니다."


def test_format_for_agent_truncates_long_abstract():
    from tools.literature_search import PubMedArticle

    long_abstract = "A" * 600
    article = PubMedArticle(pmid="1", title="T", abstract=long_abstract)

    formatted = format_for_agent([article])

    assert "[PMID 1] T" in formatted
    assert formatted.count("A") == 500
    assert formatted.endswith("...")
