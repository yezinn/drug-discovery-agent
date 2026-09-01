"""agent.py의 순수 헬퍼 함수 단위 테스트.

_extract_answer_text()는 LangGraph/Gemini 호출 없이도 테스트 가능한 순수 함수라
네트워크나 GOOGLE_API_KEY 없이 검증한다. Gemini가 content를 str이 아니라
블록 리스트로 반환해서 answer 필드가 "[{'type': 'text', ...}]"처럼 지저분하게
나가던 버그(2026-09-01)의 회귀 테스트.
"""
from agent import _extract_answer_text


def test_plain_string_content_returned_as_is():
    assert _extract_answer_text("단순 문자열 답변") == "단순 문자열 답변"


def test_gemini_style_block_list_extracts_text_only():
    content = [
        {
            "type": "text",
            "text": "이매티닙은 Rule of Five를 통과합니다.",
            "extras": {"signature": "abc123"},
        }
    ]
    assert _extract_answer_text(content) == "이매티닙은 Rule of Five를 통과합니다."


def test_multiple_text_blocks_are_joined():
    content = [
        {"type": "text", "text": "첫 번째 블록."},
        {"type": "text", "text": "두 번째 블록."},
    ]
    assert _extract_answer_text(content) == "첫 번째 블록.\n두 번째 블록."


def test_non_text_blocks_are_ignored():
    content = [
        {"type": "tool_use", "id": "call_1", "name": "some_tool"},
        {"type": "text", "text": "실제 답변만 남아야 함."},
    ]
    assert _extract_answer_text(content) == "실제 답변만 남아야 함."


def test_unexpected_structure_falls_back_to_str():
    # text 블록이 하나도 없는 예상 밖 구조 — 죽지 않고 최소한 str(content)라도 반환
    content = [{"type": "unknown", "data": 123}]
    assert _extract_answer_text(content) == str(content)


def test_empty_list_falls_back_to_str():
    assert _extract_answer_text([]) == "[]"
