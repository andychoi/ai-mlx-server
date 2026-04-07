# tests/test_thinking.py
from server import _messages_to_prompt


def test_single_user_message():
    msgs = [{"role": "user", "content": "hello"}]
    assert _messages_to_prompt(msgs) == "user: hello"


def test_multi_turn():
    msgs = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "bye"},
    ]
    result = _messages_to_prompt(msgs)
    assert "user: hi" in result
    assert "assistant: hello" in result


def test_empty():
    assert _messages_to_prompt([]) == ""


def test_multimodal_content():
    msgs = [{"role": "user", "content": [
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
    ]}]
    result = _messages_to_prompt(msgs)
    assert result == "user: describe this"
