"""Unit tests for OllamaAdapter — no running server or MLX model required."""
import json
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Provide minimal stubs for heavy dependencies so we can import server.py
# without needing mlx_lm, mlx_embeddings, or an Apple Silicon machine.
# ---------------------------------------------------------------------------

def _make_stub_module(name):
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_stub(dotted_name):
    parts = dotted_name.split(".")
    for i in range(1, len(parts) + 1):
        full = ".".join(parts[:i])
        if full not in sys.modules:
            _make_stub_module(full)


for _mod in [
    "mlx_lm",
    "mlx_lm.server",
    "mlx_embeddings",
]:
    _ensure_stub(_mod)

# The stubs need the specific names that server.py imports from mlx_lm.server.
_mlx_server_stub = sys.modules["mlx_lm.server"]


class _FakeResponseGenerator:
    cli_args = None


class _FakeHandler:
    """Minimal stand-in for BaseHTTPRequestHandler."""
    def __init__(self, *a, **kw):
        pass


_mlx_server_stub.APIHandler = _FakeHandler
_mlx_server_stub.ModelProvider = object
_mlx_server_stub.LRUPromptCache = object
_mlx_server_stub.ResponseGenerator = _FakeResponseGenerator

# Now import the module under test.
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "server",
    os.path.join(os.path.dirname(__file__), "..", "server.py"),
)
_server_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_server_module)  # type: ignore[union-attr]

OllamaAdapter = _server_module.OllamaAdapter
_StreamTranslatorWfile = _server_module._StreamTranslatorWfile

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTranslateRequestGenerate(unittest.TestCase):
    """Test OllamaAdapter.translate_request with wrap_prompt=True (/api/generate)."""

    def test_prompt_becomes_user_message(self):
        body = {"model": "llama3", "prompt": "Hello world"}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertEqual(result["messages"], [{"role": "user", "content": "Hello world"}])

    def test_model_passed_through(self):
        body = {"model": "mistral", "prompt": "Hi"}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertEqual(result["model"], "mistral")

    def test_stream_defaults_to_true(self):
        body = {"model": "llama3", "prompt": "Hi"}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertTrue(result["stream"])

    def test_stream_false_preserved(self):
        body = {"model": "llama3", "prompt": "Hi", "stream": False}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertFalse(result["stream"])

    def test_empty_prompt_becomes_empty_content(self):
        body = {"model": "llama3"}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertEqual(result["messages"][0]["content"], "")


class TestTranslateRequestChat(unittest.TestCase):
    """Test OllamaAdapter.translate_request with wrap_prompt=False (/api/chat)."""

    def test_messages_passed_through(self):
        msgs = [{"role": "user", "content": "Hello"}]
        body = {"model": "llama3", "messages": msgs}
        result = OllamaAdapter.translate_request(body, wrap_prompt=False)
        self.assertEqual(result["messages"], msgs)

    def test_model_passed_through(self):
        body = {"model": "gemma", "messages": []}
        result = OllamaAdapter.translate_request(body, wrap_prompt=False)
        self.assertEqual(result["model"], "gemma")

    def test_missing_messages_not_injected(self):
        body = {"model": "gemma"}
        result = OllamaAdapter.translate_request(body, wrap_prompt=False)
        self.assertNotIn("messages", result)


class TestTranslateRequestOptions(unittest.TestCase):
    """Test options field mapping."""

    def test_num_predict_maps_to_max_tokens(self):
        body = {"model": "llama3", "prompt": "", "options": {"num_predict": 512}}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertEqual(result["max_tokens"], 512)

    def test_temperature_mapped(self):
        body = {"model": "llama3", "prompt": "", "options": {"temperature": 0.7}}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertAlmostEqual(result["temperature"], 0.7)

    def test_top_p_mapped(self):
        body = {"model": "llama3", "prompt": "", "options": {"top_p": 0.9}}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertAlmostEqual(result["top_p"], 0.9)

    def test_top_k_mapped(self):
        body = {"model": "llama3", "prompt": "", "options": {"top_k": 40}}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertEqual(result["top_k"], 40)

    def test_stop_mapped_from_options(self):
        body = {"model": "llama3", "prompt": "", "options": {"stop": ["</s>", "User:"]}}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertEqual(result["stop"], ["</s>", "User:"])

    def test_empty_options_no_extra_keys(self):
        body = {"model": "llama3", "prompt": "", "options": {}}
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        for key in ("max_tokens", "temperature", "top_p", "top_k", "stop"):
            self.assertNotIn(key, result)

    def test_all_options_at_once(self):
        body = {
            "model": "llama3",
            "prompt": "test",
            "options": {
                "num_predict": 100,
                "temperature": 0.5,
                "top_p": 0.8,
                "top_k": 20,
                "stop": ["\n"],
            },
        }
        result = OllamaAdapter.translate_request(body, wrap_prompt=True)
        self.assertEqual(result["max_tokens"], 100)
        self.assertAlmostEqual(result["temperature"], 0.5)
        self.assertAlmostEqual(result["top_p"], 0.8)
        self.assertEqual(result["top_k"], 20)
        self.assertEqual(result["stop"], ["\n"])


class TestOpenAIResponseToOllama(unittest.TestCase):
    """Test non-streaming response translation."""

    def _make_openai_resp(self, content="Hello!", finish_reason="stop"):
        return {
            "id": "chatcmpl-abc",
            "object": "chat.completion",
            "model": "llama3",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
        }

    def test_basic_translation(self):
        openai_resp = self._make_openai_resp("Hello!")
        ollama_resp = OllamaAdapter.openai_response_to_ollama(openai_resp, "llama3")
        self.assertEqual(ollama_resp["model"], "llama3")
        self.assertTrue(ollama_resp["done"])
        self.assertEqual(ollama_resp["done_reason"], "stop")
        self.assertEqual(ollama_resp["message"]["role"], "assistant")
        self.assertEqual(ollama_resp["message"]["content"], "Hello!")

    def test_created_at_present(self):
        openai_resp = self._make_openai_resp()
        ollama_resp = OllamaAdapter.openai_response_to_ollama(openai_resp, "llama3")
        self.assertIn("created_at", ollama_resp)
        self.assertIsInstance(ollama_resp["created_at"], str)

    def test_finish_reason_length(self):
        openai_resp = self._make_openai_resp("...", finish_reason="length")
        ollama_resp = OllamaAdapter.openai_response_to_ollama(openai_resp, "llama3")
        self.assertEqual(ollama_resp["done_reason"], "length")

    def test_empty_choices_graceful(self):
        openai_resp = {"choices": []}
        ollama_resp = OllamaAdapter.openai_response_to_ollama(openai_resp, "llama3")
        self.assertTrue(ollama_resp["done"])
        self.assertEqual(ollama_resp["message"]["content"], "")


class TestSSEChunkToOllama(unittest.TestCase):
    """Test streaming SSE chunk → Ollama NDJSON translation."""

    def _make_sse_data(self, content="token", finish_reason=None):
        choice: dict = {
            "index": 0,
            "delta": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }
        return json.dumps({"id": "chatcmpl-x", "choices": [choice]})

    def test_intermediate_chunk(self):
        data = self._make_sse_data("Hello")
        chunk = OllamaAdapter.openai_sse_chunk_to_ollama(data, "llama3")
        self.assertIsNotNone(chunk)
        self.assertFalse(chunk["done"])
        self.assertEqual(chunk["message"]["content"], "Hello")
        self.assertEqual(chunk["model"], "llama3")

    def test_final_chunk_with_finish_reason(self):
        data = self._make_sse_data("", finish_reason="stop")
        chunk = OllamaAdapter.openai_sse_chunk_to_ollama(data, "llama3")
        self.assertIsNotNone(chunk)
        self.assertTrue(chunk["done"])
        self.assertEqual(chunk["done_reason"], "stop")

    def test_done_sentinel_returns_none(self):
        result = OllamaAdapter.openai_sse_chunk_to_ollama("[DONE]", "llama3")
        self.assertIsNone(result)

    def test_invalid_json_returns_none(self):
        result = OllamaAdapter.openai_sse_chunk_to_ollama("not-json", "llama3")
        self.assertIsNone(result)

    def test_created_at_present(self):
        data = self._make_sse_data("tok")
        chunk = OllamaAdapter.openai_sse_chunk_to_ollama(data, "llama3")
        self.assertIn("created_at", chunk)


class TestStreamTranslatorWfile(unittest.TestCase):
    """Test the SSE→NDJSON wfile wrapper."""

    def _make_translator(self):
        buf = []

        class FakeWfile:
            def write(self, data):
                buf.append(data)
            def flush(self):
                pass

        translator = _StreamTranslatorWfile(FakeWfile(), "llama3")
        return translator, buf

    def _sse_line(self, content, finish_reason=None):
        choice = {
            "index": 0,
            "delta": {"content": content},
            "finish_reason": finish_reason,
        }
        payload = json.dumps({"choices": [choice]})
        return f"data: {payload}\n".encode()

    def test_single_token_chunk(self):
        translator, buf = self._make_translator()
        translator.write(self._sse_line("Hello"))
        self.assertEqual(len(buf), 1)
        chunk = json.loads(buf[0].decode())
        self.assertEqual(chunk["message"]["content"], "Hello")
        self.assertFalse(chunk["done"])

    def test_done_sentinel_not_emitted(self):
        translator, buf = self._make_translator()
        translator.write(b"data: [DONE]\n")
        self.assertEqual(len(buf), 0)

    def test_keepalive_comment_ignored(self):
        translator, buf = self._make_translator()
        translator.write(b": keepalive 10/100\n")
        self.assertEqual(len(buf), 0)

    def test_multiple_chunks_in_one_write(self):
        translator, buf = self._make_translator()
        data = self._sse_line("A") + self._sse_line("B")
        translator.write(data)
        self.assertEqual(len(buf), 2)
        self.assertEqual(json.loads(buf[0])["message"]["content"], "A")
        self.assertEqual(json.loads(buf[1])["message"]["content"], "B")

    def test_partial_write_buffered(self):
        """Partial lines should be held in buffer until newline arrives."""
        translator, buf = self._make_translator()
        full = self._sse_line("tok")
        # Write in two halves
        translator.write(full[:5])
        self.assertEqual(len(buf), 0)
        translator.write(full[5:])
        self.assertEqual(len(buf), 1)


if __name__ == "__main__":
    unittest.main()
