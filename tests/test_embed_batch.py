"""Unit tests for batched embeddings — no running server or MLX model required."""
import io
import json
import sys
import types
import unittest

# ---------------------------------------------------------------------------
# Stub heavy dependencies so server.py can be imported without mlx / Apple SI
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


for _mod in ["mlx_lm", "mlx_lm.server", "mlx_embeddings"]:
    _ensure_stub(_mod)

_mlx_server_stub = sys.modules["mlx_lm.server"]


class _FakeResponseGenerator:
    cli_args = None


class _FakeHandler:
    def __init__(self, *a, **kw):
        pass


_mlx_server_stub.APIHandler = _FakeHandler
_mlx_server_stub.ModelProvider = object
_mlx_server_stub.LRUPromptCache = object
_mlx_server_stub.ResponseGenerator = _FakeResponseGenerator

import importlib.util, os

_spec = importlib.util.spec_from_file_location(
    "server",
    os.path.join(os.path.dirname(__file__), "..", "server.py"),
)
_server_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_server_module)  # type: ignore[union-attr]

# ---------------------------------------------------------------------------
# Minimal mock infrastructure
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Mimics an mlx array returned by tokenizer.encode(return_tensors='mlx').

    Shape is (1, seq_len) — the last dim is the token count.
    """

    def __init__(self, token_ids):
        self._ids = token_ids

    @property
    def shape(self):
        return (1, len(self._ids))

    def __len__(self):
        return len(self._ids)


class _FakeEmbedTensor:
    """Wraps a plain list so .tolist() works like an mlx array."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


class _FakeOutputs:
    def __init__(self, embedding):
        self.text_embeds = [_FakeEmbedTensor(embedding)]


class _MockTokenizer:
    """Returns a _FakeTensor whose length equals the number of words in text."""

    def encode(self, text, return_tensors=None):
        # Simple word-count tokenizer for deterministic tests
        token_ids = list(range(len(text.split())))
        if return_tensors == "mlx":
            return _FakeTensor(token_ids)
        return token_ids


class _MockModel:
    """Returns a fixed embedding vector of [0.1, 0.2, 0.3]."""

    def __call__(self, input_ids):
        return _FakeOutputs([0.1, 0.2, 0.3])


def _make_handler(body_dict):
    """Return a minimal MLXAPIHandler instance wired with mock I/O."""
    raw = json.dumps(body_dict).encode()

    class FakeHeaders(dict):
        def get(self, key, default=None):
            return self[key] if key in self else default

    headers = FakeHeaders()
    headers["Content-Length"] = str(len(raw))

    responses = []

    class MinimalHandler(_server_module.MLXAPIHandler):
        def __init__(self):
            # Skip BaseHTTPRequestHandler.__init__
            self.headers = headers
            self.rfile = io.BytesIO(raw)
            self._responses = responses

        def _json_response(self, status, payload):
            self._responses.append((status, payload))

    return MinimalHandler(), responses


def _patch_embed_model(mock_model, mock_tokenizer):
    """Context manager that patches _get_embed_model to return our mocks."""
    import contextlib

    @contextlib.contextmanager
    def _ctx():
        original = _server_module._get_embed_model
        _server_module._get_embed_model = lambda path: (mock_model, mock_tokenizer)
        try:
            yield
        finally:
            _server_module._get_embed_model = original

    return _ctx()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSingleStringInput(unittest.TestCase):
    def test_returns_one_item_at_index_0(self):
        body = {"model": "mymodel", "input": "hello world"}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        self.assertEqual(len(responses), 1)
        status, payload = responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(payload["object"], "list")
        self.assertEqual(len(payload["data"]), 1)
        self.assertEqual(payload["data"][0]["index"], 0)
        self.assertEqual(payload["data"][0]["object"], "embedding")
        self.assertEqual(payload["data"][0]["embedding"], [0.1, 0.2, 0.3])


class TestListInput(unittest.TestCase):
    def test_three_strings_return_three_items(self):
        texts = ["one", "two three", "four five six"]
        body = {"model": "mymodel", "input": texts}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        status, payload = responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["data"]), 3)

    def test_indices_are_correct(self):
        texts = ["a", "b c", "d e f"]
        body = {"model": "mymodel", "input": texts}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        _, payload = responses[0]
        indices = [item["index"] for item in payload["data"]]
        self.assertEqual(indices, [0, 1, 2])

    def test_each_item_has_embedding(self):
        texts = ["hello", "world"]
        body = {"model": "mymodel", "input": texts}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        _, payload = responses[0]
        for item in payload["data"]:
            self.assertIn("embedding", item)
            self.assertIsInstance(item["embedding"], list)


class TestUsageTokenCount(unittest.TestCase):
    """prompt_tokens must reflect actual token counts, not len(text) // 4."""

    def test_token_count_is_sum_of_word_counts(self):
        # "hello world" → 2 tokens, "foo bar baz" → 3 tokens → total 5
        texts = ["hello world", "foo bar baz"]
        body = {"model": "mymodel", "input": texts}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        _, payload = responses[0]
        self.assertEqual(payload["usage"]["prompt_tokens"], 5)
        self.assertEqual(payload["usage"]["total_tokens"], 5)

    def test_single_string_token_count(self):
        # "one two three" → 3 tokens; len("one two three") // 4 == 3 coincidence,
        # use a longer phrase to distinguish.
        text = "alpha beta gamma delta"  # 4 tokens; len=22 → 22//4=5
        body = {"model": "mymodel", "input": text}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        _, payload = responses[0]
        self.assertEqual(payload["usage"]["prompt_tokens"], 4)


class TestEmptyInput(unittest.TestCase):
    def test_empty_string_returns_400(self):
        body = {"model": "mymodel", "input": ""}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        status, payload = responses[0]
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_empty_list_returns_400(self):
        body = {"model": "mymodel", "input": []}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        status, payload = responses[0]
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_missing_input_returns_400(self):
        body = {"model": "mymodel"}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        status, _ = responses[0]
        self.assertEqual(status, 400)

    def test_missing_model_returns_400(self):
        body = {"input": "hello"}
        handler, responses = _make_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_embeddings()
        status, _ = responses[0]
        self.assertEqual(status, 400)


class TestOllamaEmbeddingsPath(unittest.TestCase):
    """_handle_ollama_embeddings remaps 'prompt' → 'input' then calls _handle_embeddings."""

    def _make_ollama_handler(self, body_dict):
        """Wire a handler whose _read_body returns the given dict encoded."""
        raw = json.dumps(body_dict).encode()

        responses = []

        class FakeHeaders(dict):
            def get(self, key, default=None):
                return self[key] if key in self else default

        headers = FakeHeaders()
        headers["Content-Length"] = str(len(raw))

        class OllamaHandler(_server_module.MLXAPIHandler):
            def __init__(self):
                self.headers = headers
                self.rfile = io.BytesIO(raw)
                self._responses = responses

            def _read_body(self):
                cl = int(self.headers.get("Content-Length", 0))
                return self.rfile.read(cl)

            def _json_response(self, status, payload):
                self._responses.append((status, payload))

        return OllamaHandler(), responses

    def test_prompt_key_is_remapped(self):
        body = {"model": "mymodel", "prompt": "hello world"}
        handler, responses = self._make_ollama_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_ollama_embeddings()
        status, payload = responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["data"]), 1)
        self.assertEqual(payload["data"][0]["index"], 0)

    def test_prompt_list_remapped(self):
        body = {"model": "mymodel", "prompt": ["hello", "world"]}
        handler, responses = self._make_ollama_handler(body)
        with _patch_embed_model(_MockModel(), _MockTokenizer()):
            handler._handle_ollama_embeddings()
        status, payload = responses[0]
        self.assertEqual(status, 200)
        self.assertEqual(len(payload["data"]), 2)


if __name__ == "__main__":
    unittest.main()
