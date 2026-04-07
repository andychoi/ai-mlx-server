"""Unit tests for per-model ResponseGenerator worker registry.

These tests mock ResponseGenerator so no MLX model is required.
"""
import sys
import os
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_mock_rg():
    """Return a mock ResponseGenerator with a real queue-like requests attribute."""
    import queue
    rg = MagicMock()
    rg.requests = queue.Queue()
    return rg


class TestWorkerRegistry(unittest.TestCase):

    def setUp(self):
        """Reset the worker registry before each test."""
        import server
        with server._workers_lock:
            server._model_workers.clear()

    def test_get_or_create_worker_creates_on_first_call(self):
        """First call for a model creates and stores a new worker."""
        import server
        mock_rg = _make_mock_rg()

        with patch("server.ResponseGenerator", return_value=mock_rg), \
             patch("server.ModelProvider"), \
             patch("server.LRUPromptCache"):
            worker = server._get_or_create_worker("test-model")

        self.assertIs(worker, mock_rg)
        self.assertIn("test-model", server._model_workers)

    def test_get_or_create_worker_returns_same_worker_on_second_call(self):
        """Second call for the same model returns the existing worker."""
        import server
        mock_rg = _make_mock_rg()

        with patch("server.ResponseGenerator", return_value=mock_rg) as rg_cls, \
             patch("server.ModelProvider"), \
             patch("server.LRUPromptCache"):
            w1 = server._get_or_create_worker("test-model")
            w2 = server._get_or_create_worker("test-model")

        self.assertIs(w1, w2)
        self.assertEqual(rg_cls.call_count, 1)

    def test_tear_down_worker_removes_from_registry(self):
        """_tear_down_worker removes the model from _model_workers."""
        import server
        mock_rg = _make_mock_rg()
        with server._workers_lock:
            server._model_workers["my-model"] = mock_rg

        server._tear_down_worker("my-model")

        self.assertNotIn("my-model", server._model_workers)
        mock_rg.stop_and_join.assert_called_once()

    def test_tear_down_worker_noop_for_unknown_model(self):
        """_tear_down_worker is a no-op when the model has no worker."""
        import server
        # Should not raise
        server._tear_down_worker("nonexistent-model")

    def test_get_or_create_worker_uses_model_id_for_provider(self):
        """Worker creation sets args.model = model_id for ModelProvider."""
        import server
        import argparse
        fake_args = argparse.Namespace(
            model=None, trust_remote_code=False, chat_template=None,
            use_default_chat_template=False, adapter_path=None, draft_model=None,
            num_draft_tokens=3, temp=0.0, top_p=1.0, top_k=0, min_p=0.0,
            max_tokens=4096, chat_template_args={}, decode_concurrency=32,
            prompt_concurrency=2, prefill_step_size=512, prompt_cache_size=None,
            prompt_cache_bytes=None, pipeline=False,
        )
        server._server_args = fake_args

        mock_rg = _make_mock_rg()
        captured_provider_args = []

        def capture_mp(a):
            captured_provider_args.append(a.model)
            return MagicMock()

        with patch("server.ResponseGenerator", return_value=mock_rg), \
             patch("server.ModelProvider", side_effect=capture_mp), \
             patch("server.LRUPromptCache"):
            server._get_or_create_worker("some/model")

        self.assertEqual(captured_provider_args, ["some/model"])


if __name__ == "__main__":
    unittest.main()
