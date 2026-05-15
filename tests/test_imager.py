"""Unit tests for imager.py — mflux stubbed, no Apple Silicon required."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import types
import unittest
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Stub heavy deps before imports.
# server.py imports from mlx_lm.server and mlx_embeddings; imager.py imports
# mflux lazily inside _import_mflux_qwen_image(), but tests below replace that
# function directly so a stub here is only needed if a test forgets to patch.
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


def _ensure_stub(dotted: str):
    parts = dotted.split(".")
    for i in range(1, len(parts) + 1):
        full = ".".join(parts[:i])
        if full not in sys.modules:
            _make_stub(full)


for _m in ["mlx_lm", "mlx_lm.server", "mlx_embeddings"]:
    _ensure_stub(_m)

_mlx_server_stub = sys.modules["mlx_lm.server"]


class _FakeRG:
    cli_args = None


class _FakeHandler:
    def __init__(self, *a, **kw):
        pass


_mlx_server_stub.APIHandler = _FakeHandler
_mlx_server_stub.ModelProvider = object
_mlx_server_stub.LRUPromptCache = object
_mlx_server_stub.ResponseGenerator = _FakeRG

# Import the modules under test.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import imager  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_args(**overrides) -> "imager.argparse.Namespace":
    args = imager._build_argparser().parse_args([])
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


class _FakePILImage:
    """Mimic PIL.Image.Image.save(buf, format=...) interface."""
    def __init__(self, payload: bytes = b"\x89PNG\r\n\x1a\nfakeimagebytes"):
        self.payload = payload
        self.save_calls: list[dict] = []

    def save(self, buf, format=None, **kwargs):
        self.save_calls.append({"format": format, **kwargs})
        buf.write(self.payload)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseSize(unittest.TestCase):
    def test_auto_returns_defaults(self):
        self.assertEqual(imager._parse_size("auto", 1024, 768, 4_000_000), (1024, 768))

    def test_none_returns_defaults(self):
        self.assertEqual(imager._parse_size(None, 1024, 1024, 4_000_000), (1024, 1024))

    def test_valid_size(self):
        self.assertEqual(imager._parse_size("768x512", 1024, 1024, 4_000_000), (768, 512))

    def test_non_multiple_of_8_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._parse_size("1023x1024", 1024, 1024, 4_000_000)

    def test_garbage_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._parse_size("abc", 1024, 1024, 4_000_000)

    def test_too_large_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._parse_size("3000x3000", 1024, 1024, 4_000_000)

    def test_too_small_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._parse_size("32x32", 1024, 1024, 4_000_000)

    def test_exceeds_max_pixels(self):
        with self.assertRaises(imager._BadRequest):
            imager._parse_size("2048x2048", 1024, 1024, 2_097_152)


class TestValidateRequest(unittest.TestCase):
    def setUp(self):
        self.args = _default_args(model="mlx-community/Qwen-Image-2512-4bit")

    def test_missing_prompt_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_request({}, self.args)

    def test_empty_prompt_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_request({"prompt": "   "}, self.args)

    def test_n_exceeds_max_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_request({"prompt": "x", "n": 99}, self.args)

    def test_n_zero_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_request({"prompt": "x", "n": 0}, self.args)

    def test_response_format_url_rejected(self):
        with self.assertRaises(imager._BadRequest) as ctx:
            imager._validate_request({"prompt": "x", "response_format": "url"}, self.args)
        self.assertIn("b64_json", str(ctx.exception))

    def test_response_format_other_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_request({"prompt": "x", "response_format": "jpeg"}, self.args)

    def test_steps_out_of_range_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_request({"prompt": "x", "steps": 999}, self.args)

    def test_guidance_out_of_range_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_request({"prompt": "x", "guidance_scale": 50}, self.args)

    def test_negative_prompt_must_be_string(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_request({"prompt": "x", "negative_prompt": 123}, self.args)

    def test_seed_must_be_int(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_request({"prompt": "x", "seed": "abc"}, self.args)

    def test_valid_request_returns_normalized(self):
        params = imager._validate_request(
            {"prompt": "cat", "n": 2, "size": "768x768", "seed": 42, "steps": 8},
            self.args,
        )
        self.assertEqual(params["prompt"], "cat")
        self.assertEqual(params["n"], 2)
        self.assertEqual(params["width"], 768)
        self.assertEqual(params["height"], 768)
        self.assertEqual(params["seed"], 42)
        self.assertEqual(params["steps"], 8)
        self.assertEqual(params["model"], "mlx-community/Qwen-Image-2512-4bit")

    def test_model_override(self):
        params = imager._validate_request(
            {"prompt": "x", "model": "black-forest-labs/FLUX.1-schnell"}, self.args
        )
        self.assertEqual(params["model"], "black-forest-labs/FLUX.1-schnell")


class TestEncodeImage(unittest.TestCase):
    def test_png_round_trip(self):
        img = _FakePILImage(b"\x89PNG\r\n\x1a\nABC")
        b64 = imager._encode_image(img, "png", 92)
        self.assertEqual(base64.b64decode(b64), b"\x89PNG\r\n\x1a\nABC")
        self.assertEqual(img.save_calls[0]["format"], "PNG")
        self.assertTrue(img.save_calls[0].get("optimize"))

    def test_jpeg_uses_quality(self):
        img = _FakePILImage(b"jpegbytes")
        imager._encode_image(img, "jpeg", 80)
        self.assertEqual(img.save_calls[0]["format"], "JPEG")
        self.assertEqual(img.save_calls[0]["quality"], 80)


class TestExtractPilImage(unittest.TestCase):
    def test_returns_image_attribute_when_present(self):
        img = _FakePILImage()
        wrapper = types.SimpleNamespace(image=img)
        self.assertIs(imager._extract_pil_image(wrapper), img)

    def test_returns_object_itself_when_it_has_save(self):
        img = _FakePILImage()
        self.assertIs(imager._extract_pil_image(img), img)

    def test_rejects_unrecognized_object(self):
        with self.assertRaises(RuntimeError):
            imager._extract_pil_image(object())


class TestGenerateImageDispatch(unittest.TestCase):
    """Verify _generate_image() forwards the correct kwargs to mflux's generate_image."""

    def test_kwargs_match_mflux_signature(self):
        recorded: dict = {}

        class FakeModel:
            def generate_image(self, **kw):
                recorded.update(kw)
                return _FakePILImage()

        result = imager._generate_image(
            FakeModel(),
            prompt="a cat",
            negative_prompt="blurry",
            width=512,
            height=512,
            steps=4,
            guidance=3.5,
            seed=7,
        )
        self.assertEqual(recorded["prompt"], "a cat")
        self.assertEqual(recorded["negative_prompt"], "blurry")
        self.assertEqual(recorded["width"], 512)
        self.assertEqual(recorded["height"], 512)
        self.assertEqual(recorded["num_inference_steps"], 4)
        self.assertEqual(recorded["guidance"], 3.5)
        self.assertEqual(recorded["seed"], 7)
        self.assertIsInstance(result, _FakePILImage)


class TestMultipartParser(unittest.TestCase):
    """Verify _parse_multipart correctly extracts text + file fields."""

    @staticmethod
    def _build_multipart(fields: list[tuple]) -> tuple[str, bytes]:
        """Build a minimal multipart/form-data body. fields = [(name, value, filename_or_None), ...]"""
        boundary = "----testboundary123"
        parts = []
        for name, value, filename in fields:
            disp = f'form-data; name="{name}"'
            if filename:
                disp += f'; filename="{filename}"'
            headers = f"Content-Disposition: {disp}\r\n"
            if filename:
                headers += "Content-Type: application/octet-stream\r\n"
            headers += "\r\n"
            payload = value if isinstance(value, bytes) else value.encode("utf-8")
            parts.append(f"--{boundary}\r\n".encode() + headers.encode() + payload + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        return f"multipart/form-data; boundary={boundary}", b"".join(parts)

    def test_text_and_file_fields(self):
        ct, body = self._build_multipart([
            ("prompt", "make it blue", None),
            ("image", b"\x89PNG\r\n\x1a\nfakebytes", "input.png"),
            ("n", "2", None),
        ])
        form = imager._parse_multipart(ct, body)
        self.assertEqual(form["prompt"], "make it blue")
        self.assertEqual(form["image"], b"\x89PNG\r\n\x1a\nfakebytes")
        self.assertEqual(form["n"], "2")

    def test_rejects_non_multipart_content_type(self):
        with self.assertRaises(imager._BadRequest):
            imager._parse_multipart("application/json", b"{}")


class TestValidateEditRequest(unittest.TestCase):
    def setUp(self):
        self.args = _default_args()

    def test_requires_prompt(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_edit_request({"image": b"\x89PNG\r\n"}, self.args)

    def test_requires_image_bytes(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_edit_request({"prompt": "x"}, self.args)

    def test_rejects_empty_image(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_edit_request({"prompt": "x", "image": b""}, self.args)

    def test_rejects_url_response_format(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_edit_request(
                {"prompt": "x", "image": b"abc", "response_format": "url"}, self.args
            )

    def test_size_optional(self):
        params = imager._validate_edit_request(
            {"prompt": "x", "image": b"abc"}, self.args
        )
        # When size is omitted, width/height stay None so mflux uses the source image's dimensions.
        self.assertIsNone(params["width"])
        self.assertIsNone(params["height"])

    def test_size_validation_when_provided(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_edit_request(
                {"prompt": "x", "image": b"abc", "size": "1023x1024"}, self.args
            )

    def test_n_string_coerced(self):
        params = imager._validate_edit_request(
            {"prompt": "x", "image": b"abc", "n": "3"}, self.args
        )
        self.assertEqual(params["n"], 3)

    def test_n_bad_string_rejected(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_edit_request(
                {"prompt": "x", "image": b"abc", "n": "many"}, self.args
            )

    def test_steps_clamped_to_max(self):
        with self.assertRaises(imager._BadRequest):
            imager._validate_edit_request(
                {"prompt": "x", "image": b"abc", "steps": str(self.args.max_steps + 1)}, self.args
            )

    def test_seed_string_coerced(self):
        params = imager._validate_edit_request(
            {"prompt": "x", "image": b"abc", "seed": "42"}, self.args
        )
        self.assertEqual(params["seed"], 42)

    def test_returns_default_edit_model(self):
        params = imager._validate_edit_request({"prompt": "x", "image": b"abc"}, self.args)
        self.assertEqual(params["model"], self.args.edit_model)

    def test_model_override_via_form(self):
        params = imager._validate_edit_request(
            {"prompt": "x", "image": b"abc", "model": "custom/edit-model"}, self.args
        )
        self.assertEqual(params["model"], "custom/edit-model")


class TestGenerateEditDispatch(unittest.TestCase):
    """Verify _generate_edit forwards image_paths and the right kwargs."""

    def test_kwargs_match_qwen_image_edit_signature(self):
        recorded: dict = {}

        class FakeEditModel:
            def generate_image(self, **kw):
                recorded.update(kw)
                return _FakePILImage()

        imager._generate_edit(
            FakeEditModel(),
            prompt="make it blue",
            negative_prompt="",
            image_paths=["/tmp/in.png"],
            width=None,
            height=None,
            steps=8,
            guidance=4.0,
            seed=11,
        )
        self.assertEqual(recorded["prompt"], "make it blue")
        self.assertEqual(recorded["image_paths"], ["/tmp/in.png"])
        self.assertEqual(recorded["num_inference_steps"], 8)
        self.assertIsNone(recorded["width"])
        self.assertEqual(recorded["seed"], 11)


class TestConcurrency(unittest.TestCase):
    """Two requests serialize through _gen_lock without deadlock."""

    def test_gen_lock_serializes(self):
        order: list[str] = []

        def worker(name: str, delay: float):
            with imager._gen_lock:
                order.append(f"{name}:start")
                # Spin briefly to simulate generation
                t0 = imager.time.time()
                while imager.time.time() - t0 < delay:
                    pass
                order.append(f"{name}:end")

        t1 = threading.Thread(target=worker, args=("A", 0.05))
        t2 = threading.Thread(target=worker, args=("B", 0.05))
        t1.start(); t2.start()
        t1.join(); t2.join()
        # Whichever started first must finish before the other starts.
        self.assertEqual(len(order), 4)
        self.assertTrue(order[1].endswith(":end"))
        self.assertTrue(order[2].endswith(":start"))


class TestResolveModelPathPassthrough(unittest.TestCase):
    def test_qwen_image_repo_passes_through(self):
        from server import _resolve_model_path
        # An org/repo with no aliases should pass through unchanged.
        self.assertEqual(
            _resolve_model_path("mlx-community/Qwen-Image-2512-4bit"),
            "mlx-community/Qwen-Image-2512-4bit",
        )


if __name__ == "__main__":
    unittest.main()
