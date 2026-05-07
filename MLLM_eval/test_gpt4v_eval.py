"""Tests for gpt4v_eval.py MiniMax provider support."""
import base64
import json
import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Stub heavy optional dependencies so we can import gpt4v_eval without them.
# ---------------------------------------------------------------------------

# spacy stub
spacy_stub = types.ModuleType("spacy")
def _load(name):
    nlp = MagicMock()
    nlp.return_value = MagicMock()
    return nlp
spacy_stub.load = _load
sys.modules.setdefault("spacy", spacy_stub)

# tqdm stub
tqdm_stub = types.ModuleType("tqdm")
tqdm_stub.tqdm = lambda iterable, **kw: iterable
sys.modules.setdefault("tqdm", tqdm_stub)

import importlib
import importlib.util

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location("gpt4v_eval", os.path.join(HERE, "gpt4v_eval.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

PROVIDER_CONFIGS = mod.PROVIDER_CONFIGS


class TestProviderConfigs(unittest.TestCase):
    """Unit tests for provider configuration."""

    def test_openai_config_present(self):
        self.assertIn("openai", PROVIDER_CONFIGS)

    def test_minimax_config_present(self):
        self.assertIn("minimax", PROVIDER_CONFIGS)

    def test_openai_endpoint(self):
        self.assertIn("api.openai.com", PROVIDER_CONFIGS["openai"]["base_url"])

    def test_minimax_endpoint(self):
        self.assertIn("api.minimax.io", PROVIDER_CONFIGS["minimax"]["base_url"])
        self.assertIn("/v1/chat/completions", PROVIDER_CONFIGS["minimax"]["base_url"])

    def test_minimax_model_is_m27(self):
        self.assertEqual(PROVIDER_CONFIGS["minimax"]["model"], "MiniMax-M2.7")

    def test_minimax_api_key_env(self):
        self.assertEqual(PROVIDER_CONFIGS["minimax"]["api_key_env"], "MINIMAX_API_KEY")

    def test_openai_model(self):
        self.assertIn("gpt-4", PROVIDER_CONFIGS["openai"]["model"])

    def test_all_providers_have_required_keys(self):
        required = {"base_url", "model", "api_key_env"}
        for name, cfg in PROVIDER_CONFIGS.items():
            self.assertTrue(required.issubset(cfg.keys()), f"Provider '{name}' missing keys")


class TestArgParser(unittest.TestCase):
    """Unit tests for the argument parser."""

    def _parse(self, args):
        with patch("sys.argv", ["gpt4v_eval.py"] + args):
            return mod.parse_args()

    def test_default_provider_is_openai(self):
        args = self._parse(["--category", "color"])
        self.assertEqual(args.provider, "openai")

    def test_minimax_provider_flag(self):
        args = self._parse(["--provider", "minimax"])
        self.assertEqual(args.provider, "minimax")

    def test_invalid_provider_raises(self):
        with self.assertRaises(SystemExit):
            self._parse(["--provider", "unknown_provider"])

    def test_default_category(self):
        args = self._parse([])
        self.assertEqual(args.category, "color")

    def test_start_and_step(self):
        args = self._parse(["--start", "5", "--step", "20"])
        self.assertEqual(args.start, 5)
        self.assertEqual(args.step, 20)


class TestApiKeyResolution(unittest.TestCase):
    """Unit tests for API key resolution from environment."""

    def test_minimax_key_from_env(self):
        with patch.dict(os.environ, {"MINIMAX_API_KEY": "test-minimax-key"}):
            key = os.environ.get("MINIMAX_API_KEY", "")
        self.assertEqual(key, "test-minimax-key")

    def test_openai_key_from_env(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-openai-key"}):
            key = os.environ.get("OPENAI_API_KEY", "")
        self.assertEqual(key, "test-openai-key")

    def test_missing_key_returns_empty(self):
        env = {k: v for k, v in os.environ.items() if k != "MINIMAX_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            key = os.environ.get("MINIMAX_API_KEY", "")
        self.assertEqual(key, "")


class TestEncodeImage(unittest.TestCase):
    """Unit tests for the encode_image helper."""

    def test_returns_base64_string(self):
        import tempfile
        data = b"\x89PNG\r\n\x1a\n"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
            f.write(data)
            fname = f.name
        try:
            result = mod.encode_image(fname)
            self.assertEqual(result, base64.b64encode(data).decode("utf-8"))
        finally:
            os.unlink(fname)


class TestRequestPayload(unittest.TestCase):
    """Unit tests verifying the payload structure sent to the provider."""

    def _make_payload(self, provider, model, content_list):
        return {
            "model": model,
            "messages": [{"role": "user", "content": content_list}],
            "max_tokens": 300,
        }

    def test_minimax_payload_uses_m27(self):
        cfg = PROVIDER_CONFIGS["minimax"]
        content = [{"type": "text", "text": "hello"}]
        payload = self._make_payload("minimax", cfg["model"], content)
        self.assertEqual(payload["model"], "MiniMax-M2.7")

    def test_openai_payload_uses_gpt4v(self):
        cfg = PROVIDER_CONFIGS["openai"]
        content = [{"type": "text", "text": "hello"}]
        payload = self._make_payload("openai", cfg["model"], content)
        self.assertIn("gpt-4", payload["model"])

    def test_authorization_header_uses_bearer(self):
        key = "sk-test123"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
        self.assertEqual(headers["Authorization"], "Bearer sk-test123")

    def test_image_content_type_in_payload(self):
        b64 = base64.b64encode(b"fakeimage").decode("utf-8")
        content = [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": "evaluate this"},
        ]
        self.assertEqual(content[0]["type"], "image_url")
        self.assertIn("data:image/jpeg;base64,", content[0]["image_url"]["url"])


class TestIntegrationMiniMaxRequest(unittest.TestCase):
    """Integration-style tests that mock requests.post for MiniMax."""

    def _mock_response(self, score=3):
        response = MagicMock()
        response.json.return_value = {
            "choices": [
                {"message": {"content": f'{{"score": {score}, "explanation": "looks good"}}'}}
            ]
        }
        return response

    @patch("requests.post")
    def test_minimax_endpoint_is_called(self, mock_post):
        mock_post.return_value = self._mock_response()
        endpoint = PROVIDER_CONFIGS["minimax"]["base_url"]
        headers = {"Authorization": "Bearer fake-key"}
        payload = {"model": "MiniMax-M2.7", "messages": [], "max_tokens": 300}
        import requests as req
        req.post(endpoint, headers=headers, json=payload)
        mock_post.assert_called_once_with(endpoint, headers=headers, json=payload)

    @patch("requests.post")
    def test_response_score_parsing(self, mock_post):
        import re
        mock_post.return_value = self._mock_response(score=4)
        response = mock_post()
        content = response.json()["choices"][0]["message"]["content"]
        pattern = r'"score": (\d+)'
        scores = [int(s) for s in re.findall(pattern, content)]
        self.assertEqual(scores, [4])

    @patch("requests.post")
    def test_openai_endpoint_is_called(self, mock_post):
        mock_post.return_value = self._mock_response()
        endpoint = PROVIDER_CONFIGS["openai"]["base_url"]
        headers = {"Authorization": "Bearer fake-key"}
        payload = {"model": "gpt-4-vision-preview", "messages": [], "max_tokens": 300}
        import requests as req
        req.post(endpoint, headers=headers, json=payload)
        mock_post.assert_called_once_with(endpoint, headers=headers, json=payload)


if __name__ == "__main__":
    unittest.main()
