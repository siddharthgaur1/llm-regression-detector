"""Tests for src/feature.py using a mocked AsyncOpenAI client — no real API calls."""

from __future__ import annotations

import pytest

from src.feature import PromptConfig, classify_email
from tests.conftest import make_parse_response


class TestPromptConfigBuildMessages:
    def test_no_few_shot_examples(self, prompt_config):
        messages = prompt_config.build_messages("Hello, my card was charged twice.")
        assert messages == [
            {"role": "system", "content": "Classify the email."},
            {"role": "user", "content": "Hello, my card was charged twice."},
        ]

    def test_few_shot_examples_expand_to_user_assistant_pairs(self):
        config = PromptConfig(
            version_id="v2",
            timestamp="2026-01-01T00:00:00Z",
            system_prompt="Classify.",
            few_shot_examples=[
                {"input_email": "example email", "category": "billing", "summary": "s", "confidence": 0.9}
            ],
        )
        messages = config.build_messages("real email")
        assert len(messages) == 4  # system, few-shot user, few-shot assistant, real user
        assert messages[1] == {"role": "user", "content": "example email"}
        assert messages[2]["role"] == "assistant"
        assert '"category": "billing"' in messages[2]["content"]
        assert messages[3] == {"role": "user", "content": "real email"}

    def test_empty_email_raises(self, prompt_config):
        with pytest.raises(ValueError):
            prompt_config.build_messages("")

    def test_whitespace_only_email_raises(self, prompt_config):
        with pytest.raises(ValueError):
            prompt_config.build_messages("   ")


class TestPromptConfigLoad:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            PromptConfig.load(tmp_path / "nonexistent.yaml")

    def test_missing_required_key_raises(self, tmp_path):
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("version_id: v1\n", encoding="utf-8")
        with pytest.raises(KeyError):
            PromptConfig.load(bad_yaml)

    def test_loads_valid_file(self, tmp_path):
        path = tmp_path / "v1.yaml"
        path.write_text("version_id: v1\nsystem_prompt: 'do the thing'\n", encoding="utf-8")
        config = PromptConfig.load(path)
        assert config.version_id == "v1"
        assert config.system_prompt == "do the thing"
        assert config.few_shot_examples == []


@pytest.mark.asyncio
class TestClassifyEmail:
    async def test_returns_parsed_result_with_telemetry(
        self, mock_openai_client, prompt_config, classification_result_factory
    ):
        parsed = classification_result_factory(category="technical", confidence=0.8)
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(
            parsed, input_tokens=150, output_tokens=30
        )

        result = await classify_email(mock_openai_client, prompt_config, "the app crashed", "gpt-4o-mini")

        assert result.result.category == "technical"
        assert result.result.confidence == 0.8
        assert result.input_tokens == 150
        assert result.output_tokens == 30
        assert result.latency_ms >= 0
        mock_openai_client.chat.completions.parse.assert_awaited_once()

    async def test_none_parsed_raises(self, mock_openai_client, prompt_config):
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(None)
        with pytest.raises(ValueError):
            await classify_email(mock_openai_client, prompt_config, "some email", "gpt-4o-mini")

    async def test_missing_usage_defaults_to_zero_tokens(
        self, mock_openai_client, prompt_config, classification_result_factory
    ):
        response = make_parse_response(classification_result_factory())
        response.usage = None
        mock_openai_client.chat.completions.parse.return_value = response

        result = await classify_email(mock_openai_client, prompt_config, "some email", "gpt-4o-mini")

        assert result.input_tokens == 0
        assert result.output_tokens == 0
