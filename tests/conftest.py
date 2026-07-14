"""Shared pytest fixtures: golden dataset samples, mock OpenAI responses, prompt configs."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.evaluator import JudgeScore
from src.feature import ClassificationResult, PromptConfig


@pytest.fixture
def sample_test_case() -> dict:
    return {
        "id": "bill-001",
        "input_email": "I was charged twice for my subscription, please refund the duplicate.",
        "expected_category": "billing",
        "ideal_summary_keywords": ["double charge", "refund"],
        "difficulty": "easy",
        "notes": "Clear duplicate charge.",
    }


@pytest.fixture
def prompt_config() -> PromptConfig:
    return PromptConfig(
        version_id="v1",
        timestamp="2026-01-01T00:00:00Z",
        system_prompt="Classify the email.",
        few_shot_examples=[],
    )


def make_parse_response(parsed, input_tokens: int = 100, output_tokens: int = 20):
    """Build a fake object shaped like an OpenAI ``.chat.completions.parse()`` response."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed))],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
    )


@pytest.fixture
def mock_openai_client():
    """An AsyncOpenAI-shaped mock whose .chat.completions.parse() can be configured per-test."""
    client = AsyncMock()
    client.chat.completions.parse = AsyncMock()
    return client


@pytest.fixture
def classification_result_factory():
    def _make(category="billing", summary="Customer was double-charged.", confidence=0.95):
        return ClassificationResult(category=category, summary=summary, confidence=confidence)

    return _make


@pytest.fixture
def judge_score_factory():
    def _make(score=4):
        return JudgeScore(score=score)

    return _make
