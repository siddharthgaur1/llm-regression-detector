"""Tests for src/evaluator.py — each scoring dimension tested independently, mocked OpenAI."""

from __future__ import annotations

import pytest

from src.evaluator import evaluate_case, judge_summary_relevance
from src.feature import FeatureResult
from tests.conftest import make_parse_response


@pytest.mark.asyncio
class TestJudgeSummaryRelevance:
    @pytest.mark.parametrize(
        "raw_score,expected_normalized",
        [(1, 0.0), (2, 0.25), (3, 0.5), (4, 0.75), (5, 1.0)],
    )
    async def test_normalizes_1_to_5_scale_into_0_to_1(
        self, mock_openai_client, judge_score_factory, raw_score, expected_normalized
    ):
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(
            judge_score_factory(score=raw_score)
        )
        result = await judge_summary_relevance(mock_openai_client, "gpt-4o", "email text", "summary text")
        assert result == pytest.approx(expected_normalized)

    async def test_none_parsed_raises(self, mock_openai_client):
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(None)
        with pytest.raises(ValueError):
            await judge_summary_relevance(mock_openai_client, "gpt-4o", "email", "summary")


@pytest.mark.asyncio
class TestEvaluateCase:
    async def _feature_result(self, classification_result_factory, **overrides):
        parsed = classification_result_factory(**overrides)
        return FeatureResult(result=parsed, latency_ms=120.0, input_tokens=100, output_tokens=20)

    async def test_category_match_scores_one_when_correct(
        self, mock_openai_client, sample_test_case, classification_result_factory, judge_score_factory
    ):
        feature_result = await self._feature_result(classification_result_factory, category="billing")
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(judge_score_factory(score=5))

        score = await evaluate_case(mock_openai_client, "gpt-4o", sample_test_case, feature_result)

        assert score.category_match == 1.0
        assert score.passed is True

    async def test_category_match_scores_zero_when_wrong(
        self, mock_openai_client, sample_test_case, classification_result_factory, judge_score_factory
    ):
        feature_result = await self._feature_result(classification_result_factory, category="technical")
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(judge_score_factory(score=5))

        score = await evaluate_case(mock_openai_client, "gpt-4o", sample_test_case, feature_result)

        assert score.category_match == 0.0
        assert score.passed is False  # wrong category always fails regardless of summary quality

    async def test_low_summary_relevance_fails_even_with_correct_category(
        self, mock_openai_client, sample_test_case, classification_result_factory, judge_score_factory
    ):
        feature_result = await self._feature_result(classification_result_factory, category="billing")
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(judge_score_factory(score=2))

        score = await evaluate_case(mock_openai_client, "gpt-4o", sample_test_case, feature_result)

        assert score.category_match == 1.0
        assert score.summary_relevance == pytest.approx(0.25)
        assert score.passed is False

    async def test_telemetry_passthrough(
        self, mock_openai_client, sample_test_case, classification_result_factory, judge_score_factory
    ):
        feature_result = await self._feature_result(classification_result_factory, category="billing")
        mock_openai_client.chat.completions.parse.return_value = make_parse_response(judge_score_factory(score=5))

        score = await evaluate_case(mock_openai_client, "gpt-4o", sample_test_case, feature_result)

        assert score.latency_ms == 120.0
        assert score.input_tokens == 100
        assert score.output_tokens == 20
        assert score.test_case_id == "bill-001"

    async def test_missing_required_key_raises(self, mock_openai_client, classification_result_factory):
        feature_result = await self._feature_result(classification_result_factory, category="billing")
        incomplete_case = {"id": "x"}  # missing input_email, expected_category
        with pytest.raises(KeyError):
            await evaluate_case(mock_openai_client, "gpt-4o", incomplete_case, feature_result)
