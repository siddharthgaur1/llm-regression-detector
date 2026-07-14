"""Multi-dimensional scoring engine: category match + LLM-as-judge summary quality."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.feature import FeatureResult

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are grading a customer-support email summary. Given the
original email and a one-sentence summary, score how well the summary captures the
customer's actual issue on a 1-5 scale:
1 = misses the issue entirely, 3 = captures the topic but is vague or generic,
5 = precisely captures the concrete issue. Return only the integer score."""

PASS_SUMMARY_THRESHOLD = 0.5


class JudgeScore(BaseModel):
    """Raw 1-5 score returned by the LLM judge for a single summary."""

    score: int = Field(..., ge=1, le=5, description="Judge's rating of summary quality, 1 (worst) to 5 (best).")


class EvalScore(BaseModel):
    """Per-test-case scoring result across all evaluation dimensions."""

    test_case_id: str = Field(..., description="ID of the golden dataset test case this score belongs to.")
    category_match: float = Field(..., description="1.0 if predicted category matches expected, else 0.0.")
    summary_relevance: float = Field(..., ge=0.0, le=1.0, description="LLM-judge summary score, normalized 1-5 -> 0-1.")
    latency_ms: float = Field(..., ge=0.0, description="Classifier request latency in milliseconds.")
    input_tokens: int = Field(..., ge=0, description="Prompt tokens consumed by the classifier call.")
    output_tokens: int = Field(..., ge=0, description="Completion tokens consumed by the classifier call.")
    passed: bool = Field(..., description="True if category matched and summary_relevance met threshold.")


async def judge_summary_relevance(
    client: AsyncOpenAI, judge_model: str, input_email: str, summary: str
) -> float:
    """Score a summary's relevance to its source email using an LLM judge.

    Returns:
        A float in [0, 1], the judge's 1-5 score normalized.

    Raises:
        ValueError: if the judge model returns no parsable score.
    """
    response = await client.chat.completions.parse(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Email:\n{input_email}\n\nSummary:\n{summary}",
            },
        ],
        response_format=JudgeScore,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"Judge model {judge_model} returned no parsable score")
    logger.debug("Judge scored summary relevance: %d/5", parsed.score)
    return (parsed.score - 1) / 4  # normalize 1-5 to 0-1


async def evaluate_case(
    client: AsyncOpenAI,
    judge_model: str,
    test_case: dict,
    feature_result: FeatureResult,
) -> EvalScore:
    """Score a single classifier result against its golden test case.

    Args:
        client: An initialized AsyncOpenAI client, used for the judge call.
        judge_model: Model name for the LLM-as-judge (e.g. "gpt-4o").
        test_case: A golden dataset entry with at least "id", "input_email",
            and "expected_category".
        feature_result: The classifier's output for this test case.

    Returns:
        An EvalScore combining category accuracy and judged summary relevance.

    Raises:
        KeyError: if test_case is missing required fields.
    """
    for required_key in ("id", "input_email", "expected_category"):
        if required_key not in test_case:
            raise KeyError(f"Test case is missing required key '{required_key}': {test_case}")

    category_match = 1.0 if feature_result.result.category == test_case["expected_category"] else 0.0
    summary_relevance = await judge_summary_relevance(
        client, judge_model, test_case["input_email"], feature_result.result.summary
    )
    passed = category_match == 1.0 and summary_relevance >= PASS_SUMMARY_THRESHOLD
    return EvalScore(
        test_case_id=test_case["id"],
        category_match=category_match,
        summary_relevance=summary_relevance,
        latency_ms=feature_result.latency_ms,
        input_tokens=feature_result.input_tokens,
        output_tokens=feature_result.output_tokens,
        passed=passed,
    )
