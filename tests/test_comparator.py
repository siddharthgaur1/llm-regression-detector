"""Tests for src/comparator.py — regression/improvement detection against mock run data."""

from __future__ import annotations

import pytest

from src.comparator import Severity, compare_runs, moving_average_drift

TEST_CASES_BY_ID = {
    "a": {"expected_category": "billing"},
    "b": {"expected_category": "technical"},
    "c": {"expected_category": "billing"},
}


def score(test_case_id: str, passed: bool) -> dict:
    return {"test_case_id": test_case_id, "passed": passed}


class TestCompareRuns:
    def test_detects_regression_and_improvement(self):
        baseline = [score("a", True), score("b", False), score("c", True)]
        current = [score("a", False), score("b", True), score("c", True)]

        result = compare_runs("run2", current, "run1", baseline, TEST_CASES_BY_ID)

        assert result.regressions == ["a"]
        assert result.improvements == ["b"]
        assert result.pass_rate_delta == pytest.approx(0.0)

    def test_no_baseline_treated_as_ok_with_no_diffs(self):
        current = [score("a", True)]
        result = compare_runs("run1", current, None, None, {"a": {"expected_category": "general"}})

        assert result.severity == Severity.OK
        assert result.baseline_pass_rate == result.current_pass_rate
        assert result.regressions == []
        assert result.improvements == []

    def test_empty_current_scores_raises(self):
        with pytest.raises(ValueError):
            compare_runs("run1", [], None, None, {})

    def test_unknown_test_case_id_raises(self):
        current = [score("unknown-id", True)]
        with pytest.raises(KeyError):
            compare_runs("run1", current, None, None, TEST_CASES_BY_ID)

    def test_category_deltas_computed_per_category(self):
        baseline = [score("a", True), score("b", True), score("c", True)]
        current = [score("a", False), score("b", True), score("c", True)]

        result = compare_runs("run2", current, "run1", baseline, TEST_CASES_BY_ID)

        # billing has 2 cases (a, c): baseline 100% -> current 50%, delta -50%
        assert result.category_deltas["billing"] == pytest.approx(-0.5)
        # technical has 1 case (b): unchanged
        assert result.category_deltas["technical"] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "num_failing,expected_severity",
        [
            (0, Severity.OK),
            (2, Severity.OK),  # 2% drop, under 3% warning threshold
            (5, Severity.WARNING),  # 5% drop
            (10, Severity.CRITICAL),  # 10% drop
        ],
    )
    def test_severity_thresholds(self, num_failing, expected_severity):
        baseline = [score(str(i), True) for i in range(100)]
        current = [score(str(i), i >= num_failing) for i in range(100)]
        cases = {str(i): {"expected_category": "general"} for i in range(100)}

        result = compare_runs("run2", current, "run1", baseline, cases)

        assert result.severity == expected_severity


class TestMovingAverageDrift:
    def test_returns_none_with_insufficient_history(self):
        assert moving_average_drift([]) is None
        assert moving_average_drift([0.9]) is None

    def test_stable_history_is_ok(self):
        drift = moving_average_drift([0.95] * 7 + [0.95])
        assert drift["severity"] == Severity.OK

    def test_slow_drop_below_any_single_threshold_flags_critical_over_window(self):
        drift = moving_average_drift([0.95] * 7 + [0.85])
        assert drift["severity"] == Severity.CRITICAL
        assert drift["delta"] == pytest.approx(-0.10)

    def test_window_caps_trailing_history(self):
        # 20 runs at 0.95, then one at 0.90 — only the last `window` runs before
        # the latest should be averaged, not the full history.
        drift = moving_average_drift([0.95] * 20 + [0.90], window=7)
        assert drift["moving_average"] == pytest.approx(0.95)
        assert drift["severity"] == Severity.WARNING
