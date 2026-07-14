"""Generate two demo eval runs (v1, v2) using deterministic mock scoring — no OpenAI calls.

For portfolio/demo purposes: populates data/runs/ with a realistic v1-vs-v2
comparison so the dashboard and HTML report have something to show without
requiring an API key. v1 (zero-shot) is modeled as weaker on hard cases;
v2 (few-shot + CoT) is modeled as stronger there but marginally slower.

Usage: python scripts/generate_demo_data.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.comparator import compare_runs  # noqa: E402
from src.reporter import generate_report  # noqa: E402

DIFFICULTY_PASS_RATE = {
    "v1": {"easy": 0.97, "medium": 0.80, "hard": 0.45},
    "v2": {"easy": 0.98, "medium": 0.90, "hard": 0.75},
}
DIFFICULTY_LATENCY_MS = {"v1": 420, "v2": 560}


def mock_score(test_case: dict, prompt_version: str, rng: random.Random) -> dict:
    difficulty = test_case["difficulty"]
    pass_probability = DIFFICULTY_PASS_RATE[prompt_version][difficulty]
    passed = rng.random() < pass_probability
    category = test_case["expected_category"] if passed else rng.choice(
        [c for c in ("billing", "technical", "account", "general") if c != test_case["expected_category"]]
    )
    summary_relevance = rng.uniform(0.75, 1.0) if passed else rng.uniform(0.1, 0.45)
    latency_ms = DIFFICULTY_LATENCY_MS[prompt_version] + rng.uniform(-40, 40)
    return {
        "test_case_id": test_case["id"],
        "category": category,
        "summary": f"[demo] mock summary for {test_case['id']}",
        "confidence": rng.uniform(0.6, 0.98),
        "category_match": 1.0 if category == test_case["expected_category"] else 0.0,
        "summary_relevance": summary_relevance,
        "latency_ms": latency_ms,
        "input_tokens": rng.randint(80, 160),
        "output_tokens": rng.randint(15, 35),
        "passed": (category == test_case["expected_category"]) and summary_relevance >= 0.5,
    }


def build_run(prompt_version: str, test_cases: list[dict], seed: int) -> dict:
    rng = random.Random(seed)
    results = [mock_score(tc, prompt_version, rng) for tc in test_cases]
    run_id = f"{prompt_version}_demo"
    return {
        "run_id": run_id,
        "prompt_version": prompt_version,
        "classifier_model": "gpt-4o-mini",
        "judge_model": "gpt-4o",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cost": round(rng.uniform(0.01, 0.03), 4),
        "results": results,
    }


def main() -> None:
    test_cases = json.loads((ROOT / "data" / "golden_dataset.json").read_text(encoding="utf-8"))
    test_cases_by_id = {tc["id"]: tc for tc in test_cases}

    v1_run = build_run("v1", test_cases, seed=1)
    v2_run = build_run("v2", test_cases, seed=2)

    for run in (v1_run, v2_run):
        run_dir = ROOT / "data" / "runs" / run["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "results.json").write_text(json.dumps(run, indent=2), encoding="utf-8")

    comparison = compare_runs(
        v2_run["run_id"], v2_run["results"], v1_run["run_id"], v1_run["results"], test_cases_by_id
    )
    current_details = {r["test_case_id"]: r for r in v2_run["results"]}
    baseline_details = {r["test_case_id"]: r for r in v1_run["results"]}
    trend = [
        {"run_id": v1_run["run_id"], "pass_rate": sum(1 for r in v1_run["results"] if r["passed"]) / len(v1_run["results"])},
        {"run_id": v2_run["run_id"], "pass_rate": sum(1 for r in v2_run["results"] if r["passed"]) / len(v2_run["results"])},
    ]
    report_path = generate_report(
        run_dir=ROOT / "data" / "runs" / v2_run["run_id"],
        run_id=v2_run["run_id"],
        prompt_version="v2",
        model=v2_run["classifier_model"],
        timestamp=v2_run["timestamp"],
        cost=v2_run["cost"],
        comparison=comparison,
        current_details=current_details,
        baseline_details=baseline_details,
        trend=trend,
    )

    print(f"v1 pass rate: {trend[0]['pass_rate']:.1%}")
    print(f"v2 pass rate: {trend[1]['pass_rate']:.1%}")
    print(f"delta: {comparison.pass_rate_delta:+.1%}, severity: {comparison.severity.value}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
