"""Main orchestrator: runs the golden dataset through the classifier, scores it,
compares against a baseline, writes a report, and alerts Slack on regressions.

Usage:
    python -m src.runner --prompt-version v1 [--baseline-run <run_id>]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from src.comparator import Severity, compare_runs, moving_average_drift
from src.evaluator import evaluate_case
from src.feature import PromptConfig, classify_email
from src.reporter import generate_report
from src.alerting import send_slack_alert, send_drift_alert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
MAX_CONCURRENCY = 10

# gpt-4o-mini / gpt-4o pricing, USD per 1K tokens (input, output)
PRICING = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.0025, 0.01),
}


def load_golden_dataset() -> list[dict]:
    """Load the golden dataset test cases from data/golden_dataset.json.

    Raises:
        FileNotFoundError: if the golden dataset file is missing.
    """
    dataset_path = DATA_DIR / "golden_dataset.json"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")
    return json.loads(dataset_path.read_text(encoding="utf-8"))


def list_run_ids() -> list[str]:
    """List saved run IDs under data/runs/, oldest first."""
    if not RUNS_DIR.exists():
        return []
    return sorted(
        (p.name for p in RUNS_DIR.iterdir() if p.is_dir() and (p / "results.json").exists()),
        key=lambda name: (RUNS_DIR / name / "results.json").stat().st_mtime,
    )


def load_run(run_id: str) -> dict:
    """Load a saved run's results.json by run ID.

    Raises:
        FileNotFoundError: if no run with this ID has been saved.
    """
    results_path = RUNS_DIR / run_id / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No run found with ID '{run_id}' (expected {results_path})")
    return json.loads(results_path.read_text(encoding="utf-8"))


async def run_one_case(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    prompt_config: PromptConfig,
    classifier_model: str,
    judge_model: str,
    test_case: dict,
) -> dict:
    """Classify and score a single golden dataset case, bounded by semaphore concurrency."""
    async with semaphore:
        feature_result = await classify_email(client, prompt_config, test_case["input_email"], classifier_model)
        score = await evaluate_case(client, judge_model, test_case, feature_result)
        return {
            "test_case_id": test_case["id"],
            "category": feature_result.result.category,
            "summary": feature_result.result.summary,
            "confidence": feature_result.result.confidence,
            **score.model_dump(exclude={"test_case_id"}),
        }


async def run_eval(
    prompt_version: str,
    classifier_model: str,
    judge_model: str,
) -> tuple[str, list[dict]]:
    """Run the full golden dataset through the classifier + evaluator for one prompt version.

    Args:
        prompt_version: Filename stem under prompts/ (e.g. "v1" for prompts/v1.yaml).
        classifier_model: Model name for the feature under test.
        judge_model: Model name for the LLM-as-judge summary scorer.

    Returns:
        (run_id, results) where results is one score dict per golden dataset case.
    """
    prompt_path = ROOT / "prompts" / f"{prompt_version}.yaml"
    prompt_config = PromptConfig.load(prompt_path)
    test_cases = load_golden_dataset()

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = [
        run_one_case(client, semaphore, prompt_config, classifier_model, judge_model, tc)
        for tc in test_cases
    ]
    results = await asyncio.gather(*tasks)

    run_id = f"{prompt_version}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    return run_id, results


def estimate_cost(results: list[dict], classifier_model: str, judge_model: str) -> float:
    """Estimate USD cost for a run from classifier token usage plus a flat per-case judge estimate.

    Judge call tokens aren't tracked per-case (judge input is short and roughly
    constant), so this approximates them at ~200 input / ~5 output tokens per
    case rather than reading exact usage off each judge response. Good enough
    for a relative cost signal, not exact billing reconciliation.
    """
    in_price, out_price = PRICING.get(classifier_model, (0.0, 0.0))
    total = 0.0
    for r in results:
        total += r["input_tokens"] / 1000 * in_price + r["output_tokens"] / 1000 * out_price
    # judge calls aren't token-tracked per case here; approximate at a flat small cost
    judge_in, judge_out = PRICING.get(judge_model, (0.0, 0.0))
    total += len(results) * (200 / 1000 * judge_in + 5 / 1000 * judge_out)
    return total


async def main_async(args: argparse.Namespace) -> int:
    """Run an eval, save results, generate a report, optionally alert Slack.

    Returns:
        Process exit code: 1 if the run is a CRITICAL regression, else 0.
    """
    classifier_model = args.classifier_model
    judge_model = args.judge_model

    run_id, results = await run_eval(args.prompt_version, classifier_model, judge_model)
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cost = estimate_cost(results, classifier_model, judge_model)
    run_payload = {
        "run_id": run_id,
        "prompt_version": args.prompt_version,
        "classifier_model": classifier_model,
        "judge_model": judge_model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cost": cost,
        "results": results,
    }
    (run_dir / "results.json").write_text(json.dumps(run_payload, indent=2), encoding="utf-8")
    logger.info("Saved run %s (%d cases, est. cost $%.4f)", run_id, len(results), cost)

    prior_run_ids = [r for r in list_run_ids() if r != run_id]
    baseline_run_id = args.baseline_run or (prior_run_ids[-1] if prior_run_ids else None)
    baseline_results = load_run(baseline_run_id)["results"] if baseline_run_id else None

    test_cases_by_id = {tc["id"]: tc for tc in load_golden_dataset()}
    comparison = compare_runs(run_id, results, baseline_run_id, baseline_results, test_cases_by_id)

    current_details = {r["test_case_id"]: r for r in results}
    baseline_details = {r["test_case_id"]: r for r in (baseline_results or [])}

    trend_run_ids = (prior_run_ids + [run_id])[-10:]
    trend = [
        {
            "run_id": rid,
            "pass_rate": sum(1 for r in load_run(rid)["results"] if r["passed"]) / len(load_run(rid)["results"]),
        }
        for rid in trend_run_ids
    ]

    report_path = generate_report(
        run_dir=run_dir,
        run_id=run_id,
        prompt_version=args.prompt_version,
        model=classifier_model,
        timestamp=run_payload["timestamp"],
        cost=cost,
        comparison=comparison,
        current_details=current_details,
        baseline_details=baseline_details,
        trend=trend,
    )

    if args.slack:
        await send_slack_alert(comparison, report_url=str(report_path))
        drift = moving_average_drift([t["pass_rate"] for t in trend])
        if drift and drift["severity"] != Severity.OK:
            await send_drift_alert(drift, report_url=str(report_path))

    logger.info(
        "Pass rate %.1f%% (baseline %.1f%%, delta %+.1f%%), %d regressions, %d improvements, severity=%s",
        comparison.current_pass_rate * 100,
        comparison.baseline_pass_rate * 100,
        comparison.pass_rate_delta * 100,
        len(comparison.regressions),
        len(comparison.improvements),
        comparison.severity.value,
    )

    return 1 if comparison.severity == Severity.CRITICAL else 0


def main() -> None:
    """CLI entrypoint: parse args, load .env, run the eval, exit with the eval's status code."""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run the golden-dataset eval for a prompt version.")
    parser.add_argument("--prompt-version", required=True, help="e.g. v1 or v2")
    parser.add_argument("--baseline-run", default=None, help="Run ID to compare against (default: most recent prior run)")
    parser.add_argument("--classifier-model", default="gpt-4o-mini")
    parser.add_argument("--judge-model", default="gpt-4o")
    parser.add_argument("--slack", action="store_true", help="Send Slack alert after the run")
    args = parser.parse_args()

    exit_code = asyncio.run(main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
