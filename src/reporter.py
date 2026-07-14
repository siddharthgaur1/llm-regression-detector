"""HTML report generator for a single eval run."""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path

from src.comparator import ComparisonResult

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Eval Report — {run_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 2rem; color: #1a1a1a; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 14px; }}
  th {{ background: #f4f4f4; }}
  .scorecard {{ display: flex; gap: 1.5rem; margin: 1rem 0; }}
  .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 1rem 1.5rem; }}
  .ok {{ color: #1a7f37; }}
  .warning {{ color: #9a6700; }}
  .critical {{ color: #cf222e; }}
  .regressed {{ background: #fff1f0; }}
</style>
</head>
<body>
<h1>Eval Report — {run_id}</h1>
<p>Prompt version: <b>{prompt_version}</b> | Model: <b>{model}</b> | Timestamp: {timestamp} | Est. cost: ${cost:.4f}</p>

<div class="scorecard">
  <div class="card"><div>Pass rate</div><h2>{current_pass_rate:.1%}</h2></div>
  <div class="card"><div>Baseline</div><h2>{baseline_pass_rate:.1%}</h2></div>
  <div class="card"><div>Delta</div><h2 class="{severity}">{pass_rate_delta:+.1%}</h2></div>
  <div class="card"><div>Regressions</div><h2>{num_regressions}</h2></div>
  <div class="card"><div>Improvements</div><h2>{num_improvements}</h2></div>
</div>

<h2>Regressed cases</h2>
{regressed_table}

<h2>Trend (last {num_trend} runs)</h2>
<canvas id="trend" height="80"></canvas>
<script>
  new Chart(document.getElementById('trend'), {{
    type: 'line',
    data: {{
      labels: {trend_labels},
      datasets: [{{ label: 'Pass rate', data: {trend_values}, borderColor: '#1a7f37', tension: 0.2 }}]
    }},
    options: {{ scales: {{ y: {{ min: 0, max: 1 }} }} }}
  }});
</script>
</body>
</html>
"""


def _escape(value: object) -> str:
    """HTML-escape a value before interpolating it into the report template.

    Report fields (LLM summaries in particular) are model output, not trusted
    markup, so every value that lands in the regressed-case table goes through
    this to avoid stored XSS if a summary ever contains HTML/script content.
    """
    return html.escape(str(value))


def _regressed_table(regressions: list[str], current_details: dict, baseline_details: dict) -> str:
    """Render the baseline-vs-current output table for regressed test cases."""
    if not regressions:
        return "<p>No regressions.</p>"
    rows = []
    for tc_id in regressions:
        cur = current_details.get(tc_id, {})
        base = baseline_details.get(tc_id, {})
        rows.append(
            f"<tr class='regressed'><td>{_escape(tc_id)}</td>"
            f"<td>{_escape(base.get('category', '—'))} / {_escape(base.get('summary', '—'))}</td>"
            f"<td>{_escape(cur.get('category', '—'))} / {_escape(cur.get('summary', '—'))}</td></tr>"
        )
    return (
        "<table><tr><th>Test case</th><th>Baseline output</th><th>Current output</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def generate_report(
    run_dir: Path,
    run_id: str,
    prompt_version: str,
    model: str,
    timestamp: str,
    cost: float,
    comparison: ComparisonResult,
    current_details: dict,
    baseline_details: dict,
    trend: list[dict],
) -> Path:
    """Render and write an HTML report for one eval run.

    Args:
        run_dir: Directory to write report.html into (created if missing).
        run_id: The current run's ID, shown in the report header.
        prompt_version: Prompt version used for this run.
        model: Classifier model name.
        timestamp: ISO-ish timestamp string for the run.
        cost: Estimated USD cost of the run.
        comparison: Output of comparator.compare_runs for this run.
        current_details: test_case_id -> {category, summary, ...} for the current run.
        baseline_details: Same shape, for the baseline run (may be empty).
        trend: List of {"run_id": str, "pass_rate": float} in chronological order.

    Returns:
        Path to the written report.html.
    """
    html_doc = REPORT_TEMPLATE.format(
        run_id=_escape(run_id),
        prompt_version=_escape(prompt_version),
        model=_escape(model),
        timestamp=_escape(timestamp),
        cost=cost,
        current_pass_rate=comparison.current_pass_rate,
        baseline_pass_rate=comparison.baseline_pass_rate,
        pass_rate_delta=comparison.pass_rate_delta,
        severity=comparison.severity.value,
        num_regressions=len(comparison.regressions),
        num_improvements=len(comparison.improvements),
        regressed_table=_regressed_table(comparison.regressions, current_details, baseline_details),
        num_trend=len(trend),
        trend_labels=json.dumps([t["run_id"] for t in trend]),
        trend_values=json.dumps([t["pass_rate"] for t in trend]),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.html"
    report_path.write_text(html_doc, encoding="utf-8")
    logger.info("Report written to %s", report_path)
    return report_path
