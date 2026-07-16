# llm-regression-detector

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Prompts drift. A tweak that fixes one failure mode quietly breaks three
others, and unless something is watching, the first signal is a support
ticket. This is a CI harness for an LLM-powered support-email classifier: it
runs a golden dataset through whatever prompt just changed, scores category
accuracy and summary quality, diffs the result against the last known-good
run, and fails the build (with a Slack ping) when the drop crosses a
threshold. It exists to make "did this prompt change get worse" a testable
question instead of a vibe.

## Architecture

```
                         ┌──────────────┐
 prompts/v1.yaml ───────▶│              │
 prompts/v2.yaml ───────▶│  feature.py  │  classify_email()
                         │ (classifier) │  gpt-4o-mini, structured output
                         └──────┬───────┘
                                │ ClassificationResult
                                ▼
 data/golden_dataset.json ─▶┌──────────────┐
        (60 cases)          │ evaluator.py │  category_match (exact)
                             │              │  summary_relevance (LLM judge, gpt-4o)
                             └──────┬───────┘
                                    │ EvalScore per case
                                    ▼
 data/runs/{prev_run}/  ───▶┌──────────────┐
   results.json             │comparator.py │  pass-rate delta, regressions,
                             │              │  category deltas, 7-run drift
                             └──────┬───────┘
                                    │ ComparisonResult
                       ┌────────────┼────────────┐
                       ▼                          ▼
               ┌──────────────┐          ┌──────────────┐
               │ reporter.py  │          │ alerting.py  │
               │ report.html  │          │ Slack webhook│
               └──────────────┘          └──────────────┘
                       ▲
                       │ reads all runs
               ┌──────────────┐
               │ dashboard/   │  Streamlit: trends, run diff, case explorer
               │   app.py     │
               └──────────────┘

runner.py orchestrates all of the above; .github/workflows/eval.yml
triggers it on every push that touches prompts/.
```

## Setup

```bash
git clone <repo-url> && cd llm-regression-detector
cp .env.example .env        # OPENAI_API_KEY required, SLACK_WEBHOOK_URL optional
docker compose up dashboard # http://localhost:8501, reads data/runs/
```

For running evals locally without Docker: `pip install -r requirements.txt`.

## Running an eval

```bash
python -m src.runner --prompt-version v1
```

This loads `prompts/v1.yaml`, runs all 60 golden dataset cases through the
classifier (max 10 concurrent requests), scores each one, diffs against the
most recent prior run (or pass `--baseline-run <run_id>` to pin a specific
one), writes `data/runs/{run_id}/{results.json,report.html}`, and exits 1 if
the drop is CRITICAL. `--slack` sends the alert; omit it for a silent local
run.

## Adding golden dataset cases

Append an object to `data/golden_dataset.json`:

```json
{
  "id": "bill-016",
  "input_email": "...",
  "expected_category": "billing",
  "ideal_summary_keywords": ["..."],
  "difficulty": "medium",
  "notes": "why this case exists"
}
```

`ideal_summary_keywords` isn't scored automatically today (the judge grades
holistically) — it's there for a human skimming regressions to sanity-check
what the summary *should* mention. Bias new cases toward the hard end:
sarcasm, typos, one-word emails, and category-ambiguous requests are where
prompt changes actually diverge. A case that every prompt version passes
trivially isn't buying you regression coverage.

## Adjusting regression thresholds

`src/comparator.py`:

```python
WARNING_THRESHOLD = 0.03   # pass-rate drop that triggers a warning
CRITICAL_THRESHOLD = 0.08  # pass-rate drop that fails the build
```

Same thresholds are reused by `moving_average_drift()` for the 7-run
trailing comparison — there's no separate drift config, so tightening one
tightens both. If per-category thresholds ever become necessary (e.g.
billing regressions matter more than general), that's a `category_deltas`
lookup away in `compare_runs()`, but there was no case for it yet.

## How the GitHub Action blocks bad merges

`.github/workflows/eval.yml` runs on any push to `main` touching
`prompts/**`, executes `python -m src.runner`, and relies on the runner's
own exit code — 1 on CRITICAL, 0 otherwise. A failing exit code fails the CI
job, and a failing required check blocks merge (once branch protection is
turned on for this repo; the workflow doesn't configure that itself, GitHub
repo settings do). On PRs it also drops a pass/fail summary comment so the
reviewer doesn't have to open Actions to see the number.

## Design decisions

**Flat JSON files under `data/runs/`, not SQLite.** Each run is small (60
cases), runs are append-only, and the dashboard's access pattern is "load
the last ~10 runs and diff two of them" — a directory of JSON files handles
that with zero schema migrations and lets you `git diff` a regression by
eye. SQLite would earn its keep once queries need to span thousands of runs
or filter across runs by arbitrary fields; that's not this dataset's shape
yet.

**LLM-as-judge for summary quality, not string similarity.** Category match
is exact-match because there's a fixed label set — no judgment call needed.
Summary quality has no fixed target string; "the customer's card was
double-billed" and "duplicate subscription charge, needs refund" are both
correct and share almost no tokens. A judge model reading for whether the
summary names the *right issue* catches that; ROUGE/BLEU-style overlap
scoring would just penalize valid paraphrases.

**Slow drift is tracked separately from run-vs-baseline diffing.**
`compare_runs()` only ever looks at the immediately preceding run. That
misses the failure mode where each prompt edit degrades accuracy by 1-2%
against its immediate predecessor — under the 3% warning threshold every
single time — but the tenth edit has quietly walked the pass rate down 15%
from where it started. `moving_average_drift()` catches that by comparing
the latest run to a 7-run trailing average instead of just run N-1.

## Screenshots

The dashboard isn't screenshotted here — it needs live run data and a
browser to render. To capture it yourself:

```bash
python -m src.runner --prompt-version v1
python -m src.runner --prompt-version v2
streamlit run dashboard/app.py
```

Then, with the app open at `http://localhost:8501`:

1. **Overview** — screenshot the "Recent runs" table showing both `v1` and
   `v2` runs with their pass/warn/fail status.
2. **Trends** — screenshot the three trend charts (accuracy, summary
   quality, latency) once at least 2-3 runs exist.
3. **Run comparison** — select the `v1` run as baseline and `v2` as current,
   screenshot the regressions/improvements output.
4. **Test case explorer** — filter to `difficulty: hard`, screenshot the
   table to show where the two prompt versions actually diverge.

Drop the images in a `docs/screenshots/` folder and reference them here as
`![Overview](docs/screenshots/overview.png)` once captured.

## Related projects

- [rag-hybrid-search](https://github.com/siddharthgaur1/rag-hybrid-search) — hybrid dense+BM25 RAG pipeline; its eval output is shaped to plug into this project's `comparator.py`/`alerting.py` directly (see that repo's "Connection to Project 1").
- [ipo-gmp](https://github.com/siddharthgaur1/ipo-gmp) — XGBoost IPO listing-return predictor.
