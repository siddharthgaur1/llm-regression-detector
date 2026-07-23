# Security

## Threat model

A CI-oriented tool that runs an LLM classifier against a golden dataset, scores it
with an LLM-as-judge, and flags regressions between prompt versions. It processes
its own committed golden dataset, not arbitrary user input, and posts alerts to an
operator-configured Slack webhook. The dashboard is read-only over committed run
artifacts.

## What is mitigated

| Risk | Status | Where |
|---|---|---|
| Secrets in git history | **Clean** — `gitleaks`: 0 findings; `.env` gitignored and never tracked |
| Dependency CVEs | **Clean** — `pip-audit`: no known vulnerabilities; versions pinned |
| Container running as root | **Fixed** — image now runs as uid 10001 `evaluator` (was root) | `Dockerfile` |
| Alerting failing the run | **Mitigated** — a missing `SLACK_WEBHOOK_URL` makes `--slack` no-op with a warning rather than crash | `src/alerting.py` |
| Code execution / injection | **Not present** — no `eval`/`exec`/`subprocess`/`pickle` in the app path |
| Dashboard needing a key | **N/A** — the dashboard reads committed run JSON only; it makes no LLM calls, so the demo runs with no key |

## What is NOT mitigated / notes

- **No authentication** on the dashboard. It only exposes committed evaluation
  results (scores, pass rates), not secrets.
- **The eval run sends the golden dataset to the model provider** (OpenAI, or any
  endpoint set via `OPENAI_BASE_URL`). The golden dataset is synthetic and
  committed; if you replace it with sensitive data, that data goes to your chosen
  provider — a data-handling decision you make by choosing the provider.
- **LLM-as-judge is itself an LLM** and inherits LLM failure modes (miscalibration,
  susceptibility to adversarial inputs in the data it scores). Regression *signals*
  from it are directional; this is the point of the tool, which is to surface
  drift for a human to inspect, not to make a release decision autonomously.

## Reporting

Open an issue. Portfolio/demo project, no production deployment, no security SLA.
