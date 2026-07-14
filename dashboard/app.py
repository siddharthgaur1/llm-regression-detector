"""Streamlit dashboard for browsing eval runs, trends, and case-level diffs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.comparator import compare_runs  # noqa: E402
from src.runner import list_run_ids, load_run, load_golden_dataset  # noqa: E402

st.set_page_config(page_title="LLM Regression Detector", layout="wide")
st.title("LLM Regression Detector")

run_ids = list_run_ids()
if not run_ids:
    st.info("No runs yet. Run `python -m src.runner --prompt-version v1` first.")
    st.stop()

runs = {rid: load_run(rid) for rid in run_ids}
test_cases_by_id = {tc["id"]: tc for tc in load_golden_dataset()}

overview_rows = []
for rid in run_ids[-10:]:
    r = runs[rid]
    pass_rate = sum(1 for x in r["results"] if x["passed"]) / len(r["results"])
    status = "✅ pass" if pass_rate >= 0.9 else ("⚠️ warn" if pass_rate >= 0.85 else "🔴 fail")
    overview_rows.append(
        {
            "run_id": rid,
            "prompt_version": r["prompt_version"],
            "pass_rate": pass_rate,
            "status": status,
            "cost": r.get("cost", 0.0),
            "timestamp": r["timestamp"],
        }
    )

st.header("Recent runs")
st.dataframe(pd.DataFrame(overview_rows), use_container_width=True)

st.header("Trends")
trend_df = pd.DataFrame(
    [
        {
            "run_id": rid,
            "pass_rate": sum(1 for x in runs[rid]["results"] if x["passed"]) / len(runs[rid]["results"]),
            "avg_summary_relevance": sum(x["summary_relevance"] for x in runs[rid]["results"]) / len(runs[rid]["results"]),
            "avg_latency_ms": sum(x["latency_ms"] for x in runs[rid]["results"]) / len(runs[rid]["results"]),
        }
        for rid in run_ids
    ]
)
col1, col2, col3 = st.columns(3)
col1.plotly_chart(px.line(trend_df, x="run_id", y="pass_rate", title="Accuracy"), use_container_width=True)
col2.plotly_chart(px.line(trend_df, x="run_id", y="avg_summary_relevance", title="Summary quality"), use_container_width=True)
col3.plotly_chart(px.line(trend_df, x="run_id", y="avg_latency_ms", title="Latency (ms)"), use_container_width=True)

st.header("Compare two runs")
c1, c2 = st.columns(2)
run_a = c1.selectbox("Run A (baseline)", run_ids, index=max(0, len(run_ids) - 2))
run_b = c2.selectbox("Run B (current)", run_ids, index=len(run_ids) - 1)
if run_a and run_b:
    comparison = compare_runs(run_b, runs[run_b]["results"], run_a, runs[run_a]["results"], test_cases_by_id)
    st.metric("Pass rate delta", f"{comparison.pass_rate_delta:+.1%}", delta=f"{comparison.pass_rate_delta:.1%}")
    st.write("Regressions:", comparison.regressions or "none")
    st.write("Improvements:", comparison.improvements or "none")

st.header("Test case explorer")
selected_run = st.selectbox("Run", run_ids, index=len(run_ids) - 1, key="explorer_run")
df = pd.DataFrame(runs[selected_run]["results"])
df["expected_category"] = df["test_case_id"].map(lambda tc_id: test_cases_by_id[tc_id]["expected_category"])
df["difficulty"] = df["test_case_id"].map(lambda tc_id: test_cases_by_id[tc_id]["difficulty"])

f1, f2, f3 = st.columns(3)
category_filter = f1.multiselect("Category", sorted(df["expected_category"].unique()))
difficulty_filter = f2.multiselect("Difficulty", sorted(df["difficulty"].unique()))
pass_filter = f3.selectbox("Status", ["all", "passed", "failed"])

filtered = df.copy()
if category_filter:
    filtered = filtered[filtered["expected_category"].isin(category_filter)]
if difficulty_filter:
    filtered = filtered[filtered["difficulty"].isin(difficulty_filter)]
if pass_filter != "all":
    filtered = filtered[filtered["passed"] == (pass_filter == "passed")]

st.dataframe(filtered, use_container_width=True)
st.metric("Run cost", f"${runs[selected_run].get('cost', 0.0):.4f}")
