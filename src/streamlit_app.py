"""Streamlit frontend.

Pages:
    Analyze a dataset   upload a CSV of network events and analyse it with the trained models
    Live simulation     watch a simulated live feed being scored in near real time

Run it with::

    python main.py dashboard
    streamlit run src/streamlit_app.py            # same thing, directly

All numbers come from the backend modules (``upload_analysis``, ``dashboard_state``);
display preparation lives in ``upload_view`` and ``dashboard_view`` so it can be tested
without a browser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `streamlit run src/streamlit_app.py` puts src/ (not the project root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from src import dashboard_view, pipeline, upload_analysis, upload_view  # noqa: E402
from src.dashboard_state import DashboardState  # noqa: E402
from src.live_runner import LiveRunner  # noqa: E402
from src.models import load_bundle  # noqa: E402

st.set_page_config(page_title="Threat Detection", layout="wide")

PAGE_ANALYZE = "Analyze a dataset"
PAGE_LIVE = "Live simulation"
DISCLAIMER = ("The detection models were trained on simulated network traffic. Results on other data are "
              "indicative only; retrain on your own labelled data before relying on them.")


def parse_args() -> argparse.Namespace:
    """Options passed after ``--`` on the streamlit command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=config.DASHBOARD_DEFAULT_HOURS)
    parser.add_argument("--speed", type=float, default=config.DASHBOARD_DEFAULT_SPEED)
    return parser.parse_known_args(sys.argv[1:])[0]


# ----------------------------------------------------------------------------- shared resources
@st.cache_resource
def get_bundle():
    """The trained models, loaded once per server process."""
    return load_bundle()


@st.cache_resource
def get_runtime():
    """Shared live-run state and runner (the run itself starts only when the live page is opened)."""
    state = DashboardState()
    return state, LiveRunner(get_bundle(), state)


@st.cache_data(show_spinner=False)
def sample_csv() -> bytes:
    """The downloadable sample dataset."""
    return upload_analysis.make_sample_events().to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner=False, max_entries=3)
def run_analysis(data: bytes):
    """Read, validate, prepare and score an uploaded file (cached per file content)."""
    return upload_analysis.analyze_bytes(get_bundle(), data)


def ensure_model() -> bool:
    """True when a trained model is available; otherwise offer to train one."""
    try:
        get_bundle()
        return True
    except FileNotFoundError:
        st.warning("No trained model was found. Training one takes about two minutes and only has to be done once.")
        if st.button("Train the model now"):
            with st.spinner("Training... this takes about two minutes."):
                pipeline.train(force_data=True)
            clear = getattr(get_bundle, "clear", None)
            if clear:
                clear()
            st.rerun()
        return False


# ----------------------------------------------------------------------------- dataset analysis page
def show_cards(cards: list[dict]) -> None:
    for column, card in zip(st.columns(len(cards)), cards):
        column.metric(card["label"], card["value"], help=card["help"])


def render_analysis(analysis) -> None:
    """Results of an analysed upload."""
    summary, results = analysis.summary, analysis.results
    with st.expander("How your file was prepared", expanded=False):
        for note in analysis.notes or ["The file was used as it is."]:
            st.write("- " + note)
    show_cards(upload_view.kpi_cards(summary))

    left, right = st.columns(2)
    left.subheader("Events and alerts over time")
    right.subheader("Threat score distribution")
    series = upload_view.time_series(results)
    left.bar_chart(series[["Alerts"]])
    right.bar_chart(upload_view.score_histogram(results))

    left, right = st.columns(2)
    left.subheader("Alerts by threat type")
    right.subheader("Recommended actions")
    if summary["alerts"]:
        left.bar_chart(upload_view.counts_frame(summary["types"], "Alerts"))
        right.bar_chart(upload_view.counts_frame(summary["actions"], "Alerts"))
    else:
        left.info("No alerts were raised.")
        right.info("No alerts were raised.")

    if analysis.evaluation:
        st.subheader("Accuracy against the labels in your file")
        show_cards(upload_view.evaluation_cards(analysis.evaluation))
        st.dataframe(upload_view.evaluation_table(analysis.evaluation), hide_index=True)
        st.caption("Any label other than Normal/Benign is treated as an attack.")

    st.subheader("Alerts")
    if not summary["alerts"]:
        st.info("No event reached the alert threshold.")
        return
    types = sorted(summary["types"])
    severities = ["Critical", "High", "Medium", "Low", "Info"]
    present = [s for s in severities if s in summary["severities"]]
    col1, col2, col3 = st.columns(3)
    chosen_severity = col1.multiselect("Severity", present, default=present)
    chosen_types = col2.multiselect("Threat type", types, default=types)
    min_score = col3.slider("Minimum score", min_value=float(summary["alert_threshold"]), max_value=100.0,
                            value=float(summary["alert_threshold"]), step=1.0)
    table = upload_view.alerts_table(results, chosen_severity, min_score, chosen_types)
    st.caption(f"Showing the {len(table):,} highest-scoring alerts that match the filters "
               f"(of {summary['alerts']:,}). The download contains all alerts.")
    st.dataframe(table, hide_index=True)
    st.download_button("Download all alerts (CSV)", data=upload_view.export_csv(results),
                       file_name="alerts.csv", mime="text/csv")


def analyze_page() -> None:
    st.title("Analyze a dataset")
    st.write("Upload a CSV of network events. The app computes behavioural counters, scores every event with the "
             "trained models, and shows alerts with the threat type, risk score, recommended action and evidence.")
    with st.expander("What should the CSV look like?"):
        st.markdown(
            f"**Required columns:** `{', '.join(upload_analysis.REQUIRED_UPLOAD_COLUMNS)}`  \n"
            f"**Optional columns:** `{', '.join(upload_analysis.OPTIONAL_UPLOAD_COLUMNS)}` "
            "(defaults are used when they are missing; a `label` column adds an accuracy report).  \n"
            f"Up to {config.UPLOAD_MAX_ROWS:,} rows and {config.UPLOAD_MAX_MB} MB. Common headers from other "
            "tools (for example CICFlowMeter's *Source IP*, *Destination Port*, *Flow Duration*) are recognised. "
            "Timestamps should look like `2026-01-06 08:00:00`.")
    st.download_button("Download a sample dataset", data=sample_csv(), file_name="sample_events.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if st.button("Analyze the built-in sample instead"):
        st.session_state["use_sample"] = True
    data = None
    if uploaded is not None:
        st.session_state["use_sample"] = False
        data = uploaded.getvalue()
    elif st.session_state.get("use_sample"):
        data = sample_csv()
        st.info("Showing the built-in sample dataset.")
    if data is None:
        st.info("Upload a CSV file or use the sample to begin.")
        st.caption(DISCLAIMER)
        return
    if not ensure_model():
        return
    try:
        with st.spinner("Analysing..."):
            analysis = run_analysis(data)
    except upload_analysis.UploadError as exc:
        st.error(str(exc))
        return
    render_analysis(analysis)
    st.caption(DISCLAIMER)


# ----------------------------------------------------------------------------- live simulation page
@st.fragment(run_every=config.DASHBOARD_REFRESH_MS / 1000.0)
def live_view(state: DashboardState) -> None:
    """Redrawn every second without rerunning the whole page."""
    snap = state.snapshot()
    if snap["status"] == "error":
        st.error(f"Run failed: {snap['message']}")
    st.progress(min(max(snap["progress"], 0.0), 1.0),
                text=f"{snap['status'].capitalize()} - simulated clock {snap['sim_clock'] or '-'}")
    show_cards(dashboard_view.kpi_cards(snap))

    left, right = st.columns(2)
    events, alerts = dashboard_view.events_frame(snap), dashboard_view.alerts_per_minute_frame(snap)
    left.subheader("Events per simulated minute")
    right.subheader("Alerts per simulated minute")
    if events.empty:
        left.info("Waiting for data...")
        right.info("Waiting for data...")
    else:
        left.area_chart(events)
        right.bar_chart(alerts)

    left, right = st.columns(2)
    left.subheader("Alerts by threat type")
    right.subheader("Scoring latency per batch (ms)")
    if snap["types"]:
        left.bar_chart(dashboard_view.counts_frame(snap["types"], "Alerts"))
    else:
        left.info("No alerts yet.")
    if snap["batches"]:
        right.line_chart(dashboard_view.latency_frame(snap))
    else:
        right.info("Waiting for data...")

    left, right = st.columns(2)
    left.subheader("Recommended actions")
    right.subheader("Severity")
    if snap["actions"]:
        left.bar_chart(dashboard_view.counts_frame(snap["actions"], "Alerts"))
        right.bar_chart(dashboard_view.counts_frame(snap["severities"], "Alerts"))
    else:
        left.info("No alerts yet.")
        right.info("No alerts yet.")

    table = dashboard_view.summary_table(snap)
    if table is not None:
        st.subheader("Run summary - attack episodes (simulator ground truth)")
        st.caption(dashboard_view.summary_headline(snap))
        st.dataframe(table, hide_index=True)

    st.subheader("Recent alerts")
    recent = dashboard_view.alerts_table(snap)
    if recent.empty:
        st.info("No alerts yet.")
    else:
        st.dataframe(recent, hide_index=True)


def live_page(args: argparse.Namespace) -> None:
    st.title("Live simulation")
    st.caption("A simulated event feed scored as events arrive. Ground-truth cards exist only because the feed is a simulator.")
    if not ensure_model():
        return
    if get_bundle().metadata.get("data_source") != "events":
        st.error("This model was trained on pre-aggregated records. Retrain with: "
                 "python main.py train --source events --force-data")
        return
    state, runner = get_runtime()
    with st.sidebar:
        st.header("Run settings")
        hours = st.number_input("Simulated hours", min_value=config.DASHBOARD_HOURS_RANGE[0],
                                max_value=config.DASHBOARD_HOURS_RANGE[1], value=float(args.hours), step=0.5)
        speeds = sorted(set(config.STREAMLIT_SPEEDS) | {int(round(args.speed))})
        speed = st.selectbox("Speed (simulated seconds per real second)", speeds,
                             index=speeds.index(int(round(args.speed))))
        restart = st.button("Restart run")
    if restart:
        runner.start(float(hours), float(speed))
    elif state.snapshot()["status"] == "idle":
        runner.start(float(args.hours), float(args.speed))
    live_view(state)


def main() -> None:
    args = parse_args()
    page = st.sidebar.radio("Page", [PAGE_ANALYZE, PAGE_LIVE])
    if page == PAGE_ANALYZE:
        analyze_page()
    else:
        live_page(args)


main()
