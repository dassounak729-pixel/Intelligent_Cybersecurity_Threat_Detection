"""Pipeline orchestration: generate data, train, detect, stream and demo."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

import config
from src import (data_loader, entity_baseline, evaluation, event_simulator, live_detector, preprocessing,
                 reporting, synthetic_data, visualization)
from src.dashboard_server import DashboardServer
from src.dashboard_state import DashboardState
from src.event_source import PacedEventSource, ReplayEventSource
from src.live_runner import LiveRunner
from src.detection import ThreatDetector
from src.models import (AnomalyDetector, ModelBundle, build_baseline, compare_models,
                        compute_feature_importances, load_bundle, save_bundle, train_classifier)
from src.utils import ensure_directories, get_logger, save_json, timed

logger = get_logger("pipeline")


class ModelCompatibilityError(Exception):
    """Raised when a saved model cannot be used for the requested operation."""


class DashboardDependencyError(Exception):
    """Raised when an optional dashboard dependency (Streamlit) is missing."""


def generate_data(n_flows: int = config.N_FLOWS, force: bool = False) -> None:
    """Create the raw datasets if absent (or when ``force`` is True).

    Uses ``config.DATA_SOURCE``: "events" (simulated raw events with live-computed
    features; ``n_flows`` is ignored) or "aggregated" (original flow-record sampler).
    """
    ensure_directories()
    if config.DATA_SOURCE == "events":
        if force or not config.RAW_FLOWS_FILE.exists():
            data_loader.save_flows(event_simulator.build_training_flows(), config.RAW_FLOWS_FILE)
        if force or not config.NOVEL_FLOWS_FILE.exists():
            data_loader.save_flows(event_simulator.build_novel_flows(), config.NOVEL_FLOWS_FILE)
        return
    if force or not config.RAW_FLOWS_FILE.exists():
        data_loader.save_flows(synthetic_data.generate_flows(n_flows), config.RAW_FLOWS_FILE)
    if force or not config.NOVEL_FLOWS_FILE.exists():
        data_loader.save_flows(synthetic_data.generate_novel_flows(), config.NOVEL_FLOWS_FILE)


def train(n_flows: int = config.N_FLOWS, force_data: bool = False) -> dict:
    """Full training pipeline. Returns the metrics dictionary (also saved as JSON)."""
    generate_data(n_flows, force=force_data)
    raw = data_loader.load_flows(config.RAW_FLOWS_FILE, require_label=True)
    data_source = "events" if "episode_id" in raw.columns else "aggregated"
    if data_source != config.DATA_SOURCE:
        logger.warning("Raw data is '%s' but DATA_SOURCE is '%s'; re-run with --force-data to regenerate.",
                       data_source, config.DATA_SOURCE)
    flows = preprocessing.clean_flows(raw)
    train_df, test_df = preprocessing.split_dataset(flows)
    data_loader.save_flows(train_df, config.TRAIN_FILE)
    data_loader.save_flows(test_df, config.TEST_FILE)
    X_train, y_train = preprocessing.prepare_xy(train_df)
    X_test, y_test = preprocessing.prepare_xy(test_df)
    entity_store = None
    if config.USE_ENTITY_BASELINES:
        benign_mask = (train_df[config.LABEL_COLUMN] == config.NORMAL_LABEL).to_numpy()
        X_train = X_train.join(entity_baseline.cross_fit_features(train_df, benign_mask))
        entity_store = entity_baseline.EntityBaselineStore().fit(train_df[benign_mask])
        X_test = X_test.join(entity_store.transform(test_df))
    metrics: dict = {"dataset": {"data_source": data_source, "total_flows": len(flows), "train": len(train_df), "test": len(test_df),
                                 "class_counts": flows[config.LABEL_COLUMN].value_counts().to_dict(),
                                 "n_features": X_train.shape[1],
                                 "entity_baselines": entity_store is not None,
                                 "entities_profiled": entity_store.n_entities if entity_store else 0}}

    with timed("model comparison"):
        comparison = compare_models(X_train, y_train)
    comparison.to_csv(config.MODEL_COMPARISON_FILE, index=False)
    best_name = comparison.iloc[0]["model"]
    logger.info("Selected classifier: %s", best_name)

    with timed("final training") as clock:
        classifier = train_classifier(best_name, X_train, y_train)
        benign_train = X_train[y_train == config.NORMAL_LABEL]
        anomaly_detector = AnomalyDetector().fit(benign_train)
        importances = compute_feature_importances(classifier, X_train, y_train)
        median, log_mean, log_std = build_baseline(benign_train)
    bundle = ModelBundle(classifier=classifier, anomaly_detector=anomaly_detector,
                         feature_columns=list(X_train.columns), classes=list(classifier.classes_),
                         importances=importances, baseline_median=median,
                         baseline_log_mean=log_mean, baseline_log_std=log_std,
                         metadata={"selected_model": best_name, "seed": config.RANDOM_SEED, "data_source": data_source},
                         entity_baseline=entity_store)
    save_bundle(bundle)
    metrics["model_selection"] = {"selected": best_name, "training_seconds": clock["seconds"],
                                  "comparison": comparison.to_dict(orient="records")}

    classes = list(classifier.classes_)
    metrics["classifier_test"] = evaluation.evaluate_classifier(
        y_test, classifier.predict(X_test), [l for l in config.ALL_LABELS if l in classes])

    detector = ThreatDetector(bundle)
    test_results = detector.analyze(test_df)
    metrics["detection_test"] = evaluation.evaluate_detection(test_results)
    metrics["response_test"] = evaluation.evaluate_action_distribution(test_results)
    metrics["top_features"] = importances.head(10).to_dict()

    novel_results = detector.analyze(data_loader.load_flows(config.NOVEL_FLOWS_FILE, require_label=True))
    metrics["novel_threats"] = evaluation.evaluate_novel(novel_results)
    metrics["streaming"] = _stream_benchmark(detector, test_df)

    _render_figures(flows, comparison, metrics, importances, test_results)
    reporting.write_alerts(test_results)
    reporting.write_sample_alerts(test_results)
    save_json(metrics, config.METRICS_FILE)
    logger.info("Accuracy=%.4f macroF1=%.4f | alert precision=%.4f recall=%.4f FPR=%.4f",
                metrics["classifier_test"]["accuracy"], metrics["classifier_test"]["f1_macro"],
                metrics["detection_test"]["precision"], metrics["detection_test"]["recall"],
                metrics["detection_test"]["false_positive_rate"])
    return metrics


def _stream_benchmark(detector: ThreatDetector, frame: pd.DataFrame) -> dict:
    """Run the streaming simulation and report latency/throughput."""
    latencies, sizes = [], []
    for result, latency in detector.analyze_stream(frame):
        latencies.append(latency)
        sizes.append(len(result))
    return evaluation.measure_throughput(latencies, sizes)


def _render_figures(flows, comparison, metrics, importances, test_results) -> None:
    """Generate all report figures."""
    fig_dir = config.FIGURES_DIR
    visualization.plot_class_distribution(flows[config.LABEL_COLUMN], fig_dir / "class_distribution.png")
    visualization.plot_model_comparison(comparison, fig_dir / "model_comparison.png")
    cls = metrics["classifier_test"]
    visualization.plot_confusion_matrix(cls["confusion_matrix"], cls["labels"], fig_dir / "confusion_matrix.png")
    visualization.plot_feature_importance(importances, fig_dir / "feature_importance.png")
    visualization.plot_score_distribution(test_results, fig_dir / "threat_score_distribution.png")
    visualization.plot_alert_timeline(test_results, fig_dir / "alert_timeline.png")
    visualization.plot_alert_breakdown(test_results, fig_dir / "alert_breakdown.png")


def detect(input_path: Path, output_path: Path = config.ALERTS_FILE,
           min_score: float | None = None) -> pd.DataFrame:
    """Score a flow CSV with the trained model and export alerts."""
    if min_score is not None:
        config.ALERT_SCORE_THRESHOLD = min_score
    detector = ThreatDetector(load_bundle())
    results = detector.analyze(data_loader.load_flows(input_path))
    results["is_alert"] = results["threat_score"] >= config.ALERT_SCORE_THRESHOLD
    reporting.write_alerts(results, output_path)
    logger.info("%d of %d flows raised alerts", int(results["is_alert"].sum()), len(results))
    return results


def stream(input_path: Path, batch_size: int = config.STREAM_BATCH_SIZE) -> dict:
    """Simulate near-real-time detection and print per-batch status."""
    detector = ThreatDetector(load_bundle())
    frame = data_loader.load_flows(input_path)
    latencies, sizes = [], []
    for number, (result, latency) in enumerate(detector.analyze_stream(frame, batch_size), start=1):
        latencies.append(latency)
        sizes.append(len(result))
        logger.info("batch %03d | flows=%d alerts=%d latency=%.0fms", number, len(result),
                    int(result["is_alert"].sum()), latency * 1000)
    return evaluation.measure_throughput(latencies, sizes)


def demo(count: int = 6) -> str:
    """Show formatted example alerts from the held-out test file."""
    results = ThreatDetector(load_bundle()).analyze(data_loader.load_flows(config.TEST_FILE))
    return reporting.write_sample_alerts(results, count=count)


def run_live(hours: float = config.LIVE_RUN_HOURS, speed: float = config.LIVE_DEFAULT_SPEED,
             seed: int | None = None, quiet: bool = False) -> dict:
    """Simulate a live event feed, detect threats as events arrive, and report latency.

    ``speed`` is simulated seconds per wall-clock second; ``speed <= 0`` replays as
    fast as possible (throughput test). Alerts stream to the console and to
    ``config.LIVE_ALERTS_FILE``.
    """
    bundle = load_bundle()
    if bundle.metadata.get("data_source") != "events":
        raise ModelCompatibilityError(
            "This model was trained on pre-aggregated records, so its features would not match the live "
            "counters. Retrain on event data: python main.py train --source events --force-data")
    seed = config.RANDOM_SEED + config.LIVE_SEED_OFFSET if seed is None else seed
    events = event_simulator.simulate_events(hours, config.LIVE_START,
                                             event_simulator.scaled_episodes(hours), seed)
    source = (PacedEventSource(events, speed) if speed > 0
              else ReplayEventSource(events))
    config.LIVE_ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    with config.LIVE_ALERTS_FILE.open("w", encoding="utf-8") as sink:
        def on_alert(alert: pd.Series) -> None:
            sink.write(json.dumps(_alert_record(alert), default=str) + "\n")
            if not quiet:
                print(f"[{alert['timestamp']:%H:%M:%S}] {alert['severity']:<8} {alert['src_ip']:<15} -> "
                      f"{alert['dst_ip']}:{int(alert['dst_port'])}  {alert['threat_type']:<18} "
                      f"score={alert['threat_score']:5.1f}  action={alert['action']:<11} "
                      f"latency={alert.get('latency_ms', float('nan')):.0f}ms", flush=True)

        detector = live_detector.LiveDetector(bundle, on_alert=on_alert)
        started = time.perf_counter()
        for batch in source.batches():
            frames.append(detector.process_batch(batch))
        wall = time.perf_counter() - started
    summary = live_detector.summarize(pd.concat(frames, ignore_index=True), wall, config.ALERT_COOLDOWN_SECONDS)
    save_json(summary, config.LIVE_SUMMARY_FILE)
    print(live_detector.format_summary(summary))
    return summary


def _alert_record(alert: pd.Series) -> dict:
    """Fields written to the live alert log (one JSON object per line)."""
    fields = ["timestamp", "src_ip", "dst_ip", "dst_port", "protocol", "user", "device", "threat_score",
              "threat_type", "confidence", "severity", "action", "mitre_tactic", "mitre_technique",
              "entity_deviation", "latency_ms", "evidence"]
    return {name: alert[name] for name in fields if name in alert.index}


def run_dashboard(host: str = config.DASHBOARD_HOST, port: int = config.DASHBOARD_PORT,
                  hours: float = config.DASHBOARD_DEFAULT_HOURS, speed: float = config.DASHBOARD_DEFAULT_SPEED,
                  open_browser: bool = True) -> None:
    """Serve the live dashboard and run a simulated live feed behind it (Ctrl+C to stop)."""
    import webbrowser

    bundle = load_bundle()
    if bundle.metadata.get("data_source") != "events":
        raise ModelCompatibilityError(
            "This model was trained on pre-aggregated records, so its features would not match the live "
            "counters. Retrain on event data: python main.py train --source events --force-data")
    state = DashboardState()
    runner = LiveRunner(bundle, state)
    server = DashboardServer(state, runner.start, host, port)
    bound_host, bound_port = server.address
    url = f"http://{bound_host}:{bound_port}/"
    runner.start(hours, speed)
    print(f"Dashboard running at {url}   (press Ctrl+C to stop)", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:  # headless machines: the URL above is enough
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
    finally:
        runner.stop()
        server.close()


def streamlit_command(host: str, port: int, hours: float, speed: float) -> list[str]:
    """The command line that starts the Streamlit dashboard."""
    import sys
    return [sys.executable, "-m", "streamlit", "run", str(config.STREAMLIT_APP),
            "--server.address", host, "--server.port", str(port),
            "--server.headless", "true",              # never block on Streamlit's first-run email prompt
            "--browser.gatherUsageStats", "false",
            "--", "--hours", str(hours), "--speed", str(speed)]


def run_streamlit_dashboard(host: str = config.DASHBOARD_HOST, port: int = config.DASHBOARD_PORT,
                            hours: float = config.DASHBOARD_DEFAULT_HOURS,
                            speed: float = config.DASHBOARD_DEFAULT_SPEED, open_browser: bool = True) -> int:
    """Launch the Streamlit app (Ctrl+C to stop). Returns the process exit code.

    No model is required to start: the app offers to train one on first use.
    """
    import importlib.util
    import subprocess
    import threading
    import webbrowser

    if importlib.util.find_spec("streamlit") is None:
        raise DashboardDependencyError(
            'Streamlit is not installed. Install it with:  pip install "streamlit>=1.37"   '
            "(or use the built-in page instead:  python main.py dashboard --ui web)")
    url = f"http://{host}:{port}/"
    print(f"Starting Streamlit dashboard at {url}   (press Ctrl+C to stop)", flush=True)
    if open_browser:
        timer = threading.Timer(3.0, webbrowser.open, args=[url])
        timer.daemon = True
        timer.start()
    try:
        return subprocess.call(streamlit_command(host, port, hours, speed), cwd=config.PROJECT_ROOT)
    except KeyboardInterrupt:
        return 0
