"""Command-line entry point for the Intelligent Cybersecurity Threat Detection system.

Examples:
    python main.py train              # generate data, train, evaluate, write reports
    python main.py detect --input data/processed/test_flows.csv
    python main.py stream --batch-size 500
    python main.py demo
    python main.py live --hours 2 --speed 240   # simulated live detection
    python main.py dashboard                    # Streamlit app (dataset upload + live simulation) on http://127.0.0.1:8050
    python main.py sample                       # write data/sample/sample_events.csv for the upload page
    python main.py dashboard --ui web           # built-in page (no extra install)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config
from src import pipeline
from src.data_loader import DataLoadError
from src.utils import ensure_directories, get_logger, setup_logging


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Intelligent Cybersecurity Threat Detection")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate synthetic flow datasets")
    gen.add_argument("--flows", type=int, default=config.N_FLOWS)
    gen.add_argument("--force", action="store_true", help="Overwrite existing data")
    gen.add_argument("--source", choices=["events", "aggregated"], help="Data source (default: config.DATA_SOURCE)")

    train = sub.add_parser("train", help="Train and evaluate the models")
    train.add_argument("--flows", type=int, default=config.N_FLOWS)
    train.add_argument("--force-data", action="store_true", help="Regenerate raw data first")
    train.add_argument("--source", choices=["events", "aggregated"], help="Data source (default: config.DATA_SOURCE)")

    detect = sub.add_parser("detect", help="Score a flow CSV and export alerts")
    detect.add_argument("--input", type=Path, default=config.TEST_FILE)
    detect.add_argument("--output", type=Path, default=config.ALERTS_FILE)
    detect.add_argument("--min-score", type=float, default=None, help="Alert threshold (0-100)")

    stream = sub.add_parser("stream", help="Simulate near-real-time detection")
    stream.add_argument("--input", type=Path, default=config.TEST_FILE)
    stream.add_argument("--batch-size", type=int, default=config.STREAM_BATCH_SIZE)

    live = sub.add_parser("live", help="Simulate a live event feed and detect threats as events arrive")
    live.add_argument("--hours", type=float, default=config.LIVE_RUN_HOURS, help="Simulated duration")
    live.add_argument("--speed", type=float, default=config.LIVE_DEFAULT_SPEED,
                      help="Simulated seconds per wall-clock second (0 = as fast as possible)")
    live.add_argument("--seed", type=int, default=None)
    live.add_argument("--quiet", action="store_true", help="Do not print individual alerts")

    dash = sub.add_parser("dashboard", help="Serve a live web dashboard (simulated feed)")
    dash.add_argument("--ui", choices=["streamlit", "web"], default="streamlit",
                      help="streamlit (needs `pip install streamlit`) or web (built-in, no extra install)")
    dash.add_argument("--host", default=config.DASHBOARD_HOST, help="Bind address (default: this computer only)")
    dash.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    dash.add_argument("--hours", type=float, default=config.DASHBOARD_DEFAULT_HOURS, help="Simulated duration per run")
    dash.add_argument("--speed", type=float, default=config.DASHBOARD_DEFAULT_SPEED,
                      help="Simulated seconds per wall-clock second")
    dash.add_argument("--no-browser", action="store_true", help="Do not open a browser window")

    sample = sub.add_parser("sample", help="Write a small sample dataset for the upload page")
    sample.add_argument("--hours", type=float, default=config.SAMPLE_HOURS)

    demo = sub.add_parser("demo", help="Print example alerts")
    demo.add_argument("--count", type=int, default=6)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the requested command; returns a process exit code."""
    args = build_parser().parse_args(argv)
    ensure_directories()
    setup_logging()
    logger = get_logger("main")
    if getattr(args, "source", None):
        config.DATA_SOURCE = args.source
    try:
        if args.command == "generate":
            pipeline.generate_data(args.flows, force=args.force)
        elif args.command == "train":
            pipeline.train(args.flows, force_data=args.force_data)
        elif args.command == "detect":
            pipeline.detect(args.input, args.output, args.min_score)
        elif args.command == "stream":
            summary = pipeline.stream(args.input, args.batch_size)
            logger.info("Streaming summary: %s", summary)
        elif args.command == "live":
            pipeline.run_live(args.hours, args.speed, args.seed, args.quiet)
        elif args.command == "sample":
            from src import upload_analysis
            config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
            upload_analysis.make_sample_events(args.hours).to_csv(config.SAMPLE_FILE, index=False)
            logger.info("Sample dataset written to %s", config.SAMPLE_FILE)
        elif args.command == "dashboard":
            if args.ui == "web":
                pipeline.run_dashboard(args.host, args.port, args.hours, args.speed, not args.no_browser)
            else:
                return pipeline.run_streamlit_dashboard(args.host, args.port, args.hours, args.speed,
                                                        not args.no_browser)
        elif args.command == "demo":
            print(pipeline.demo(args.count))
    except (DataLoadError, FileNotFoundError, OSError, pipeline.ModelCompatibilityError,
            pipeline.DashboardDependencyError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
