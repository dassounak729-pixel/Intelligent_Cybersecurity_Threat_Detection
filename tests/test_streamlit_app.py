"""Runs src/streamlit_app.py against a stand-in for Streamlit.

Real Streamlit is not needed: this catches wrong names, bad data handling and broken
control flow in the app script. It does NOT prove the real Streamlit API accepts every
call - run `python main.py dashboard` once to confirm that.
"""

import runpy
import sys
import types
import unittest
from unittest import mock

import config
from src import upload_analysis
from src.dashboard_state import DashboardState

try:
    from tests.helpers import make_results, make_upload_results
except ImportError:  # run via `unittest discover -s tests`
    from helpers import make_results, make_upload_results

PAGE_ANALYZE, PAGE_LIVE = "Analyze a dataset", "Live simulation"
_SUMMARY = {
    "episodes_detected": 1, "episodes_total": 1, "episode_detection_rate": 1.0,
    "normal_event_false_alert_rate": 0.0,
    "detection_by_class": {"DDoS": {"episodes": 1, "detected": 1, "detection_rate": 1.0,
                                    "median_seconds_to_detect": 0.0, "first_alert_type_correct": 1.0}},
}


class _Stop(Exception):
    """Raised by the fake st.stop()."""


class _Upload:
    """Stand-in for the object st.file_uploader returns."""

    def __init__(self, data: bytes):
        self._data = data
        self.name = "upload.csv"

    def getvalue(self) -> bytes:
        return self._data


class _Container:
    """Stand-in for columns / sidebar / expander / spinner: records calls, usable in `with`."""

    def __init__(self, fake):
        self._fake = fake

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        return getattr(self._fake, name)


class FakeStreamlit(types.ModuleType):
    def __init__(self, page=PAGE_ANALYZE, upload=None, pressed=(), hours=None, speed=None):
        super().__init__("streamlit")
        self.calls, self.page, self.upload, self.pressed = [], page, upload, set(pressed)
        self._hours, self._speed = hours, speed
        self.session_state = {}
        self.sidebar = _Container(self)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)

        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))
        return record

    # widgets that return values (each call is recorded too)
    def _widget(self, name, label, kwargs):
        self.calls.append((name, (label,), kwargs))

    def radio(self, label, options, **kwargs):
        self._widget("radio", label, kwargs)
        return self.page

    def file_uploader(self, label, **kwargs):
        self._widget("file_uploader", label, kwargs)
        return self.upload

    def button(self, label, **kwargs):
        self._widget("button", label, kwargs)
        return label in self.pressed

    def number_input(self, label, **kwargs):
        self._widget("number_input", label, kwargs)
        return self._hours if self._hours is not None else kwargs["value"]

    def selectbox(self, label, options, index=0, **kwargs):
        self._widget("selectbox", label, kwargs)
        return self._speed if self._speed is not None else options[index]

    def multiselect(self, label, options, default=None, **kwargs):
        self._widget("multiselect", label, kwargs)
        return default if default is not None else []

    def slider(self, label, **kwargs):
        self._widget("slider", label, kwargs)
        return kwargs["value"]

    def columns(self, spec):
        self.calls.append(("columns", (spec,), {}))
        return [_Container(self) for _ in range(spec if isinstance(spec, int) else len(spec))]

    def expander(self, *args, **kwargs):
        return _Container(self)

    def spinner(self, *args, **kwargs):
        return _Container(self)

    # decorators
    def cache_resource(self, func=None, **kwargs):
        return func if func is not None else (lambda f: f)

    def cache_data(self, func=None, **kwargs):
        return func if func is not None else (lambda f: f)

    def fragment(self, func=None, **kwargs):
        self.calls.append(("fragment", (), kwargs))
        return func if func is not None else (lambda f: f)

    def stop(self):
        raise _Stop()

    def names(self):
        return [name for name, _, _ in self.calls]

    def labels(self, name):
        return [args[0] for n, args, _ in self.calls if n == name and args]


class FakeRunner:
    started: list = []

    def __init__(self, bundle, state):
        self.state = state

    def start(self, hours, speed):
        FakeRunner.started.append((hours, speed))
        token = self.state.reset(hours, speed)
        self.state.begin(token, 6)
        self.state.record_batch(token, make_results())
        self.state.finish(token, _SUMMARY)


def _analysis():
    results = make_upload_results()
    return upload_analysis.AnalysisResult(
        results=results, summary=upload_analysis.summarize_results(results),
        evaluation=upload_analysis.evaluate_labels(results, ["Normal", "DDoS"]), notes=["Counters were computed."])


def _run_app(fake, data_source="events", bundle_error=None, analysis=None, analysis_error=None):
    FakeRunner.started = []
    bundle = types.SimpleNamespace(metadata={"data_source": data_source}, classes=["Normal", "DDoS"])
    load = mock.Mock(side_effect=bundle_error) if bundle_error else mock.Mock(return_value=bundle)
    analyze = mock.Mock(side_effect=analysis_error) if analysis_error else mock.Mock(return_value=analysis or _analysis())
    train = mock.Mock()
    with mock.patch.dict(sys.modules, {"streamlit": fake}), \
            mock.patch("src.models.load_bundle", load), \
            mock.patch("src.live_runner.LiveRunner", FakeRunner), \
            mock.patch("src.upload_analysis.analyze_bytes", analyze), \
            mock.patch("src.pipeline.train", train), \
            mock.patch.object(sys, "argv", ["streamlit_app.py", "--hours", "2", "--speed", "240"]):
        runpy.run_path(str(config.STREAMLIT_APP), run_name="streamlit_app_under_test")
    return analyze, train


class AnalyzePageTests(unittest.TestCase):
    def test_empty_page_offers_upload_and_sample(self):
        fake = FakeStreamlit()
        analyze, _ = _run_app(fake)
        names = fake.names()
        self.assertEqual(names[0], "set_page_config")
        self.assertIn("file_uploader", names)
        self.assertEqual(fake.labels("download_button"), ["Download a sample dataset"])
        self.assertNotIn("error", names)
        self.assertNotIn("metric", names)
        analyze.assert_not_called()

    def test_uploaded_file_is_analysed_and_results_shown(self):
        fake = FakeStreamlit(upload=_Upload(b"timestamp,src_ip\n1,2\n"))
        analyze, _ = _run_app(fake)
        analyze.assert_called_once()
        names = fake.names()
        self.assertNotIn("error", names)
        self.assertEqual(names.count("metric"), 6 + 4)  # headline cards + accuracy cards (file had labels)
        for element in ("bar_chart", "dataframe", "multiselect", "slider"):
            self.assertIn(element, names)
        self.assertIn("Download all alerts (CSV)", fake.labels("download_button"))

    def test_bad_upload_shows_the_error_message(self):
        fake = FakeStreamlit(upload=_Upload(b"x"))
        _run_app(fake, analysis_error=upload_analysis.UploadError("Missing required column(s): dst_ip."))
        errors = [args[0] for n, args, _ in fake.calls if n == "error"]
        self.assertEqual(errors, ["Missing required column(s): dst_ip."])
        self.assertNotIn("metric", fake.names())

    def test_sample_button_runs_the_built_in_sample(self):
        fake = FakeStreamlit(pressed={"Analyze the built-in sample instead"})
        analyze, _ = _run_app(fake)
        analyze.assert_called_once()
        self.assertEqual(fake.names().count("metric"), 10)

    def test_no_alerts_case_is_handled(self):
        results = make_upload_results()
        results["is_alert"] = False
        analysis = upload_analysis.AnalysisResult(results=results, summary=upload_analysis.summarize_results(results),
                                                  evaluation=None, notes=[])
        fake = FakeStreamlit(upload=_Upload(b"x"))
        _run_app(fake, analysis=analysis)
        self.assertNotIn("error", fake.names())
        self.assertIn("info", fake.names())

    def test_missing_model_offers_training(self):
        fake = FakeStreamlit(upload=_Upload(b"x"), pressed={"Train the model now"})
        _, train = _run_app(fake, bundle_error=FileNotFoundError("no model"))
        self.assertIn("warning", fake.names())
        train.assert_called_once_with(force_data=True)
        self.assertIn("rerun", fake.names())

    def test_missing_model_without_pressing_does_not_train(self):
        fake = FakeStreamlit(upload=_Upload(b"x"))
        _, train = _run_app(fake, bundle_error=FileNotFoundError("no model"))
        train.assert_not_called()
        self.assertNotIn("metric", fake.names())


class LivePageTests(unittest.TestCase):
    def test_live_page_renders_all_sections(self):
        fake = FakeStreamlit(page=PAGE_LIVE)
        _run_app(fake)
        names = fake.names()
        self.assertNotIn("error", names)
        self.assertEqual(names.count("metric"), 6)
        for element in ("area_chart", "bar_chart", "line_chart", "dataframe", "progress"):
            self.assertIn(element, names)
        refresh = [k for n, _, k in fake.calls if n == "fragment"][0]["run_every"]
        self.assertAlmostEqual(refresh, config.DASHBOARD_REFRESH_MS / 1000.0)

    def test_command_line_options_start_the_run(self):
        _run_app(FakeStreamlit(page=PAGE_LIVE))
        self.assertEqual(FakeRunner.started, [(2.0, 240.0)])

    def test_restart_button_uses_the_sidebar_settings(self):
        _run_app(FakeStreamlit(page=PAGE_LIVE, pressed={"Restart run"}, hours=3.0, speed=480))
        self.assertEqual(FakeRunner.started, [(3.0, 480.0)])

    def test_wrong_model_shows_error(self):
        fake = FakeStreamlit(page=PAGE_LIVE)
        _run_app(fake, data_source="aggregated")
        self.assertIn("error", fake.names())
        self.assertNotIn("metric", fake.names())

    def test_analyze_page_does_not_start_a_live_run(self):
        _run_app(FakeStreamlit())
        self.assertEqual(FakeRunner.started, [])


class LauncherTests(unittest.TestCase):
    def test_command_line_is_built_correctly(self):
        from src import pipeline
        command = pipeline.streamlit_command("127.0.0.1", 8050, 4.0, 120.0)
        self.assertEqual(command[1:5], ["-m", "streamlit", "run", str(config.STREAMLIT_APP)])
        self.assertIn("--server.headless", command)
        self.assertEqual(command[command.index("--server.port") + 1], "8050")
        self.assertEqual(command[command.index("--") + 1:], ["--hours", "4.0", "--speed", "120.0"])

    def test_missing_streamlit_gives_a_clear_error(self):
        from src import pipeline
        with mock.patch("importlib.util.find_spec", return_value=None):
            with self.assertRaises(pipeline.DashboardDependencyError) as caught:
                pipeline.run_streamlit_dashboard()
        self.assertIn("pip install", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
