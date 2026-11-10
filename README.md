# Intelligent Cybersecurity Threat Detection

AI-driven threat detection for banking networks. It learns what is normal for each host, flags
unusual activity as events arrive, names the threat (MITRE ATT&CK aligned), gives a 0-100 risk
score, recommends a response and explains why.

* **Streamlit web app** - upload a dataset and analyse it, or watch a live simulation
* Two models: a Random Forest names known attacks; an Isolation Forest catches unusual behaviour
* Per-host baselines (UEBA), live sliding-window features, duplicate-alert suppression
* Pure Python scripts, configuration in `config.py`, 96 automated tests, no notebooks

## Quick start

```bash
pip install -r requirements.txt
python main.py dashboard          # opens http://127.0.0.1:8050
```

On a fresh clone there is no trained model yet. Open the **Analyze a dataset** page; it offers a
**Train the model now** button (about 2 minutes, once). You can also train from the terminal:

```bash
python main.py train --source events --force-data
```

## The Streamlit app

| Page | What it does |
|---|---|
| **Analyze a dataset** | Upload a CSV, or use the built-in sample. Shows alerts over time, score distribution, threat types, recommended actions, a filterable alerts table with evidence, an accuracy report if the file has a `label` column, and a CSV download of all alerts. |
| **Live simulation** | Replays a simulated event feed in accelerated real time and shows throughput, latency, alerts, and a per-attack summary. Sidebar controls restart the run. |

### Dataset format

A CSV with one row per network event.

| Required | `timestamp`, `src_ip`, `dst_ip`, `dst_port`, `protocol`, `src_bytes`, `dst_bytes` |
|---|---|
| **Optional** | `duration`, `src_packets`, `dst_packets`, `syn_ratio`, `auth_failed`, `asset_criticality`, `src_port`, `user`, `device`, `flow_id`, `label` |

* Missing optional columns get documented defaults; the page lists everything it adjusted.
* Behavioural counters (connections, distinct ports/hosts, failed logins, DNS rate, beacon
  regularity) are computed automatically over a 60-second sliding window per source IP. If your
  file already contains those six columns they are used as they are.
* Headers from other tools are recognised (for example CICFlowMeter's *Source IP*,
  *Destination Port*, *Flow Duration*).
* Limits: 100,000 rows and 50 MB per file.
* `label`: any value other than Normal/Benign is treated as an attack.

Get a ready-made example with `python main.py sample` (writes `data/sample/sample_events.csv`,
already included in the repository) or the **Download a sample dataset** button.

> The models were trained on simulated traffic. Results on other data are indicative only;
> retrain on your own labelled data before relying on them.

## Command line

```bash
python main.py train --source events --force-data    # simulate data, train, evaluate, write reports
python main.py live --hours 2 --speed 240            # live detection in the terminal
python main.py dashboard                             # Streamlit app
python main.py dashboard --ui web                    # built-in page without Streamlit
python main.py detect --input data/processed/test_flows.csv
python main.py demo --count 4
python main.py sample
python -m unittest discover -s tests                 # 96 tests
```

## Project structure

```
threat_detection/
├── main.py                  # command-line entry point
├── config.py                # all settings
├── requirements.txt
├── .streamlit/config.toml   # Streamlit settings (upload limit)
├── data/sample/             # sample_events.csv for the upload page
├── src/
│   ├── streamlit_app.py     # Streamlit frontend (upload analysis + live simulation)
│   ├── upload_analysis.py   # read, validate, prepare and score an uploaded CSV
│   ├── upload_view.py       # tables and numbers for the upload page
│   ├── dashboard_view.py    # tables and numbers for the live page
│   ├── stream_features.py   # sliding-window counters
│   ├── entity_baseline.py   # per-host baselines (UEBA)
│   ├── models.py  risk_scoring.py  response.py  explainability.py  detection.py
│   ├── event_simulator.py  event_source.py  live_detector.py  live_runner.py
│   ├── dashboard_state.py  dashboard_server.py  dashboard/index.html  (built-in page)
│   ├── data_loader.py  preprocessing.py  synthetic_data.py
│   ├── evaluation.py  visualization.py  reporting.py  pipeline.py  utils.py
├── tests/                   # 96 unit and integration tests
└── outputs/                 # figures, reports, alerts (models and logs are generated)
```

## How it works

1. Events arrive; a 60-second window per source computes behavioural counters.
2. Each event is compared with its own host's history (per-host baseline).
3. The Random Forest names the threat; the Isolation Forest scores how unusual it is.
4. Risk score = 0.65 x attack likelihood + 0.35 x anomaly, scaled by asset importance.
5. A score of 60 or more raises an alert, with a recommended action, MITRE ATT&CK technique
   and the features that caused it. Repeat alerts are suppressed for 60 seconds.

## Publishing to GitHub

```bash
cd threat_detection
git init
git add .
git commit -m "Intelligent Cybersecurity Threat Detection"
git branch -M main
git remote add origin https://github.com/<your-user>/<repo-name>.git
git push -u origin main
```

`.gitignore` keeps generated data and the 60 MB trained model out of the repository; anyone who
clones it can recreate them with one command (or the app's **Train the model now** button). Make
the repository **public** (or share access) so the link opens for the reviewer.

## Notes

All traffic is simulated, so the reported metrics show that the pipeline works, not how well it
would perform on a real network. The system only recommends actions; it never blocks anything.
