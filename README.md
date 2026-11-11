# UrbanFlow AI - Spatio-Temporal Graph Learning for Urban Traffic Forecasting and Analysis

UrbanFlow AI is an MSc Data Science and Artificial Intelligence capstone project investigating short-term traffic-speed forecasting on the METR-LA road-sensor network.

## Research question

> To what extent does incorporating spatial road-network relationships improve multi-horizon traffic-speed forecasting compared with conventional and temporal-only forecasting approaches?

The study does not assume that a graph model will perform best. Negative or mixed results are valid if fair held-out experiments show that spatial information provides limited or conditional benefit.

## Dataset and forecasting setup

METR-LA contains traffic speeds from **207 sensors** across **34,272 observations** sampled every five minutes. Task 02 verified these dimensions, the timestamp cadence, and sensor ordering across the speed data and graph metadata.

Later experiments will use 12 historical steps (60 minutes) to predict 12 future steps (60 minutes), with evaluation at **+15, +30, and +60 minutes** (output steps 3, 6, and 12). Dataset sources, paths, hashes, and safety notes are in [data/README.md](data/README.md).

## Planned models

The planned comparison is **Persistence**, **Historical Average**, **XGBoost**, one **LSTM or GRU**, and one **spatio-temporal graph neural network (ST-GNN)**. Tasks 04 and 05 implement the deterministic baselines and conventional XGBoost model; temporal and graph models remain future work.

## Current repository structure

```text
Capstone-Project_268513X/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- README.md
|   `-- raw/                         # Local METR-LA and optional PEMS-BAY files
|-- notebooks/
|   |-- 01_metr_la_eda.ipynb        # Completed Task 02 notebook
|   |-- 02_metr_la_preprocessing.ipynb # Completed Task 03 notebook
|   |-- 03_metr_la_baselines.ipynb  # Completed Task 04 notebook
|   `-- 04_metr_la_xgboost.ipynb   # Task 05 artifact-review notebook
|-- scripts/check_environment.py
|-- scripts/smoke_preprocessing.py  # Real-data Task 03 verification
|-- scripts/run_xgboost.py          # Explicit --smoke or --full mode
|-- src/data/                       # Inspection, preprocessing, and windows
|-- src/models/xgboost.py           # Sensor-local pooled features
|-- src/training/xgboost.py         # Deterministic training/inference
`-- tests/                          # Data integrity and preprocessing tests
```

Task 04 adds reusable baseline models under `src/models/`, shared masked metrics
under `src/evaluation/`, and `scripts/run_baselines.py`. Backend and frontend
components remain planned work.

## Leakage-safe preprocessing

Task 03 uses chronological 70%/10%/20% row ranges and 12-input/12-output
windows. Validation and test windows may use already-observed history from the
preceding split, but every target remains inside its assigned split. Per-sensor
normalization is fitted only on finite, nonzero training observations and then
reused unchanged. The fixed input policy for every learned model is
`preserve_recorded_zeros_with_masks`: recorded zero speeds are standardized as
recorded, accompanied by explicit zero/validity masks, and never imputed. The proposed
primary evaluation mask excludes exact-zero targets because their meaning is
uncertain, while retaining a configurable finite-value mask for sensitivity
analysis.

Run the focused real-data verification with:

```powershell
python scripts/smoke_preprocessing.py
```

## Local setup

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The verified local environment uses Python 3.12.14 and CPU-only PyTorch. Place the required METR-LA files as described in [data/README.md](data/README.md) before running the notebook or tests.

## Run the EDA notebook

In VS Code, open `notebooks/01_metr_la_eda.ipynb`, select `.venv\Scripts\python.exe` as the kernel, and choose **Run All**.

If needed, register the environment as a named Jupyter kernel:

```powershell
python -m ipykernel install --user --name urbanflow-ai --display-name "Python (UrbanFlow AI)"
```

After registration, the notebook can also be executed from PowerShell:

```powershell
python -m nbconvert --to notebook --execute --inplace notebooks/01_metr_la_eda.ipynb --ExecutePreprocessor.kernel_name=urbanflow-ai --ExecutePreprocessor.timeout=300
```

## Testing

With `.venv` activated:

```powershell
python scripts/check_environment.py
python -m pytest -q
```

The current tests check METR-LA dimensions, timestamp consistency, and exact graph sensor alignment.
They also verify chronological split boundaries, leakage-safe normalization,
zero/validity masks, 12-input/12-output windows, forecast horizons, and
time-of-day feature alignment.
Extreme-case tests cover all-zero input windows, sensors with no valid training
readings, and validation/test input context crossing a split boundary while
targets remain wholly inside their assigned split. Task 04 tests cover
persistence, training-only historical averages, fallbacks, and masked MAE/RMSE.
Task 05 tests cover pooled sensor-local features, exact horizon/time alignment,
target filtering, categorical identity, deterministic training, JSON model
reload, split boundaries, and Task 04 artifact-alignment guards.

## Task 04 baselines

Run the deterministic baselines from the repository root after placing the
verified METR-LA CSV at the documented path:

```powershell
.\.venv\Scripts\python.exe scripts\run_baselines.py
```

Persistence repeats the last value in each 60-minute input window. Historical
Average uses only training rows to calculate a per-sensor mean for each
five-minute slot of the week, with a training-only per-sensor fallback for a
slot without finite observations. Exact zeros remain recorded values when the
Historical Average is fitted.

The script evaluates the locked chronological test windows at forecast steps
3, 6, and 12 (+15, +30, and +60 minutes). Primary masked MAE/RMSE excludes
finite exact-zero targets because Task 02 found their meaning uncertain. A
second sensitivity policy includes all finite targets, including zeros. Both
policies use the same saved targets and forecasts, and metrics are in original
speed units (mph).

Small metrics and provenance are written to
`results/task04_baselines/metrics.json`. The larger compressed prediction file
is written to ignored `artifacts/task04_baselines/test_predictions_horizons.npz`;
the metrics document records its SHA-256 hash, dataset hash, split boundaries,
package versions, model definitions, and mask policies.

The Task 04 notebook was manually executed from top to bottom on 9 November
2026. All six code cells completed in order without errors, its saved assertions
passed, and it reproduced 6,844 test windows across 207 sensors at the three
required horizons. The generated prediction and dataset SHA-256 hashes match
the metrics manifest. At Task 04 completion, the automated suite passed 19 tests.

## Task 05 XGBoost

Task 05 is complete. The model uses only each sensor's preceding
12 standardized speeds, zero and finite indicators, target-time sine/cosine
features, and stable categorical sensor identity. It does not use neighbouring
sensor speeds, adjacency, graph distances, or future observed traffic. The
verified METR-LA file has no non-finite observations, so its finite-input
indicators are constant; they remain present for preprocessing-contract
consistency. Primary supervised rows exclude exact-zero targets, while final
evaluation also includes an all-finite, zero-inclusive sensitivity policy.

The deterministic validation search selected the shared `depth8_eta005`
configuration. Its selected boosting rounds were 787, 993, and 994 for the
+15-, +30-, and +60-minute models, respectively. The complete validation
selection took 4,107.98 seconds (approximately 1 hour 8 minutes) using four CPU
threads. Selection used validation MAE only; test metrics were calculated after
the shared configuration was frozen and exact Task 04 artifact alignment was
verified.

Primary held-out results in original speed units are:

| Model | Metric | +15 min | +30 min | +60 min |
| --- | --- | ---: | ---: | ---: |
| Persistence | MAE (mph) | 4.0175 | 5.0952 | 6.7980 |
| Persistence | RMSE (mph) | 8.6955 | 11.1353 | 14.2136 |
| Historical Average | MAE (mph) | 7.1104 | 7.1104 | 7.1108 |
| Historical Average | RMSE (mph) | 9.8311 | 9.8311 | 9.8313 |
| XGBoost | MAE (mph) | **3.0044** | **3.5235** | **4.0866** |
| XGBoost | RMSE (mph) | **5.4669** | **6.4889** | **7.4888** |

XGBoost performs best at all three required horizons under the primary policy.
This is predictive evidence for a strong sensor-local conventional model, not
evidence about graph structure or causality. Under the zero-inclusive
sensitivity policy, XGBoost MAE rises to 9.5093, 9.9989, and 10.5082 mph. This
large deterioration reinforces that METR-LA's unresolved synchronized zero
periods materially affect conclusions and must remain explicitly qualified.

Run the small deterministic engineering check with:

```powershell
.\.venv\Scripts\python.exe scripts\run_xgboost.py --smoke --nthread 1
```

Reproduce the full experiment with:

```powershell
.\.venv\Scripts\python.exe scripts\run_xgboost.py --full --nthread 4 --batch-samples 512
```

Keep that terminal open while training. The script reports four major phases,
matrix-quantization batches, validation fit count and percentage, boosting-round
progress, elapsed time, and rolling ETA. Early stopping can finish a fit before
its configured 1,000-round maximum, so round percentages and round ETAs are
explicit upper-bound indicators rather than promises of the final duration.
Use **Ctrl+C** for a controlled interruption. Closing the terminal may terminate
Python immediately; an interrupted run does not produce valid Task 05 results.
Running the same full command again removes only incomplete Task 05 selection
temporaries and restarts the experiment cleanly.

The full command performs predetermined validation-only selection, freezes one
shared configuration, verifies exact Task 04 artifact alignment, and only then
evaluates the held-out test set. Large models and predictions are written under
the Git-ignored `artifacts/task05_xgboost/`; small summaries are written under
`results/task05_xgboost/`.

The presentation notebook only loads completed artifacts and never starts
training. It was manually executed and saved on 11 November 2026. All seven
code cells completed sequentially without errors, its comparison chart and
feature-importance table were produced, and its final artifact, alignment,
horizon, policy, and model-file assertions passed.

## Current progress

| Task | Status |
| --- | --- |
| Task 01 - Environment setup | **Completed** |
| Task 02 - METR-LA inspection and EDA | **Completed** |
| Task 03 - Leakage-safe preprocessing | **Completed** |
| Task 04 - Simple baselines | **Completed** |
| Task 05 - XGBoost | **Completed** |

Tasks 01 through 05 are complete. Task 04 includes deterministic baseline
inference, shared masked metrics, reproducible prediction/metric artifacts, five
focused baseline tests, and a successfully executed six-code-cell notebook.
Task 05 includes validation-only XGBoost selection, three horizon-specific
models, aligned held-out predictions and metrics, an executed seven-code-cell
presentation notebook, and ten focused tests. Temporal-model, graph-model, and
demonstration-application work remain planned. The current automated suite
passes 29 tests.
