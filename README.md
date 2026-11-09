# UrbanFlow AI - Spatio-Temporal Graph Learning for Urban Traffic Forecasting and Analysis

UrbanFlow AI is an MSc Data Science and Artificial Intelligence capstone project investigating short-term traffic-speed forecasting on the METR-LA road-sensor network.

## Research question

> To what extent does incorporating spatial road-network relationships improve multi-horizon traffic-speed forecasting compared with conventional and temporal-only forecasting approaches?

The study does not assume that a graph model will perform best. Negative or mixed results are valid if fair held-out experiments show that spatial information provides limited or conditional benefit.

## Dataset and forecasting setup

METR-LA contains traffic speeds from **207 sensors** across **34,272 observations** sampled every five minutes. Task 02 verified these dimensions, the timestamp cadence, and sensor ordering across the speed data and graph metadata.

Later experiments will use 12 historical steps (60 minutes) to predict 12 future steps (60 minutes), with evaluation at **+15, +30, and +60 minutes** (output steps 3, 6, and 12). Dataset sources, paths, hashes, and safety notes are in [data/README.md](data/README.md).

## Planned models

The planned comparison is **Persistence**, **Historical Average**, **XGBoost**, one **LSTM or GRU**, and one **spatio-temporal graph neural network (ST-GNN)**. Task 04 implements the two deterministic baselines; learned forecasting models remain future work.

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
|   `-- 03_metr_la_baselines.ipynb  # Completed Task 04 notebook
|-- scripts/check_environment.py
|-- scripts/smoke_preprocessing.py  # Real-data Task 03 verification
|-- src/data/                       # Inspection, preprocessing, and windows
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
the metrics manifest. The full automated suite also passes with 19 tests.

## Current progress

| Task | Status |
| --- | --- |
| Task 01 - Environment setup | **Completed** |
| Task 02 - METR-LA inspection and EDA | **Completed** |
| Task 03 - Leakage-safe preprocessing | **Completed** |
| Task 04 - Simple baselines | **Completed** |

Tasks 01 through 04 are complete. Task 04 includes deterministic baseline
inference, shared masked metrics, reproducible prediction/metric artifacts, five
focused baseline tests, and a successfully executed six-code-cell notebook. The
complete automated suite passes with 19 tests. Learned-model training and the
demonstration application remain planned work.
