# UrbanFlow AI - Spatio-Temporal Graph Learning for Urban Traffic Forecasting and Analysis

UrbanFlow AI is an MSc Data Science and Artificial Intelligence capstone project investigating short-term traffic-speed forecasting on the METR-LA road-sensor network.

## Research question

> To what extent does incorporating spatial road-network relationships improve multi-horizon traffic-speed forecasting compared with conventional and temporal-only forecasting approaches?

The study does not assume that a graph model will perform best. Negative or mixed results are valid if fair held-out experiments show that spatial information provides limited or conditional benefit.

## Dataset and forecasting setup

METR-LA contains traffic speeds from **207 sensors** across **34,272 observations** sampled every five minutes. Task 02 verified these dimensions, the timestamp cadence, and sensor ordering across the speed data and graph metadata.

Later experiments will use 12 historical steps (60 minutes) to predict 12 future steps (60 minutes), with evaluation at **+15, +30, and +60 minutes** (output steps 3, 6, and 12). Dataset sources, paths, hashes, and safety notes are in [data/README.md](data/README.md).

## Planned models

The planned comparison is **Persistence**, **Historical Average**, **XGBoost**, one **LSTM or GRU**, and one **spatio-temporal graph neural network (ST-GNN)**.

These forecasting models have not yet been implemented.

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
|   `-- 02_metr_la_preprocessing.ipynb # Completed Task 03 notebook
|-- scripts/check_environment.py
|-- scripts/smoke_preprocessing.py  # Real-data Task 03 verification
|-- src/data/                       # Inspection, preprocessing, and windows
`-- tests/                          # Data integrity and preprocessing tests
```

Forecasting models, evaluation, backend, and frontend components remain planned work.

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
targets remain wholly inside their assigned split. Masked MAE/RMSE integration
tests will be added with the shared metric implementation in Task 04.

## Current progress

| Task | Status |
| --- | --- |
| Task 01 - Environment setup | **Completed** |
| Task 02 - METR-LA inspection and EDA | **Completed** |
| Task 03 - Leakage-safe preprocessing | **Completed** |

Tasks 01, 02, and 03 are complete. Task 03 code, 14 automated tests, the real-data smoke test, and the manually executed educational notebook all passed. Model training, evaluation, and the demonstration application remain planned work.
