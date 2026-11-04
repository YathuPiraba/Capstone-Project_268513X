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
|   `-- 01_metr_la_eda.ipynb        # Completed Task 02 notebook
|-- scripts/check_environment.py
|-- src/data/inspection.py          # Dataset inspection helpers
`-- tests/test_data_inspection.py
```

Preprocessing, model, evaluation, backend, and frontend components are planned work and are not currently implemented.

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

## Current progress

| Task | Status |
| --- | --- |
| Task 01 - Environment setup | **Completed** |
| Task 02 - METR-LA inspection and EDA | **Completed** |
| Task 03 - Leakage-safe preprocessing | **Not started** |

Only Tasks 01 and 02 are complete. Chronological splitting, preprocessing, forecasting windows, model training, evaluation, and the demonstration application remain planned work.
