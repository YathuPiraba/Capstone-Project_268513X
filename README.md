# UrbanFlow AI

UrbanFlow AI investigates whether spatial road-network relationships improve
multi-horizon traffic-speed forecasting on METR-LA. Local development uses
ordinary Python modules and scripts so reusable code can also run on Kaggle or
Google Colab without Kaggle-specific notebook dependencies.

## Prerequisites

- Windows 10 or 11 with PowerShell
- Git
- Python 3.11 preferred; Python 3.12 is supported and was used for the verified
  local environment
- VS Code with the Microsoft **Python** (`ms-python.python`) and **Jupyter**
  (`ms-toolsai.jupyter`) extensions

Install the extensions from VS Code's Extensions view, or run:

```powershell
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
```

## Local environment setup (PowerShell)

From the repository root, create and activate the virtual environment:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.11 is unavailable, use a compatible Python 3.12 interpreter, for
example `py -3.12 -m venv .venv`. The checked local environment uses Python
3.12.14 and CPU-only PyTorch. The PyTorch CPU wheel index is recorded in
`requirements.txt`; no CUDA packages are required for local smoke testing.

## VS Code and Jupyter

1. Open the repository folder in VS Code.
2. Run **Python: Select Interpreter** from the Command Palette and choose
   `.venv\Scripts\python.exe`.
3. Open a notebook, choose **Select Kernel**, then select **Python Environments**
   and the `.venv` interpreter (displayed as `Python (UrbanFlow AI)` when the
   optional local kernel registration is present).
4. If the kernel is not listed, activate `.venv` and run:

   ```powershell
   python -m ipykernel install --user --name urbanflow-ai --display-name "Python (UrbanFlow AI)"
   ```

Notebook checkpoint directories are ignored, while `.ipynb` source files remain
trackable. Clear large or unnecessary cell outputs before committing notebooks.

## Verify the environment

With `.venv` activated:

```powershell
python scripts/check_environment.py
python -m pytest
```

The environment script checks package imports and versions, runs small NumPy and
PyTorch operations, and reports CUDA availability for information only. Pytest
may report that no tests were collected until project tests are added.

## Local development and Kaggle responsibilities

Use local Windows development for notebooks, preprocessing, baseline models,
unit tests, FastAPI and Vue development, and Git work. Use Kaggle GPU sessions
for computationally expensive LSTM/GRU and ST-GNN training. GitHub remains the
source of truth for source code, configuration, experiment metadata, and
documentation; datasets and model checkpoints stay outside Git.

PyTorch Geometric and PyTorch Geometric Temporal are intentionally not installed
yet. Before Kaggle ST-GNN training, select versions that match Kaggle's Python,
PyTorch, CUDA, and platform wheel availability. Keep graph-learning dependencies
in a separate Kaggle-specific specification once that environment is confirmed.
