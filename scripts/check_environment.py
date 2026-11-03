"""Verify the UrbanFlow AI local Python environment."""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
import sklearn
import torch
import xgboost as xgb


def package_version(distribution: str) -> str:
    """Return an installed distribution version without failing the whole check."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    """Print versions and run small NumPy and PyTorch smoke tests."""
    print("UrbanFlow AI environment check")
    print(f"Python: {platform.python_version()} ({sys.executable})")

    packages = {
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "XGBoost": xgb.__version__,
        "PyTorch": torch.__version__,
        "SciPy": package_version("scipy"),
        "h5py": package_version("h5py"),
        "Matplotlib": package_version("matplotlib"),
        "Seaborn": package_version("seaborn"),
        "PyYAML": package_version("PyYAML"),
        "JupyterLab": package_version("jupyterlab"),
        "ipykernel": package_version("ipykernel"),
        "pytest": package_version("pytest"),
    }
    for name, installed_version in packages.items():
        print(f"{name}: {installed_version}")

    numpy_result = np.array([1, 2, 3], dtype=np.float64).mean()
    assert numpy_result == 2.0
    print(f"NumPy smoke test: PASS (mean={numpy_result})")

    torch_result = torch.tensor([1.0, 2.0, 3.0]).mean().item()
    assert torch_result == 2.0
    print(f"PyTorch smoke test: PASS (mean={torch_result})")
    print(f"CUDA available: {torch.cuda.is_available()} (informational only)")
    print("Environment check: PASS")


if __name__ == "__main__":
    main()

