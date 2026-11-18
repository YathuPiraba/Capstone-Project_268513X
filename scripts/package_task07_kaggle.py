"""Build and verify the minimal Task 07 Kaggle upload archive."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STAGE_PARENT = ROOT / ".tmp" / "task07_kaggle_upload"
STAGE_ROOT = STAGE_PARENT / ROOT.name
ARCHIVE = ROOT / ".tmp" / "task07_kaggle_upload.zip"

PACKAGE_FILES = (
    "data/raw/metr-la/METR-LA.csv",
    "data/raw/metr-la/adj_mx_METR-LA.pkl",
    "data/raw/metr-la/graph_sensor_ids.txt",
    "artifacts/task04_baselines/test_predictions_horizons.npz",
    "artifacts/task05_xgboost/test_predictions_horizons.npz",
    "artifacts/task06_gru/test_predictions_horizons.npz",
    "scripts/run_dcrnn.py",
    "src/data/__init__.py",
    "src/data/inspection.py",
    "src/data/preprocessing.py",
    "src/data/windows.py",
    "src/evaluation/__init__.py",
    "src/evaluation/metrics.py",
    "src/models/__init__.py",
    "src/models/baselines.py",
    "src/models/dcrnn.py",
    "src/training/__init__.py",
    "src/training/dcrnn.py",
    "requirements.txt",
    "README.md",
    "results/task07_dcrnn/graph_manifest.json",
)

EXPECTED_DATA_HASHES = {
    "data/raw/metr-la/METR-LA.csv": "8D67A35472DB1719D7D4BE851F2BF64CB21D9C52577C8A6B4B873D43205AF381",
    "data/raw/metr-la/adj_mx_METR-LA.pkl": "A35687C6E15AA228DC45027B0ED2A0EA0F4EC78F573DEB992C595092D12F61B3",
    "data/raw/metr-la/graph_sensor_ids.txt": "3BA026CAA2E6263AB0EA54B0FA1B125DBFA7216544CD05313B555E826292B990",
    "artifacts/task04_baselines/test_predictions_horizons.npz": "48FD54D29C5BDB130F978F2526865649BF247CE8D6C371C8807EE8DC4D5B4321",
    "artifacts/task05_xgboost/test_predictions_horizons.npz": "19AF5FC9C78053D8B7F61DDB7B5D9A7122765328B1403004E3E1E5429037D236",
    "artifacts/task06_gru/test_predictions_horizons.npz": "27C165BDE7FD72E1CD12D21F4D0D1087F9FA1CBCB08E90EE800461C39EC5FCAF",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    resolved_root = ROOT.resolve()
    resolved_stage = STAGE_PARENT.resolve()
    allowed_parent = (ROOT / ".tmp").resolve()
    if resolved_stage.parent != allowed_parent:
        raise RuntimeError("Package staging directory must remain inside the repository .tmp directory")
    if STAGE_PARENT.exists():
        shutil.rmtree(STAGE_PARENT)
    STAGE_ROOT.mkdir(parents=True)

    entries = []
    for relative in PACKAGE_FILES:
        source = (ROOT / relative).resolve()
        if resolved_root not in source.parents:
            raise RuntimeError(f"Package source escaped the repository: {relative}")
        if not source.is_file():
            raise FileNotFoundError(f"Required Task 07 package file is missing: {relative}")
        actual_hash = sha256(source)
        expected_hash = EXPECTED_DATA_HASHES.get(relative)
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"Required data/reference hash mismatch: {relative}")
        destination = STAGE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        entries.append({"path": relative, "bytes": source.stat().st_size, "sha256": actual_hash})

    manifest = {
        "format_version": 1,
        "package_root": ROOT.name,
        "purpose": "Task 07 Kaggle GPU probe, later validation training, and separately approved finalization",
        "file_count": len(entries),
        "files": entries,
    }
    manifest_path = STAGE_ROOT / "kaggle_package_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if ARCHIVE.exists():
        ARCHIVE.unlink()
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(STAGE_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGE_PARENT).as_posix())

    with zipfile.ZipFile(ARCHIVE) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"Archive CRC verification failed: {bad_member}")
        archived = set(archive.namelist())
        expected = {f"{ROOT.name}/{entry['path']}" for entry in entries}
        expected.add(f"{ROOT.name}/kaggle_package_manifest.json")
        if archived != expected:
            raise RuntimeError("Archive members do not exactly match the package manifest")
    print(json.dumps({
        "archive": ARCHIVE.relative_to(ROOT).as_posix(),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": sha256(ARCHIVE),
        "file_count": len(entries) + 1,
        "staged_root": STAGE_ROOT.relative_to(ROOT).as_posix(),
    }, indent=2))


if __name__ == "__main__":
    main()
