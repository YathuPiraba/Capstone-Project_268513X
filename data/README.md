# Dataset setup and provenance

This directory contains local data used by UrbanFlow AI. METR-LA is the required primary dataset. PEMS-BAY is optional and is not a dependency of the completed environment or EDA tasks.

## Expected local structure

Place the METR-LA files at these exact repository-relative paths:

```text
data/
|-- README.md
`-- raw/
    |-- metr-la/
    |   |-- METR-LA.csv
    |   |-- adj_mx_METR-LA.pkl
    |   `-- graph_sensor_ids.txt
    `-- pems-bay/                  # Optional future validation only
```

Reusable code uses repository-relative paths, so the repository can be moved without editing machine-specific absolute paths.

## Authoritative sources

- **METR-LA speed data and adjacency:** [Zenodo record 5724362](https://zenodo.org/records/5724362)
- **DCRNN sensor graph metadata:** [DCRNN `data/sensor_graph`](https://github.com/liyaguang/DCRNN/tree/master/data/sensor_graph)

Download or copy the following files:

| Local file | Source | Purpose |
| --- | --- | --- |
| `data/raw/metr-la/METR-LA.csv` | Zenodo | Five-minute traffic-speed measurements for 207 sensors |
| `data/raw/metr-la/adj_mx_METR-LA.pkl` | Zenodo | Weighted 207 × 207 adjacency matrix and its sensor metadata |
| `data/raw/metr-la/graph_sensor_ids.txt` | DCRNN sensor-graph directory | Independent reference ordering for the 207 graph sensor IDs |

The large `METR-LA.csv` file is intentionally excluded from Git and must be downloaded locally. The smaller `adj_mx_METR-LA.pkl` and `graph_sensor_ids.txt` files are included in the repository so the graph definition and sensor ordering can be reproduced. Do not add the optional PEMS-BAY files to the core workflow.

## Verified SHA-256 hashes

Task 02 calculated the following hashes from the local files that were inspected:

| File | SHA-256 |
| --- | --- |
| `METR-LA.csv` | `8D67A35472DB1719D7D4BE851F2BF64CB21D9C52577C8A6B4B873D43205AF381` |
| `adj_mx_METR-LA.pkl` | `A35687C6E15AA228DC45027B0ED2A0EA0F4EC78F573DEB992C595092D12F61B3` |
| `graph_sensor_ids.txt` | `3BA026CAA2E6263AB0EA54B0FA1B125DBFA7216544CD05313B555E826292B990` |

These hashes identify the exact local copies used for Task 02. A mismatch means the file differs and should be investigated before it is loaded or used in an experiment.

## Verify the files in PowerShell

From the repository root, verify all three files:

```powershell
Get-FileHash -Algorithm SHA256 `
  data/raw/metr-la/METR-LA.csv, `
  data/raw/metr-la/adj_mx_METR-LA.pkl, `
  data/raw/metr-la/graph_sensor_ids.txt
```

To check one file individually:

```powershell
Get-FileHash -Algorithm SHA256 data/raw/metr-la/METR-LA.csv
```

Compare the `Hash` values printed by PowerShell with the table above. PowerShell displays hexadecimal hashes using uppercase letters, but letter case does not change the hash value.

## Pickle security

Python pickle files can execute code when they are loaded. Only load `adj_mx_METR-LA.pkl` when it comes from the documented Zenodo record or another source that has been explicitly reviewed and trusted. Do not load arbitrary or user-supplied pickle files. Verify the hash before loading when file provenance is uncertain.

## Verified sensor alignment

Task 02 checked the 207 sensor identifiers in three places:

1. the `METR-LA.csv` column order;
2. the sensor-ID list and ID-to-index mapping stored in `adj_mx_METR-LA.pkl`; and
3. the ordering in `graph_sensor_ids.txt`.

For the verified files, all three sources contain the same sensor IDs in exactly the same order. The adjacency shape is also `207 × 207`, so no reorder mapping is required for these copies.

This alignment must still be asserted whenever data and graph artifacts are loaded. If the order changes silently, traffic measurements could be assigned to the wrong graph nodes even though the array dimensions still appear valid.

## Optional PEMS-BAY data

PEMS-BAY may be considered later for optional external validation if the required METR-LA experiments are complete and the project owner approves the additional work. It is not currently processed, tested, trained on, or required to run the Task 02 notebook and tests.
