"""Validate output/TAWOS.parquet against the statistics recorded in the original extraction."""

import sys
from pathlib import Path

import pandas as pd

PATH = Path(sys.argv[1] if len(sys.argv) > 1 else "output/TAWOS.parquet")

EXPECTED = {
    "rows": 458_232,
    "columns": 19,
    "unique_issues": 458_232,
    "projects": 39,
    "labelled": 65_427,
    "unlabelled": 392_805,
}

df = pd.read_parquet(PATH)

actual = {
    "rows": len(df),
    "columns": df.shape[1],
    "unique_issues": df["Issue_ID"].nunique(),
    "projects": df["Project_Name"].nunique(),
    "labelled": int(df["Story_Point"].notna().sum()),
    "unlabelled": int(df["Story_Point"].isna().sum()),
}

ok = True
for key, expected in EXPECTED.items():
    status = "OK  " if actual[key] == expected else "FAIL"
    ok &= actual[key] == expected
    print(f"[{status}] {key:<14} expected={expected:>9,}  actual={actual[key]:>9,}")

print("\nColumn dtypes:")
print(df.dtypes.to_string())

sys.exit(0 if ok else 1)
