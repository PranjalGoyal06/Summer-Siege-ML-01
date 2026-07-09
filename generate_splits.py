from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "CB513.csv"
SPLITS_PATH = ROOT / "splits.json"


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    total_proteins = len(df)
    if total_proteins != 511:
        raise ValueError(f"Expected 511 proteins in {CSV_PATH}, found {total_proteins}")

    train_ids, test_ids = train_test_split(range(total_proteins), test_size=0.2, random_state=42)
    train_ids, val_ids = train_test_split(train_ids, test_size=0.1875, random_state=42)

    splits = {
        "source_csv": str(CSV_PATH.relative_to(ROOT)),
        "seed": 42,
        "test_size": 0.2,
        "val_size_within_train": 0.1875,
        "total_proteins": total_proteins,
        "train_ids": list(train_ids),
        "val_ids": list(val_ids),
        "test_ids": list(test_ids),
    }

    with open(SPLITS_PATH, "w") as f:
        json.dump(splits, f, indent=2)
        f.write("\n")

    print(f"Wrote {SPLITS_PATH}")


if __name__ == "__main__":
    main()