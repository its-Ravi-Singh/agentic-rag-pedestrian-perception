import pandas as pd
from apr.config import ROOT

SPLIT_DIR = ROOT / "JAAD" / "split_ids"
STEP1_CSV = ROOT / "Generated_Data" / "jaad_annotations_extracted_data_step1.csv"


def load_splits(subset="default"):
    """Map each video id ('0001') to 'train', 'val' or 'test' using JAAD's official split file"""
    out = {}
    for split in ("train", "val", "test"):
        for line in (SPLIT_DIR / subset / f"{split}.txt").read_text().split():
            out[line.replace("video_", "")] = split
    return out


def load_annotations(csv_path=STEP1_CSV, subset="default"):
    """Step 1 annotations with a 'split' column. Videos outside the official split get NaN."""
    df = pd.read_csv(csv_path, dtype={"video_id": str})
    df["video_id"] = df["video_id"].str.zfill(4)
    df["split"] = df["video_id"].map(load_splits(subset))
    return df
