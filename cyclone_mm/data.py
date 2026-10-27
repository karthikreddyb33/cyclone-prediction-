"""Sample table, splits, preprocessing and the PyTorch Dataset."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as T

from . import config as C

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def image_path(storm_id: str, ts: pd.Timestamp) -> str:
    return str(C.IMAGES / f"{storm_id}_{ts:%Y%m%d%H}.npy")


def load_samples(require_images: bool = True) -> pd.DataFrame:
    """Synoptic best-track fixes joined with SST and (optionally) image crops.

    `require_images=True` keeps only fixes that have a crop, so the tabular
    baseline and the multimodal model are scored on exactly the same rows.
    """
    df = pd.read_csv(C.TRACK_CSV, parse_dates=["datetime"])
    df = df[df["datetime"].dt.hour.isin(C.SYNOPTIC_HOURS)]
    sst_csv = C.PROCESSED / "sst.csv"
    if sst_csv.exists():
        sst = pd.read_csv(sst_csv, parse_dates=["datetime"])
        df = df.merge(sst, on=["storm_id", "datetime"], how="left")
    else:
        df["sst"] = np.nan
    df["image_path"] = [image_path(s, t) for s, t in zip(df.storm_id, df.datetime)]
    if require_images:
        df = df[df["image_path"].map(lambda p: Path(p).exists())]
    return df.reset_index(drop=True)


def split(df: pd.DataFrame, mode: str = "storm", val_frac: float = 0.15, seed: int = 42):
    """Train/val from <= VAL_YEAR, test = TEST_YEAR.

    mode="storm":    85/15 random split grouped by storm (no storm in both).
    mode="temporal": train <= TRAIN_LAST_YEAR, validate on VAL_YEAR.
    """
    test = df[df["year"] == C.TEST_YEAR]
    dev = df[df["year"] <= C.VAL_YEAR]
    if mode == "temporal":
        train, val = dev[dev["year"] <= C.TRAIN_LAST_YEAR], dev[dev["year"] == C.VAL_YEAR]
    elif mode == "storm":
        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr, va = next(gss.split(dev, groups=dev["storm_id"]))
        train, val = dev.iloc[tr], dev.iloc[va]
    else:
        raise ValueError(mode)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


@dataclass
class TabularPreprocessor:
    """Median imputation + StandardScaler for features; z-scoring for the target.
    Fitted on the training split only."""
    features: list
    medians: dict = None
    scaler: StandardScaler = None
    y_mean: float = 0.0
    y_std: float = 1.0

    def fit(self, df: pd.DataFrame):
        self.medians = df[self.features].median().fillna(0.0).to_dict()
        self.scaler = StandardScaler().fit(df[self.features].fillna(self.medians))
        self.y_mean, self.y_std = float(df[C.TARGET].mean()), float(df[C.TARGET].std())
        return self

    def transform_x(self, df: pd.DataFrame) -> np.ndarray:
        return self.scaler.transform(df[self.features].fillna(self.medians)).astype(np.float32)

    def transform_y(self, y) -> np.ndarray:
        return ((np.asarray(y, dtype=np.float32) - self.y_mean) / self.y_std).astype(np.float32)

    def inverse_y(self, z) -> np.ndarray:
        return np.asarray(z) * self.y_std + self.y_mean


def ir_to_tensor(bt: np.ndarray) -> torch.Tensor:
    """Brightness temperature (K) -> 3x H x W float tensor in [0, 1].
    Inverted so cold cloud tops (deep convection) are bright, like IR enhancement curves."""
    x = (C.BT_MAX - bt.astype(np.float32)) / (C.BT_MAX - C.BT_MIN)
    x = np.nan_to_num(np.clip(x, 0.0, 1.0), nan=0.0)
    return torch.from_numpy(x).unsqueeze(0).expand(3, -1, -1).contiguous()


def build_image_transforms(train: bool, flips: bool = True):
    """Train: random rotation (+ optional flips) and resize to 224.
    Note: a single horizontal *or* vertical flip mirrors the storm's spin
    (counter-clockwise -> clockwise in the northern hemisphere). Pass
    flips=False to keep only physically consistent rotations."""
    ops = []
    if train:
        ops.append(T.RandomRotation(degrees=180, interpolation=T.InterpolationMode.BILINEAR))
        if flips:
            ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip()]
    ops += [T.Resize((C.IMG_SIZE, C.IMG_SIZE), antialias=True),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return T.Compose(ops)


class CycloneDataset(Dataset):
    """Pairs each timestamped best-track record with its IR crop.

    Returns dict(image [3,224,224] | absent, tabular [F], target [1], idx)."""

    def __init__(self, df: pd.DataFrame, prep: TabularPreprocessor, with_images: bool,
                 train: bool = False, flips: bool = True):
        self.df = df.reset_index(drop=True)
        self.x = prep.transform_x(self.df)
        self.y = prep.transform_y(self.df[C.TARGET])
        self.with_images = with_images
        self.tf = build_image_transforms(train, flips) if with_images else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        item = {"tabular": torch.from_numpy(self.x[i]),
                "target": torch.tensor([self.y[i]]),
                "idx": i}
        if self.with_images:
            item["image"] = self.tf(ir_to_tensor(np.load(self.df.at[i, "image_path"])))
        return item


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=workers > 0, pin_memory=torch.cuda.is_available())
