"""PyTorch Datasets for FedSafe-Fuse: MedMNIST paired arrays and BraTS per-case slices.

Designed to read directly from Google Drive (`/content/drive/MyDrive/FedSafeFuse/`)
in Colab, or from a local copy of that tree.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


class MedMNISTPaired(Dataset):
    """Pre-computed (MRI, PET, label) triples from a single .npz on Drive.

    Optional `indices` restricts to a subset of the global array (used for per-client
    partitions and calibration hold-outs).
    """

    def __init__(self, npz_path: str, indices: Iterable[int] | None = None):
        with np.load(npz_path) as d:
            self.mri = d["mri"]  # (N, 128, 128) float32 in [0, 1]
            self.pet = d["pet"]
            self.labels = d["labels"].astype(np.int64).flatten()
        self.indices = list(indices) if indices is not None else list(range(len(self.mri)))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        gi = self.indices[i]
        mri = torch.from_numpy(self.mri[gi]).unsqueeze(0)  # (1, 128, 128)
        pet = torch.from_numpy(self.pet[gi]).unsqueeze(0)
        return mri, pet, int(self.labels[gi])


class BraTSPaired(Dataset):
    """Lazily loads BraTS per-case .npz files, indexed by (case_id, slice_idx).

    A small LRU cache keeps the most recently used cases in memory to avoid
    repeated Drive reads when iterating sequentially.
    """

    def __init__(self, brats_out_dir: str, slice_list, cache_size: int = 8):
        self.brats_out_dir = brats_out_dir
        self.slice_list = list(slice_list)  # list of [case_id, slice_idx]
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.slice_list)

    def _load(self, case_id: str):
        if case_id in self._cache:
            self._cache.move_to_end(case_id)
            return self._cache[case_id]
        path = os.path.join(self.brats_out_dir, f"{case_id}.npz")
        d = np.load(path)
        t1, t2 = d["t1"], d["t2"]
        d.close()
        self._cache[case_id] = (t1, t2)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return t1, t2

    def __getitem__(self, i: int):
        case_id, sidx = self.slice_list[i]
        t1, t2 = self._load(case_id)
        return (
            torch.from_numpy(t1[sidx]).unsqueeze(0),
            torch.from_numpy(t2[sidx]).unsqueeze(0),
            case_id,
        )


def load_partition(json_path: str) -> dict:
    """Read a non-IID partition manifest JSON."""
    with open(json_path) as f:
        return json.load(f)
