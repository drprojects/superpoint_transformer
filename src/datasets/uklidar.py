"""
uklidar.py
==========
Superpoint Transformer dataset class for UK colourised LiDAR point
clouds (EA National LIDAR Programme, Bluesky, MLS, etc.).

This module provides the reader and dataset class needed to run
SPT inference on UK LiDAR data using DALES or KITTI-360 pretrained
checkpoints.

Place this file in:
    superpoint_transformer/src/datasets/uklidar.py

Then create the corresponding Hydra configs (see uklidar.yaml).

The reader function ``read_uklidar_tile()`` parses LAS/LAZ files via
PDAL and returns a ``torch_geometric.data.Data`` object with:

    pos       — (N, 3) float32: X, Y, Z coordinates
    rgb       — (N, 3) float32: R, G, B normalised to [0, 1]
    intensity — (N, 1) float32: normalised to [0, 1]
    y         — (N,) int64: semantic labels remapped to target space

For inference on unseen data, all labels are set to the void class
(``num_classes``), which SPT excludes from metrics computation.

Usage
-----
Place colourised LAS/LAZ files into::

    data/uklidar/raw/test/*.laz

Then run::

    python src/eval.py experiment=semantic/uklidar \\
        ckpt_path=/path/to/dales_checkpoint.ckpt

Author: James (Ordnance Survey)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch_geometric.data import Data as PyGData

# SPT's extended Data class with .show() visualisation
try:
    from src.data import Data
except ImportError:
    # Fallback if not running inside the SPT repo
    Data = PyGData

try:
    import pdal
    HAS_PDAL = True
except Exception as _pdal_err:
    HAS_PDAL = False
    # Store the actual error for diagnostic reporting
    _PDAL_IMPORT_ERROR = _pdal_err

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False

from src.datasets.base import BaseDataset

log = logging.getLogger(__name__)


# ===========================================================================
# DALES-compatible class definitions (default target)
# ===========================================================================

UKLIDAR_NUM_CLASSES = 8  # Matches DALES

UKLIDAR_CLASS_NAMES = {
    0: "ground",
    1: "vegetation",
    2: "cars",
    3: "trucks",
    4: "power_lines",
    5: "fences",
    6: "poles",
    7: "buildings",
}

UKLIDAR_CLASS_COLORS = [
    [128, 64, 0],     # ground — brown
    [0, 128, 0],      # vegetation — green
    [255, 0, 0],      # cars — red
    [255, 128, 0],    # trucks — orange
    [255, 255, 0],    # power_lines — yellow
    [128, 128, 128],  # fences — grey
    [0, 0, 255],      # poles — blue
    [255, 0, 255],    # buildings — magenta
]

# ASPRS → DALES label mapping
# Applied when reading classified UK LiDAR data.
# For inference on unclassified data, all points get void label (8).
ASPRS_TO_TARGET = {
    0: 8,   # Created/Never classified → void
    1: 8,   # Unclassified → void
    2: 0,   # Ground → ground
    3: 1,   # Low Vegetation → vegetation
    4: 1,   # Medium Vegetation → vegetation
    5: 1,   # High Vegetation → vegetation
    6: 7,   # Building → buildings
    7: 8,   # Low Point (Noise) → void
    9: 8,   # Water → void
    17: 8,  # Bridge Deck → void
    18: 8,  # High Noise → void
    20: 8,  # Permanent Structure → void
}


# ===========================================================================
# Reader Function
# ===========================================================================

def read_uklidar_tile(
    filepath: str | Path,
    remap: dict[int, int] | None = None,
    void_label: int = 8,
    label_all_void: bool = False,
) -> Data:
    """Read a colourised LAS/LAZ file and return an SPT-compatible
    Data object.

    Tries PDAL first, falls back to laspy if PDAL import fails.
    Both are available in the spt conda environment.

    Args:
        filepath:       Path to the LAS/LAZ file.
        remap:          ASPRS → target label dict.  None = ASPRS_TO_TARGET.
        void_label:     Label code for void/unlabeled points.
        label_all_void: If True, set all labels to void (for inference
                        on data without ground truth).

    Returns:
        Data object with pos, rgb, intensity, y attributes.
    """
    filepath = Path(filepath)
    if remap is None:
        remap = ASPRS_TO_TARGET

    # --- Read point cloud (PDAL or laspy) ---
    if HAS_PDAL:
        points_dict = _read_with_pdal(filepath)
    elif HAS_LASPY:
        log.info("PDAL not available (reason: %s), using laspy fallback",
                 _PDAL_IMPORT_ERROR if not HAS_PDAL else "N/A")
        points_dict = _read_with_laspy(filepath)
    else:
        msg = "Neither PDAL nor laspy is available."
        if not HAS_PDAL:
            msg += "\n  PDAL import error: {}".format(_PDAL_IMPORT_ERROR)
        msg += "\n  Install one via: conda install -c conda-forge python-pdal"
        msg += "\n  Or: pip install laspy[laszip]"
        raise ImportError(msg)

    n = points_dict["n"]
    pos = points_dict["pos"]
    rgb = points_dict["rgb"]
    intensity = points_dict["intensity"]
    raw_labels = points_dict["classification"]

    # --- Labels ---
    if label_all_void or raw_labels is None:
        y = np.full(n, void_label, dtype=np.int64)
    else:
        max_code = max(max(remap.keys()), int(raw_labels.max())) + 1
        lut = np.full(max_code, void_label, dtype=np.int64)
        for src, dst in remap.items():
            if src < max_code:
                lut[src] = dst
        y = lut[np.clip(raw_labels.astype(np.int64), 0, max_code - 1)]

    # --- Build Data ---
    data = Data(
        pos=torch.from_numpy(pos),
        rgb=torch.from_numpy(rgb),
        intensity=torch.from_numpy(intensity),
        y=torch.from_numpy(y),
    )

    return data


def _normalise_rgb(r, g, b):
    """Normalise RGB arrays to [0, 1] float32."""
    max_val = max(r.max(), g.max(), b.max())
    if max_val > 255:
        divisor = 65535.0
    elif max_val > 1:
        divisor = 255.0
    else:
        divisor = 1.0
    return np.column_stack([r, g, b]).astype(np.float32) / divisor


def _normalise_intensity(raw_i):
    """Normalise intensity to [0, 1] float32, shape (N, 1)."""
    raw_i = raw_i.astype(np.float32)
    i_max = raw_i.max()
    if i_max > 0:
        return (raw_i / i_max).reshape(-1, 1)
    return np.zeros((len(raw_i), 1), dtype=np.float32)


def _read_with_pdal(filepath):
    """Read LAS/LAZ via PDAL, return a dict of numpy arrays."""
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [{"type": "readers.las", "filename": str(filepath)}]
    }))
    pipeline.execute()
    points = pipeline.arrays[0]
    dim_names = list(points.dtype.names)
    n = len(points)

    pos = np.column_stack([
        points["X"].astype(np.float32),
        points["Y"].astype(np.float32),
        points["Z"].astype(np.float32),
    ])

    if "Red" in dim_names and "Green" in dim_names and "Blue" in dim_names:
        rgb = _normalise_rgb(points["Red"], points["Green"], points["Blue"])
    else:
        rgb = np.zeros((n, 3), dtype=np.float32)

    if "Intensity" in dim_names:
        intensity = _normalise_intensity(points["Intensity"])
    else:
        intensity = np.zeros((n, 1), dtype=np.float32)

    classification = None
    if "Classification" in dim_names:
        classification = points["Classification"]

    return {"n": n, "pos": pos, "rgb": rgb,
            "intensity": intensity, "classification": classification}


def _read_with_laspy(filepath):
    """Read LAS/LAZ via laspy, return a dict of numpy arrays."""
    las = laspy.read(str(filepath))
    n = len(las.points)

    pos = np.column_stack([
        np.asarray(las.x, dtype=np.float32),
        np.asarray(las.y, dtype=np.float32),
        np.asarray(las.z, dtype=np.float32),
    ])

    # RGB — laspy exposes these as point_format dimensions
    try:
        rgb = _normalise_rgb(
            np.asarray(las.red),
            np.asarray(las.green),
            np.asarray(las.blue),
        )
    except Exception:
        rgb = np.zeros((n, 3), dtype=np.float32)

    # Intensity
    try:
        intensity = _normalise_intensity(np.asarray(las.intensity))
    except Exception:
        intensity = np.zeros((n, 1), dtype=np.float32)

    # Classification
    classification = None
    try:
        classification = np.asarray(las.classification)
    except Exception:
        pass

    return {"n": n, "pos": pos, "rgb": rgb,
            "intensity": intensity, "classification": classification}


# ===========================================================================
# Dataset Class
# ===========================================================================

class UKLidarDataset(BaseDataset):
    """Dataset class for UK colourised LiDAR point clouds.

    Inherits from ``src.datasets.base.BaseDataset`` to integrate with
    SPT's preprocessing, transform, and evaluation pipeline.

    Directory structure::

        data/uklidar/
        ├── raw/
        │   ├── train/           # Training tiles (optional)
        │   │   └── tile_A.laz
        │   ├── val/             # Validation tiles (optional)
        │   │   └── tile_B.laz
        │   └── test/            # Test/inference tiles
        │       └── tile_C.laz
        └── processed/
            └── <hash>/          # Auto-generated by SPT preprocessing
                └── tile_C.h5

    For inference-only usage, place all files in ``raw/test/``.
    """

    _num_classes = UKLIDAR_NUM_CLASSES
    _class_names = list(UKLIDAR_CLASS_NAMES.values())
    _class_colors = UKLIDAR_CLASS_COLORS

    # Whether labels are available for inference data
    label_all_void: bool = True

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def all_base_cloud_ids(self):
        """Discover cloud IDs from the raw directory structure.

        Returns dict with train/val/test splits populated from the
        corresponding subdirectories.
        """
        raw_dir = Path(self.raw_dir)
        splits = {"train": [], "val": [], "test": []}

        for split in splits:
            split_dir = raw_dir / split
            if split_dir.is_dir():
                laz = sorted(split_dir.glob("*.laz"))
                las = sorted(split_dir.glob("*.las"))
                ply = sorted(split_dir.glob("*.ply"))
                all_files = sorted(
                    set(laz + las + ply), key=lambda p: p.name,
                )
                # Cloud IDs include the split prefix to ensure uniqueness
                splits[split] = [
                    f"{split}/{f.stem}" for f in all_files
                ]

        log.info(
            "UKLidar splits: train=%d, val=%d, test=%d",
            len(splits["train"]),
            len(splits["val"]),
            len(splits["test"]),
        )

        return splits

    def read_single_raw_cloud(self, raw_cloud_path):
        """Read a single raw point cloud file.

        This is the method that SPT's preprocessing pipeline calls.
        It dispatches to ``read_uklidar_tile()`` for LAS/LAZ files.

        Args:
            raw_cloud_path: Path to the raw file (LAS, LAZ, or PLY).

        Returns:
            Data object with pos, rgb, intensity, y.
        """
        raw_cloud_path = Path(raw_cloud_path)

        if raw_cloud_path.suffix.lower() in (".las", ".laz"):
            data = read_uklidar_tile(
                raw_cloud_path,
                label_all_void=self.label_all_void,
            )
        elif raw_cloud_path.suffix.lower() == ".ply":
            # Fallback for PLY files (from prepare_spt.py output)
            data = self._read_ply(raw_cloud_path)
        else:
            raise ValueError(
                f"Unsupported file format: {raw_cloud_path.suffix}"
            )

        return data

    @staticmethod
    def _read_ply(filepath):
        """Read an SPT-format PLY file."""
        try:
            from plyfile import PlyData
        except ImportError:
            raise ImportError(
                "plyfile required for PLY reading: pip install plyfile"
            )

        plydata = PlyData.read(str(filepath))
        vertex = plydata["vertex"]

        pos = np.column_stack([
            vertex["x"], vertex["y"], vertex["z"],
        ]).astype(np.float32)

        # Try standard names
        if "red" in vertex.data.dtype.names:
            rgb = np.column_stack([
                vertex["red"], vertex["green"], vertex["blue"],
            ]).astype(np.float32)
            # Normalise if needed
            if rgb.max() > 1.0:
                rgb = rgb / (65535.0 if rgb.max() > 255 else 255.0)
        else:
            rgb = np.zeros((len(pos), 3), dtype=np.float32)

        if "intensity" in vertex.data.dtype.names:
            raw_i = vertex["intensity"].astype(np.float32)
            i_max = raw_i.max()
            intensity = (raw_i / i_max if i_max > 0
                         else np.zeros_like(raw_i)).reshape(-1, 1)
        else:
            intensity = np.zeros((len(pos), 1), dtype=np.float32)

        if "scalar_Classification" in vertex.data.dtype.names:
            y = vertex["scalar_Classification"].astype(np.int64)
        elif "label" in vertex.data.dtype.names:
            y = vertex["label"].astype(np.int64)
        else:
            y = np.full(len(pos), UKLIDAR_NUM_CLASSES, dtype=np.int64)

        return Data(
            pos=torch.from_numpy(pos),
            rgb=torch.from_numpy(rgb),
            intensity=torch.from_numpy(intensity),
            y=torch.from_numpy(y),
        )
