"""Unified segmentation preprocessing for two material-segmentation sources.

Both datasets emit the SAME per-item contract, so the existing ``seg_collate_fn``
is reused unchanged:

    image_t : float32 [3, IMG_SIZE, IMG_SIZE]   ImageNet-normalized
    label_t : int64   [IMG_SIZE, IMG_SIZE]      class ids in 0..num_classes-1,
                                                IGNORE_INDEX (255) = void/unlabeled

Sources
-------
1. MINC   : ``minc/`` shape masks + ``photo_orig/`` images (reuses the proven
            MINCSegmentationDataset from preprocessing.py). 23 materials.
2. OpenSurfaces : the rendered output of process_opensurfaces_release_0.py,
            ``opensurfaces-data/photos/<id>.jpg`` + ``photos-labels/<id>.png``,
            where the RED channel encodes the material id (1..45, 0 = unlabeled).
            45 materials.

Use ``build_dataset(name)`` with name in {"minc", "opensurfaces", "both"}.
"both" concatenates them into a single DISJOINT label space (MINC = 0..22,
OpenSurfaces = 23..67, num_classes = 68) so one segmentation head can train on
both without manual material-vocabulary alignment.
"""

import os
import csv

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

# Reuse the working MINC dataset, the collate fn, and the shared constants.
from preprocessing import (
    MINCSegmentationDataset as _MINCSegmentationDataset,
    seg_collate_fn,
    IGNORE_INDEX,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES as MINC_NUM_CLASSES,
)

__all__ = [
    "MINCSegmentationDataset",
    "OpenSurfacesSegmentationDataset",
    "ConcatSegmentationDataset",
    "build_dataset",
    "seg_collate_fn",
    "IGNORE_INDEX",
    "IMG_SIZE",
]


# --------------------------------------------------------------------------- #
# 1) MINC : minc/ masks + photo_orig/ images
# --------------------------------------------------------------------------- #

MINC_CATEGORIES_FILE = os.path.join("minc", "categories.txt")


def _load_minc_class_names(path=MINC_CATEGORIES_FILE):
    if os.path.exists(path):
        with open(path) as f:
            names = [line.strip() for line in f if line.strip()]
        if len(names) == MINC_NUM_CLASSES:
            return names
    return [str(i) for i in range(MINC_NUM_CLASSES)]


class MINCSegmentationDataset(_MINCSegmentationDataset):
    """MINC-S material segmentation. Labels 0..22, IGNORE_INDEX = void.

    Thin wrapper over preprocessing.MINCSegmentationDataset that adds the
    ``name`` / ``num_classes`` / ``class_names`` attributes the unified layer
    relies on. Item format and label semantics are unchanged.
    """

    name = "minc"
    num_classes = MINC_NUM_CLASSES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_names = _load_minc_class_names()


# --------------------------------------------------------------------------- #
# 2) OpenSurfaces : rendered photos/ + photos-labels/ (red channel = material)
# --------------------------------------------------------------------------- #

OPENSURFACES_ROOT = "opensurfaces-data"

# Materials in red-channel order (red_color 1..45, sorted by name) exactly as
# emitted by process_opensurfaces_release_0.py. Used as a fallback when
# label-substance-colors.csv has not been generated yet.
OPENSURFACES_MATERIALS = [
    "Brick", "Cardboard", "Carpet/rug", "Ceramic", "Chalkboard/blackboard",
    "Concrete", "Cork/corkboard", "Dirt", "Fabric/cloth", "Fire", "Foliage",
    "Food", "Fur", "Glass", "Granite", "Granite/marble", "Hair", "Laminate",
    "Leather", "Linoleum", "Marble", "Metal", "Mirror", "Painted",
    "Paper towel/tissue", "Paper/tissue", "Plaster", "Plastic - clear",
    "Plastic - opaque", "Rubber/latex", "Skin", "Sky", "Sponge", "Stone",
    "Styrofoam", "Tile", "Wallboard - painted", "Wallboard - unpainted",
    "Wallpaper", "Water", "Wax", "Wicker", "Wood", "Wood - natural color",
    "Wood - painted",
]


def _load_opensurfaces_class_names(root):
    """Material names ordered by red_color (1..N).

    Prefers the authoritative label-substance-colors.csv written by the helper
    script; falls back to the embedded list if it has not been generated yet.
    """
    csv_path = os.path.join(root, "label-substance-colors.csv")
    if os.path.exists(csv_path):
        rows = []
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append((int(r["red_color"]), r["substance_name"]))
        rows.sort()
        return [name for _, name in rows]
    return list(OPENSURFACES_MATERIALS)


class OpenSurfacesSegmentationDataset(Dataset):
    """OpenSurfaces material segmentation from the rendered helper output.

    Reads ``<root>/photos/<id>.jpg`` and ``<root>/photos-labels/<id>.png`` (RGB;
    RED channel = material id 1..N, 0 = unlabeled). Material ids are remapped to
    0-indexed (red - 1) and unlabeled pixels to IGNORE_INDEX, matching MINC.

    The label map is rendered at the photo's aspect ratio, so resizing the image
    (bilinear) and the label map (nearest) independently to the IMG_SIZE square
    keeps them aligned -- same assumption as MINCSegmentationDataset.
    """

    name = "opensurfaces"

    def __init__(self, root=OPENSURFACES_ROOT, img_size=IMG_SIZE,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.root = root
        self.photos_dir = os.path.join(root, "photos")
        self.labels_dir = os.path.join(root, "photos-labels")

        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(
                f"{self.labels_dir} not found. Generate the label maps first by "
                f"running process_opensurfaces_release_0.py with "
                f"RENDER_PHOTO_LABELS=True (and DOWNLOAD_PHOTO_IMAGES=True)."
            )

        # One sample per rendered label map that also has a downloaded photo.
        ids = []
        for fn in sorted(os.listdir(self.labels_dir)):
            if not fn.endswith(".png"):
                continue
            pid = fn[:-4]
            if os.path.exists(os.path.join(self.photos_dir, pid + ".jpg")):
                ids.append(pid)
        if not ids:
            raise RuntimeError(
                f"No (photo, label) pairs found under {root!r}. Ensure both "
                f"photos/ and photos-labels/ are populated by the helper script."
            )
        self.ids = ids

        self.img_size = img_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

        self.class_names = _load_opensurfaces_class_names(root)
        self.num_classes = len(self.class_names)

    def __len__(self):
        return len(self.ids)

    def _label_map(self, idx):
        """Return the [H, W] uint8 label map (0-indexed ids, IGNORE_INDEX void)."""
        pid = self.ids[idx]
        png = np.asarray(
            Image.open(os.path.join(self.labels_dir, pid + ".png")).convert("RGB")
        )
        red = png[..., 0].astype(np.int64)               # material id, 1..N (0 = void)
        label_map = np.full(red.shape, IGNORE_INDEX, dtype=np.uint8)
        painted = red > 0
        label_map[painted] = (red[painted] - 1).astype(np.uint8)   # -> 0-indexed
        return label_map

    def _resize_label(self, label_map):
        return np.asarray(
            Image.fromarray(label_map).resize(
                (self.img_size, self.img_size), Image.NEAREST),
            dtype=np.int64,
        )

    def __getitem__(self, idx):
        pid = self.ids[idx]

        image = Image.open(os.path.join(self.photos_dir, pid + ".jpg")).convert("RGB")
        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        image_t = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        image_t = (image_t - self.mean) / self.std

        label_t = torch.from_numpy(self._resize_label(self._label_map(idx)))
        return image_t, label_t

    def label_only(self, idx):
        """Just the [H, W] resized label map (no photo decode), for class weights."""
        return torch.from_numpy(self._resize_label(self._label_map(idx)))


# --------------------------------------------------------------------------- #
# Concatenation into a single disjoint label space
# --------------------------------------------------------------------------- #

class ConcatSegmentationDataset(Dataset):
    """Concatenate segmentation datasets into ONE disjoint label space.

    Dataset k's class ids are shifted by the cumulative num_classes of earlier
    datasets (IGNORE_INDEX is preserved), so a single head with
    ``num_classes = sum(d.num_classes)`` can train on all of them. This avoids
    hand-aligning the MINC (23) and OpenSurfaces (45) material vocabularies; if
    you instead want a merged vocabulary, remap labels before concatenating.
    """

    name = "concat"

    def __init__(self, datasets):
        self.datasets = list(datasets)
        if not self.datasets:
            raise ValueError("ConcatSegmentationDataset needs at least one dataset")

        self.offsets = []
        offset = 0
        self.class_names = []
        for d in self.datasets:
            self.offsets.append(offset)
            offset += d.num_classes
            self.class_names += [f"{d.name}:{c}" for c in d.class_names]
        self.num_classes = offset

        lengths = [len(d) for d in self.datasets]
        self.boundaries = np.cumsum([0] + lengths)

    def __len__(self):
        return int(self.boundaries[-1])

    def _route(self, idx):
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        k = int(np.searchsorted(self.boundaries, idx, side="right") - 1)
        return k, idx - int(self.boundaries[k])

    def _shift(self, label_t, k):
        offset = self.offsets[k]
        if offset:
            label_t = label_t.clone()
            keep = label_t != IGNORE_INDEX
            label_t[keep] += offset
        return label_t

    def __getitem__(self, idx):
        k, local = self._route(idx)
        image_t, label_t = self.datasets[k][local]
        return image_t, self._shift(label_t, k)

    def label_only(self, idx):
        k, local = self._route(idx)
        return self._shift(self.datasets[k].label_only(local), k)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def build_dataset(name="minc", opensurfaces_root=OPENSURFACES_ROOT):
    """Build a segmentation dataset by name.

    name : "minc" | "opensurfaces" | "both"
    Returns a dataset exposing ``num_classes``, ``class_names``, ``__len__``,
    ``__getitem__`` -> (image_t, label_t), and ``label_only(idx)``.
    Pair it with ``seg_collate_fn``.
    """
    key = name.lower()
    if key == "minc":
        return MINCSegmentationDataset()
    if key in ("opensurfaces", "os"):
        return OpenSurfacesSegmentationDataset(root=opensurfaces_root)
    if key in ("both", "concat", "all"):
        return ConcatSegmentationDataset([
            MINCSegmentationDataset(),
            OpenSurfacesSegmentationDataset(root=opensurfaces_root),
        ])
    raise ValueError(f"unknown dataset {name!r} (choose: minc, opensurfaces, both)")


if __name__ == "__main__":
    def _summarize(tag, ds):
        image, label = ds[0]
        classes, counts = np.unique(label.numpy(), return_counts=True)
        print(f"[{tag}] scenes={len(ds)} num_classes={ds.num_classes}")
        print(f"       image={tuple(image.shape)} {image.dtype}  "
              f"label={tuple(label.shape)} {label.dtype}")
        print(f"       sample[0] label values (255=void): "
              f"{dict(zip(classes.tolist(), counts.tolist()))}")

    print("=== MINC ===")
    _summarize("minc", build_dataset("minc"))

    print("=== OpenSurfaces ===")
    try:
        _summarize("opensurfaces", build_dataset("opensurfaces"))
    except (FileNotFoundError, RuntimeError) as e:
        print(f"       skipped: {e}")

    print("=== both (disjoint concat) ===")
    try:
        both = build_dataset("both")
        print(f"[both] scenes={len(both)} num_classes={both.num_classes} "
              f"(offsets={both.offsets})")
    except (FileNotFoundError, RuntimeError) as e:
        print(f"       skipped: {e}")
