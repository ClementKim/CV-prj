import os
import csv
import torch
import collections

import numpy as np

from PIL import Image
from torch.utils.data import Dataset

# MINC material categories: ids 0..22 are all real classes (index 0 is NOT background).
MINC_NUM_CLASSES = 23
# Pixels not covered by any labeled shape are void and ignored by the loss / mIoU.
IGNORE_INDEX = 255
# The model has a fixed-size positional embedding, so inputs must be this square size.
IMG_SIZE = 512

# ImageNet RGB normalization (training from scratch). Switch to BGR mean-subtraction
# [104, 117, 124] only if loading pretrained MINC weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Repo-relative defaults for the MINC-S segmentation split.
SEGMENTS_FILE = os.path.join("minc", "minc-s", "test-segments.txt")
SEGMENTS_DIR = os.path.join("minc", "minc-s", "segments")
PHOTOS_DIR = "photo_orig"

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

def build_segment_samples(segments_file = SEGMENTS_FILE):
    """Parse test-segments.txt ("label,photo_id,shape_id") and group shapes by photo.

    Returns a list of (photo_id, [(label, shape_id), ...]) so each dataset item is
    one full scene with all of its labeled material regions.
    """
    by_photo = collections.OrderedDict()
    with open(segments_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, photo_id, shape_id = line.split(",")
            by_photo.setdefault(photo_id, []).append((int(label), shape_id))
    return list(by_photo.items())


def find_photo(photo_id, photos_dir = PHOTOS_DIR):
    """Locate the source jpg for a photo id.

    Supports the MINC sharded layout `photo_orig/<last_digit>/<photo_id>.jpg`
    as well as a flat `<photos_dir>/<photo_id>.jpg`.
    """
    candidates = [
        os.path.join(photos_dir, photo_id[-1], photo_id + ".jpg"),
        os.path.join(photos_dir, photo_id + ".jpg"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Photo {photo_id} not found. Tried: {candidates}"
    )

class MINCSegmentationDataset(Dataset):
    """MINC-S full-scene material segmentation.

    Each photo owns one or more binary shape masks at `segments_dir`/`<photo_id>_<shape_id>.png`
    (PIL mode "1"); test-segments.txt assigns a material label to each shape. We fuse the
    shapes into a single [H, W] label map (background = IGNORE_INDEX), painting the
    largest-area shape first so smaller shapes win on the rare overlap. The image is resized
    bilinearly and the label map with nearest-neighbor to a fixed IMG_SIZE square, which the
    model requires. Masks and photos share the original aspect ratio, so resizing each
    independently to the square keeps them aligned.
    """

    name = "minc"
    num_classes = MINC_NUM_CLASSES
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.samples = build_segment_samples(SEGMENTS_FILE)
        self.segments_dir = SEGMENTS_DIR
        self.photos_dir = PHOTOS_DIR
        self.img_size = IMG_SIZE
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        self.class_names = _load_minc_class_names()

    def __len__(self):
        return len(self.samples)

    def _fuse_label_map(self, photo_id, shapes):
        loaded = []
        for label, shape_id in shapes:
            mask_path = os.path.join(self.segments_dir, f"{photo_id}_{shape_id}.png")
            mask = np.array(Image.open(mask_path).convert("1"), dtype = bool)
            loaded.append((int(mask.sum()), label, mask))

        # Paint largest area first so smaller, more specific shapes overwrite on overlap.
        loaded.sort(key = lambda t: t[0], reverse = True)

        height, width = loaded[0][2].shape
        label_map = np.full((height, width), IGNORE_INDEX, dtype = np.uint8)
        for _, label, mask in loaded:
            label_map[mask] = label
        return label_map

    def __getitem__(self, idx):
        photo_id, shapes = self.samples[idx]

        image = Image.open(find_photo(photo_id, self.photos_dir)).convert("RGB")
        label_map = self._fuse_label_map(photo_id, shapes)

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        label_image = Image.fromarray(label_map).resize(
            (self.img_size, self.img_size), Image.NEAREST)

        image_t = torch.from_numpy(np.asarray(image, dtype = np.float32) / 255.0).permute(2, 0, 1)
        image_t = (image_t - self.mean) / self.std

        label_t = torch.from_numpy(np.asarray(label_image, dtype = np.int64))

        return image_t, label_t

    def label_only(self, idx):
        """Return just the fused, resized [H, W] label map for an index (no photo decode).

        Used to tally per-class pixel frequencies over the train split for loss
        weighting without paying the cost of loading/normalizing every image.
        """
        photo_id, shapes = self.samples[idx]
        label_map = self._fuse_label_map(photo_id, shapes)
        label_image = Image.fromarray(label_map).resize(
            (self.img_size, self.img_size), Image.NEAREST)
        return torch.from_numpy(np.asarray(label_image, dtype = np.int64))

def seg_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim = 0)
    labels = torch.stack([item[1] for item in batch], dim = 0)
    return images, labels

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
