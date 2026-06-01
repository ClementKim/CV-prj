import os
import collections

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

# MINC material categories: ids 0..22 are all real classes (index 0 is NOT background).
NUM_CLASSES = 23
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

    def __init__(self, segments_file = SEGMENTS_FILE, segments_dir = SEGMENTS_DIR,
                 photos_dir = PHOTOS_DIR, img_size = IMG_SIZE,
                 mean = IMAGENET_MEAN, std = IMAGENET_STD):
        self.samples = build_segment_samples(segments_file)
        self.segments_dir = segments_dir
        self.photos_dir = photos_dir
        self.img_size = img_size
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

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


if __name__ == "__main__":
    dataset = MINCSegmentationDataset()
    print(f"photos (scenes): {len(dataset)}")

    image, label = dataset[0]
    classes, counts = np.unique(label.numpy(), return_counts = True)
    print(f"image: {tuple(image.shape)} dtype={image.dtype}")
    print(f"label: {tuple(label.shape)} dtype={label.dtype}")
    print(f"label values (255 = void): {dict(zip(classes.tolist(), counts.tolist()))}")
