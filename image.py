import os
import random
import colorsys
import argparse

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")                 # file output, no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from torch.utils.data import random_split

from main import ProposedMethod
from baseline import load_model as load_baseline_model
from encoder import DualEncoder
from transformer import SelfExpert, CrossExpert, TransformerDecoder
from preprocessing2 import build_dataset, IGNORE_INDEX, IMG_SIZE
from preprocessing import IMAGENET_MEAN, IMAGENET_STD

# Baseline architectures live in baseline.py (segmentation_models_pytorch); anything
# else is treated as the proposed InteractSeg model defined in main.py.
BASELINE_MODELS = ["unet", "unet++", "fpn", "pspnet", "deeplabv3", "deeplabv3+",
                   "linknet", "manet", "pan", "upernet", "segformer", "dpt"]


def build_palette(num_classes, seed=0):
    """Deterministic, well-separated RGB colors. Index 255 (void) stays black."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed)
    golden = 0.618033988749895
    h = rng.random()
    for i in range(num_classes):
        h = (h + golden) % 1.0
        s = 0.55 + 0.35 * ((i % 3) / 2.0)       # vary saturation / value so
        v = 0.75 + 0.20 * (i % 2)               # neighbours don't collide
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        palette[i] = (int(r * 255), int(g * 255), int(b * 255))
    return palette


def colorize(label, palette):
    """[H, W] class ids (255 = void) -> [H, W, 3] uint8 via the palette."""
    return palette[np.clip(label.astype(np.int64), 0, 255)]


def denorm(image_t):
    """ImageNet-normalized [3, H, W] tensor -> [H, W, 3] float image in 0..1."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (image_t.detach().cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def get_test_split(dataset):
    """Reproduce main.py's 80/10/10 split (fixed generator seed 42) and return test_set."""
    n = len(dataset)
    test_size = max(1, int(0.1 * n))
    val_size = max(1, int(0.1 * n))
    train_size = n - val_size - test_size
    _, _, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42))
    return test_set


def load_checkpoint(path):
    """Return the model state_dict, tolerating both raw and wrapped checkpoints."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    return ckpt


def main(args):
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    # Allow TF32 on Ampere+ GPUs for faster matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = build_dataset(args.dataset)
    test_set = get_test_split(dataset)

    state = load_checkpoint(args.ckpt_path)
    is_baseline = args.model.lower() in BASELINE_MODELS

    if is_baseline:
        # Baseline checkpoints are segmentation_models_pytorch models with a totally
        # different key layout than ProposedMethod, so we MUST rebuild the matching smp
        # architecture or none of the weights load. num_classes comes straight from the
        # trained segmentation head: segmentation_head.0.weight is [num_classes, C, k, k].
        head = next((v for k, v in state.items()
                     if k.endswith("segmentation_head.0.weight")), None)
        num_classes = head.shape[0] if head is not None else dataset.num_classes
        if num_classes == dataset.num_classes:
            class_names = dataset.class_names
        else:
            print(f"[warn] checkpoint has {num_classes} classes but dataset "
                  f"'{args.dataset}' has {dataset.num_classes}; using generic names.")
            class_names = [str(i) for i in range(num_classes)]
        model = load_baseline_model(args.model, num_classes).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)
    else:
        # Infer num_classes and embed_dim straight from the trained head so the model
        # always matches the checkpoint, regardless of the dataset on disk or the
        # --embed_dim default. classifier.weight is [num_classes, embed_dim].
        if "decoder.classifier.weight" in state:
            num_classes = state["decoder.classifier.weight"].shape[0]
            embed_dim = state["decoder.classifier.weight"].shape[1]
            if embed_dim != args.embed_dim:
                print(f"[info] checkpoint embed_dim={embed_dim} overrides "
                      f"--embed_dim {args.embed_dim}")
        else:
            num_classes = dataset.num_classes
            embed_dim = args.embed_dim
        if num_classes == dataset.num_classes:
            class_names = dataset.class_names
        else:
            print(f"[warn] checkpoint has {num_classes} classes but dataset "
                  f"'{args.dataset}' has {dataset.num_classes}; using generic names.")
            class_names = [str(i) for i in range(num_classes)]

        encoder = DualEncoder(IMG_SIZE, embed_dim, pretrained=args.pretrained)
        self_expert_pool = nn.ModuleList([
            SelfExpert(img_size=embed_dim, patch_size=args.patch_size, in_channels=1,
                       embed_dim=embed_dim, num_heads=args.num_heads,
                       mlp_ratio=args.mlp_ratio, drop=args.drop, attn_drop=args.attn_drop)
            for _ in range(2)])
        cross_expert_pool = nn.ModuleList([
            CrossExpert(img_size=embed_dim, patch_size=args.patch_size, in_channels=1,
                        embed_dim=embed_dim, num_heads=args.num_heads,
                        mlp_ratio=args.mlp_ratio, drop=args.drop, attn_drop=args.attn_drop)
            for _ in range(2)])
        auxiliary_tokens = torch.randn(embed_dim, embed_dim)
        decoder = TransformerDecoder(embed_dim=embed_dim, num_classes=num_classes,
                                     num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
                                     drop=args.drop, attn_drop=args.attn_drop)

        model = ProposedMethod(encoder, self_expert_pool, cross_expert_pool,
                               auxiliary_tokens, decoder).to(device)
        missing, unexpected = model.load_state_dict(state, strict=False)

    # Fail loud if the checkpoint doesn't actually fit the model: a wholesale mismatch
    # (e.g. a baseline ckpt loaded into ProposedMethod) means we'd silently visualize an
    # untrained network, which is exactly the "all models look identical" bug.
    if len(missing) > 0.5 * len(model.state_dict()):
        raise SystemExit(
            f"[error] {len(missing)} missing keys loading '{args.ckpt_path}' into "
            f"'{args.model}' -- checkpoint and model architecture do not match. "
            f"Pass the correct --model for this checkpoint.")
    if missing:
        print(f"[warn] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.eval()

    # Pick which test samples to show.
    if args.indices:
        indices = args.indices
    else:
        n = len(test_set)
        step = max(1, n // args.num_images)
        indices = list(range(0, n, step))[:args.num_images]
    print(f"Visualizing {len(indices)} test image(s) of {len(test_set)}; indices={indices}")

    palette = build_palette(num_classes)
    rows = []
    present = set()
    for idx in indices:
        image_t, label_t = test_set[idx]
        with torch.no_grad():
            logits = model(image_t.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1)[0].cpu().numpy()
        gt = label_t.cpu().numpy()

        present.update(np.unique(gt).tolist())
        present.update(np.unique(pred).tolist())
        rows.append((denorm(image_t), colorize(gt, palette), colorize(pred, palette)))

    # ---- assemble the figure: rows x [input | ground truth | prediction] ----
    titles = ["Input", "Ground Truth", "Prediction"]
    fig, axes = plt.subplots(len(rows), 3, figsize=(12, 4 * len(rows)), squeeze=False)
    for r, panels in enumerate(rows):
        for c, panel in enumerate(panels):
            ax = axes[r][c]
            ax.imshow(panel)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c], fontsize=13)

    # # Legend of the materials that actually appear (skip void).
    # handles = [
    #     Patch(facecolor=np.array(palette[cid]) / 255.0,
    #           label=f"{cid}: {class_names[cid]}" if cid < len(class_names) else str(cid))
    #     for cid in sorted(present) if cid != IGNORE_INDEX
    # ]
    # if handles:
    #     fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)),
    #                fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.0))
    
    fig.suptitle(f"{args.dataset} test set | ckpt={os.path.basename(args.ckpt_path)}",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved visualization -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--model", type=str, default="proposed",
                        choices=["proposed"] + BASELINE_MODELS,
                        help="'proposed' = InteractSeg (main.py); otherwise the matching "
                             "segmentation_models_pytorch baseline from baseline.py")
    parser.add_argument("--dataset", type=str, default="minc",
                        choices=["minc", "opensurfaces", "both"])
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="explicit sample indices into the test split (overrides --num_images)")
    parser.add_argument("--output", type=str, default="output/predictions.png")

    # Must match the trained model's hyperparameters.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--mlp_ratio", type=float, default=2.0)
    parser.add_argument("--drop", type=float, default=0.1)
    parser.add_argument("--attn_drop", type=float, default=0.1)

    # Backbone weights come from the checkpoint, so skip the ImageNet download by default.
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false")

    args = parser.parse_args()
    main(args)
