import os
import timm
import torch
import random
import argparse

import numpy as np
import torch.nn as nn
import segmentation_models_pytorch as smp

from tqdm import tqdm
from main import AugmentSeg, compute_class_weights, train_step, evaluate, loss_fn
from torch.utils.data import DataLoader, random_split
from preprocessing import build_dataset, seg_collate_fn, IGNORE_INDEX, IMG_SIZE

def load_model(model_name, num_classes):
    if model_name.lower() == "unet":
        model = smp.Unet(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "unet++":
        model = smp.UnetPlusPlus(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "fpn":
        model = smp.FPN(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "pspnet":
        model = smp.PSPNet(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "deeplabv3+":
        model = smp.DeepLabV3Plus(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "linknet":
        model = smp.Linknet(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "manet":
        model = smp.MAnet(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "pan":
        model = smp.PAN(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "upernet":
        model = smp.UPerNet(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "segformer":
        model = smp.Segformer(
            encoder_name = "resnet34",
            encoder_weights = "imagenet",
            classes = num_classes
        )

    elif model_name.lower() == "dpt":
        # ViT encoder has a fixed patch grid / positional embedding; tell it the
        # real input resolution so it builds (and interpolates the pretrained
        # pos-embed for) IMG_SIZE instead of the backbone's default 224.
        model = smp.DPT(
            encoder_name = "tu-vit_base_patch16_224.augreg_in21k",
            encoder_weights = "imagenet",
            classes = num_classes,
            img_size = IMG_SIZE
        )

    else:
        raise ValueError(f"Model {model_name} not recognized. Please choose from: unet, unet++, fpn, pspnet, deeplabv3, deeplabv3+, linknet, manet, pan, upernet, segformer, dpt.")

    return model

def main(args):
    batch = args.batch
    lr = args.lr
    weight_decay = args.weight_decay

    seed = args.seed

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only = True)

    # Allow TF32 on Ampere+ GPUs for faster matmuls
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    dataset = build_dataset(args.dataset)
    num_classes = dataset.num_classes
    n = len(dataset)
    test_size = max(1, int(0.1 * n))
    val_size = max(1, int(0.1 * n))
    train_size = n - val_size - test_size                   # 80 / 10 / 10 train / val / test
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator = torch.Generator().manual_seed(seed))

    train_loader = DataLoader(AugmentSeg(train_set), batch_size = batch, shuffle = True,
                              collate_fn = seg_collate_fn, num_workers = args.workers, pin_memory = True)
    val_loader = DataLoader(val_set, batch_size = batch, shuffle = False,
                            collate_fn = seg_collate_fn, num_workers = args.workers, pin_memory = True)
    test_loader = DataLoader(test_set, batch_size = batch, shuffle = False,
                             collate_fn = seg_collate_fn, num_workers = args.workers, pin_memory = True)

    model = load_model(args.model, num_classes).cuda()

    if args.weight_pow > 0:
        ckpt_path = os.path.join('ckpt', f'clsw_{args.dataset}_n{n}_c{num_classes}_p{args.weight_pow}_m{args.max_class_weight}.pt')
        class_weights = compute_class_weights(dataset, train_set.indices, num_classes, args.weight_pow, args.max_class_weight, device, ckpt_path)
        print('Class weights (present):', class_weights[class_weights > 0].cpu().numpy().round(2))
    else:
        class_weights = None

    backbone_params, head_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (backbone_params if name.startswith(('encoder.vit.vit.', 'encoder.res.')) else head_params).append(p)

    optimizer = torch.optim.AdamW(
        [{'params': backbone_params, 'lr': lr * args.backbone_lr_mult},
         {'params': head_params, 'lr': lr}],
        lr = lr, weight_decay = weight_decay)
    
    scaler = torch.amp.GradScaler('cuda', enabled = (device == 'cuda'))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    if warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.01, total_iters = warmup_steps)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max(1, total_steps - warmup_steps))
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones = [warmup_steps])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = max(1, total_steps))

    best_miou = -1.0
    best_path = os.path.join('ckpt', f'best_{args.dataset}_c{num_classes}_{args.model}.pt')
    for epoch in range(1, args.epochs + 1):
        running_loss, num_batches = 0.0, 0
        for batch_data in tqdm(train_loader, desc=f'Epoch {epoch}'):
            loss = train_step(model, batch_data, class_weights, optimizer, scaler, device)
            scheduler.step()
            if loss is not None:
                running_loss += loss; num_batches += 1
        print(f'Epoch: {epoch}, Average Loss: {running_loss / max(1, num_batches):.4f}, LR: {scheduler.get_last_lr()[0]:.2e}')

        _, miou = evaluate(model, val_loader, num_classes, device)
        if miou > best_miou:
            best_miou = miou
            os.makedirs('ckpt', exist_ok = True)
            torch.save(model.state_dict(), best_path)
            print(f'  -> new best val mIoU {best_miou:.4f} (checkpoint saved)')

    print('Test (best val checkpoint):')
    model.load_state_dict(torch.load(best_path))
    evaluate(model, test_loader, num_classes, device)

if __name__ == "__main__":
    arg = argparse.ArgumentParser()

    arg.add_argument('--model', type = str, default = 'unet', choices = ['unet', 'unet++', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3+', 'linknet', 'manet', 'pan', 'upernet', 'segformer', 'dpt'])
    
    arg.add_argument('--seed', type = int, default = 42)

    arg.add_argument('--dataset', type = str, default = 'both', choices = ['minc', 'opensurfaces', 'both'])

    arg.add_argument('--batch', type = int, default = 16)
    arg.add_argument('--epochs', type = int, default = 30)
    arg.add_argument('--warmup_epochs', type = int, default = 1)
    
    arg.add_argument('--lr', type = float, default = 3e-4)
    arg.add_argument('--weight_decay', type = float, default = 0.05)

    arg.add_argument('--weight_pow', type = float, default = 0.5)        # class weight = (1 / freq) ** weight_pow; 0 disables
    arg.add_argument('--max_class_weight', type = float, default = 10.0)

    arg.add_argument('--workers', type = int, default = 8)
    arg.add_argument('--backbone_lr_mult', type = float, default = 0.1)  # pretrained backbone LR = lr * this
    arg.add_argument('--pretrained', action = 'store_true', default = True)
    arg.add_argument('--no_pretrained', dest = 'pretrained', action = 'store_false')

    args = arg.parse_args()
    main(args)