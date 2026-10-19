"""Training of the LoRA layers on pairs of density maps and floorplans.

    python train.py --data <data> --output <run>
"""
import argparse
import csv
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from tqdm import tqdm

from scenealigner.inference import SOFT_ARGMAX_TEMPERATURE, load_model, sample_descriptors
from scenealigner.training import (DensityFloorplanDataset, collate, feature_matching_loss, geometry_loss,
                                   regression_loss, topology_loss)

LAMBDA_REGR, LAMBDA_TOPO, LAMBDA_GEO = 50.0, 10.0, 10.0  # the weight of the feature matching loss is 1


def ramp(step, start, duration):
    """Weight of a loss that is introduced linearly from `start` over `duration` iterations."""
    if step < start:
        return 0.0
    if step >= start + duration:
        return 1.0
    return (step - start) / duration


def compute_loss(model, batch, step, total_steps):
    """Loss of a batch at a training step, and its terms."""
    density_img, plan_img = batch["density_img"].cuda(), batch["plan_img"].cuda()
    pts_density, pts_plan, valid = batch["density_pts"].cuda(), batch["plan_pts"].cuda(), batch["valid"].cuda()
    scene_ids = np.array(batch["scene_id"])
    same_scene = torch.tensor((scene_ids[:, None] == scene_ids[None]) & ~np.eye(len(scene_ids), dtype=bool)).cuda()

    feature_maps = []
    for img in (density_img, plan_img):
        tokens = model.forward_features(img)["x_norm_patchtokens"]
        grid_dim = img.shape[-1] // 16
        feature_maps.append(tokens.permute(0, 2, 1).reshape(len(img), -1, grid_dim, grid_dim))
    map_density, map_plan = feature_maps
    desc_density = sample_descriptors(map_density, pts_density)
    losses = {"feat": feature_matching_loss(desc_density, sample_descriptors(map_plan, pts_plan), same_scene, valid)}
    loss = losses["feat"]

    # Curriculum: the regression loss comes in after 10 % of the iterations, the structural ones after 20 %
    w_regr = ramp(step, total_steps // 10, total_steps // 10)
    w_struct = ramp(step, total_steps // 5, total_steps // 10)
    if w_regr > 0:
        # Predicted floorplan coordinates in [0, 1]: soft-argmax over the patch centers
        flat_plan = F.normalize(map_plan.flatten(2).permute(0, 2, 1), dim=-1)
        sim_matrix = torch.matmul(F.normalize(desc_density, dim=-1), flat_plan.transpose(1, 2))
        prob = F.softmax(sim_matrix / SOFT_ARGMAX_TEMPERATURE, dim=-1)
        gy, gx = torch.meshgrid(torch.arange(grid_dim), torch.arange(grid_dim), indexing="ij")
        grid_coords = (torch.stack([gx, gy], dim=-1).float().cuda().reshape(-1, 2) + 0.5) / grid_dim
        pred = torch.matmul(prob, grid_coords)
        weights = prob.max(dim=-1).values.detach() * valid.float()  # confidence of the valid correspondences
        src = (pts_density + 1) / 2
        losses["regr"] = regression_loss(pred, (pts_plan + 1) / 2, weights)
        loss = loss + LAMBDA_REGR * w_regr * losses["regr"]
        if w_struct > 0:
            losses["geo"] = geometry_loss(src, pred, weights)
            loss = loss + LAMBDA_GEO * w_struct * losses["geo"]
            losses["topo"] = topology_loss(src, pred, weights)
            loss = loss + LAMBDA_TOPO * w_struct * losses["topo"]
    return loss, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="training data (prepare_c3.py)")
    parser.add_argument("--output", required=True, help="directory of the LoRA layers and the training log")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 768, 1024, 1280], help="one per batch, at random")
    parser.add_argument("--min_points", type=int, default=10000, help="correspondences that a random crop must keep")
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    image_size = args.resolutions[0] if len(args.resolutions) == 1 else None
    dataset = DensityFloorplanDataset(args.data, min_points=args.min_points, image_size=image_size)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=partial(collate, resolutions=args.resolutions))
    total_steps = len(loader) * args.epochs

    lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=["qkv", "proj", "fc1", "fc2"])
    model = get_peft_model(load_model("pre-trained", args.dinov3_weights), lora)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    os.makedirs(args.output, exist_ok=True)
    log = csv.writer(open(os.path.join(args.output, "log.csv"), "w", buffering=1))
    log.writerow(["step", "loss", "feat", "regr", "geo", "topo"])
    step = 0
    for epoch in range(args.epochs):
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            optimizer.zero_grad()
            loss, losses = compute_loss(model, batch, step, total_steps)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                log.writerow([step, loss.item()] + [losses[k].item() if k in losses else "" for k in ("feat", "regr", "geo", "topo")])
        model.save_pretrained(args.output)


if __name__ == "__main__":
    main()
