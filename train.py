"""Training of the LoRA layers on pairs of density maps and floorplans.

    python train.py --data <data> --output <run>                                                                       # C3
    python train.py --data <data> --output <run> --resolutions 1024 1280 --min_points 1000 --wandb_project <project>  # LaMAR

With --wandb_project, the losses are logged to Weights & Biases, and every --vis_every steps an image of the first
sample of the batch: the floorplan with the ground-truth and predicted correspondences and the position of the last
camera (ground truth and estimated from the predicted correspondences), next to the photo of that camera (data
with the camera_plan, camera_density and rgb columns, see train/lamar/prepare_lamar.py). The images are also saved
to <run>/vis/.
"""
import argparse
import csv
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from scenealigner.inference import SOFT_ARGMAX_TEMPERATURE, load_model, sample_descriptors
from scenealigner.training import (DensityFloorplanDataset, collate, denormalize, feature_matching_loss, geometry_loss,
                                   regression_loss, topology_loss)
from scenealigner.visualization import draw_training_sample, estimate_camera_on_plan

LAMBDA_REGR, LAMBDA_TOPO, LAMBDA_GEO = 50.0, 10.0, 10.0  # the weight of the feature matching loss is 1


def ramp(step, start, duration):
    """Weight of a loss that is introduced linearly from `start` over `duration` iterations."""
    if step < start:
        return 0.0
    if step >= start + duration:
        return 1.0
    return (step - start) / duration


def compute_loss(model, batch, step, total_steps, negatives="scene"):
    """Loss of a batch at a training step, its terms, the weights of the curriculum and diagnostics, and the
    predicted floorplan coordinates (B, N, 2) in [0, 1] of the density-map points.

    negatives: "scene" (the points of other samples of the same scene are not negatives) or "batch" (all points of
    the batch are negatives, e.g. windows at different places of one building)."""
    density_img, plan_img = batch["density_img"].cuda(), batch["plan_img"].cuda()
    pts_density, pts_plan, valid = batch["density_pts"].cuda(), batch["plan_pts"].cuda(), batch["valid"].cuda()
    scene_ids = np.array(batch["scene_id"])
    same_scene = torch.tensor((scene_ids[:, None] == scene_ids[None]) & ~np.eye(len(scene_ids), dtype=bool)).cuda()
    if negatives == "batch":
        same_scene.zero_()

    feature_maps = []
    for img in (density_img, plan_img):
        tokens = model.forward_features(img)["x_norm_patchtokens"]
        grid_dim = img.shape[-1] // 16
        feature_maps.append(tokens.permute(0, 2, 1).reshape(len(img), -1, grid_dim, grid_dim))
    map_density, map_plan = feature_maps
    desc_density = sample_descriptors(map_density, pts_density)
    losses = {"feat": feature_matching_loss(desc_density, sample_descriptors(map_plan, pts_plan), same_scene, valid)}
    loss = losses["feat"]

    # Predicted floorplan coordinates in [0, 1]: soft-argmax over the patch centers
    flat_plan = F.normalize(map_plan.flatten(2).permute(0, 2, 1), dim=-1)
    sim_matrix = torch.matmul(F.normalize(desc_density, dim=-1), flat_plan.transpose(1, 2))
    prob = F.softmax(sim_matrix / SOFT_ARGMAX_TEMPERATURE, dim=-1)
    gy, gx = torch.meshgrid(torch.arange(grid_dim), torch.arange(grid_dim), indexing="ij")
    grid_coords = (torch.stack([gx, gy], dim=-1).float().cuda().reshape(-1, 2) + 0.5) / grid_dim
    pred = torch.matmul(prob, grid_coords)

    # Diagnostics: valid points whose most similar plan patch is within one patch of the ground truth, and the
    # feature matching loss of uniform descriptors (log of the number of valid points of a sample)
    gt_patch = ((pts_plan + 1) / 2 * grid_dim).floor().clamp(0, grid_dim - 1)
    best = sim_matrix.argmax(dim=-1)
    near = ((torch.stack([best % grid_dim, best // grid_dim], dim=-1) - gt_patch).abs().amax(dim=-1) <= 1) & valid
    losses["acc"] = (near.sum() / valid.sum().clamp(min=1)).detach()
    losses["chance"] = valid.sum(dim=1).clamp(min=1).float().log().mean()

    # Curriculum: the regression loss comes in after 10 % of the iterations, the structural ones after 20 %
    w_regr = ramp(step, total_steps // 10, total_steps // 10)
    w_struct = ramp(step, total_steps // 5, total_steps // 10)
    losses["w_regr"], losses["w_struct"] = torch.tensor(w_regr), torch.tensor(w_struct)
    if w_regr > 0:
        weights = prob.max(dim=-1).values.detach() * valid.float()  # confidence of the valid correspondences
        src = (pts_density + 1) / 2
        losses["regr"] = regression_loss(pred, (pts_plan + 1) / 2, weights)
        loss = loss + LAMBDA_REGR * w_regr * losses["regr"]
        if w_struct > 0:
            losses["geo"] = geometry_loss(src, pred, weights)
            loss = loss + LAMBDA_GEO * w_struct * losses["geo"]
            losses["topo"] = topology_loss(src, pred, weights)
            loss = loss + LAMBDA_TOPO * w_struct * losses["topo"]
    return loss, losses, pred


def visualize(batch, pred, ransac_thr=0.04):
    """Image of the first sample of a batch (draw_training_sample), with the camera estimated from the predicted
    correspondences by a similarity transform (RANSAC, as at inference)."""
    plan = denormalize(batch["plan_img"][0])
    pts_density, pts_plan, valid = batch["density_pts"][0].numpy(), batch["plan_pts"][0].numpy(), batch["valid"][0].numpy()
    pred = pred[0].detach().float().cpu().numpy() * 2 - 1  # [0, 1] -> [-1, 1]
    camera_density, camera_plan = batch["camera_density"][0].numpy(), batch["camera_plan"][0].numpy()
    camera_pred = None
    if np.isfinite(camera_density).all():  # pixels of the floorplan at the batch resolution
        size = np.float32(plan.shape[1::-1])
        camera_pred = estimate_camera_on_plan((pts_density[valid] + 1) / 2 * size, (pred[valid] + 1) / 2 * size,
                                              (camera_density + 1) / 2 * size, ransac_thr * plan.shape[0])
        if camera_pred is not None:
            camera_pred = camera_pred / size * 2 - 1
    return draw_training_sample(plan, pts_plan, pred, valid, camera_plan, camera_pred, batch["rgb_path"][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="training data (prepare_c3.py or train/lamar/prepare_lamar.py)")
    parser.add_argument("--output", required=True, help="directory of the LoRA layers and the training log")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 768, 1024, 1280], help="one per batch, at random")
    parser.add_argument("--min_points", type=int, default=10000, help="correspondences that a random crop must keep")
    parser.add_argument("--plan_grid", type=int, default=0, help="one correspondence per cell of this many floorplan px (0: all)")
    parser.add_argument("--negatives", default="scene", choices=["scene", "batch"],
                        help="batch: the points of other samples of the same scene are negatives too")
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", default=None, help="logs the losses and the visualizations to Weights & Biases")
    parser.add_argument("--wandb_name", default=None, help="name of the wandb run")
    parser.add_argument("--vis_every", type=int, default=500, help="steps between two visualizations (0: none)")
    args = parser.parse_args()
    wandb = None
    if args.wandb_project:
        import wandb  # optional dependency

        wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    image_size = args.resolutions[0] if len(args.resolutions) == 1 else None
    dataset = DensityFloorplanDataset(args.data, min_points=args.min_points, image_size=image_size, plan_grid=args.plan_grid)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=partial(collate, resolutions=args.resolutions))
    total_steps = len(loader) * args.epochs

    lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=["qkv", "proj", "fc1", "fc2"])
    model = get_peft_model(load_model("pre-trained", args.dinov3_weights), lora)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    os.makedirs(os.path.join(args.output, "vis"), exist_ok=True)
    # Resume after the last completed epoch (LoRA layers, optimizer, step), e.g. when a job reaches its walltime
    state_path = os.path.join(args.output, "state.pt")
    first_epoch, step = 0, 0
    if os.path.exists(state_path):
        state = torch.load(state_path, weights_only=True)
        set_peft_model_state_dict(model, load_peft_weights(args.output))
        optimizer.load_state_dict(state["optimizer"])
        first_epoch, step = state["epoch"] + 1, state["step"]
        print(f"resumed after epoch {first_epoch} (step {step})")
    columns = ("feat", "regr", "geo", "topo", "acc", "chance")
    log = csv.writer(open(os.path.join(args.output, "log.csv"), "a" if first_epoch else "w", buffering=1))
    if not first_epoch:
        log.writerow(["step", "loss", *columns])
    for epoch in range(first_epoch, args.epochs):
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}"):
            optimizer.zero_grad()
            loss, losses, pred = compute_loss(model, batch, step, total_steps, args.negatives)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                log.writerow([step, loss.item()] + [losses[k].item() if k in losses else "" for k in columns])
                if wandb:
                    wandb.log({"loss": loss.item(), "epoch": epoch, **{k: v.item() for k, v in losses.items()}}, step=step)
            if args.vis_every and step % args.vis_every == 0:
                image = visualize(batch, pred)
                Image.fromarray(image).save(os.path.join(args.output, "vis", f"{step:06d}.png"))
                if wandb:
                    wandb.log({"sample": wandb.Image(image, caption=f"step {step}: {batch['scene_id'][0]}")}, step=step)
        model.save_pretrained(args.output)
        torch.save({"epoch": epoch, "step": step, "optimizer": optimizer.state_dict()}, state_path)


if __name__ == "__main__":
    main()
