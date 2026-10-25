"""Training of the LoRA layers on pairs of density maps and floorplans.

    python train.py --data <data> --output <run>                                                                       # C3
    python train.py --data <data> --output <run> --resolutions 1024 1280 --min_points 1000 --wandb_project <project>  # LaMAR

With --wandb_project, the losses are logged to Weights & Biases, and every --vis_every steps an image of the first
sample of the batch: the floorplan with the ground-truth and predicted correspondences and the position of the last
camera (ground truth and estimated from the predicted correspondences), next to the photo of that camera (data
with the camera_plan, camera_density and rgb columns, see train/lamar/prepare_lamar.py). The images are also saved
to <run>/vis/. With --val_data, the losses and diagnostics on the val data are logged after every epoch (val/ on
wandb, <run>/val_log.csv).

With torchrun (shells/train/train.sh), one process per GPU of the node (DistributedDataParallel): each GPU takes a
batch of 4 samples, and the points of the batches of all GPUs are the candidates of the feature matching loss.

    torchrun --standalone --nproc_per_node=8 train.py --data <data> --output <run> ...

With --init, the LoRA layers start from a trained checkpoint (e.g. jhcho99/SceneAligner) instead of zero, to
fine-tune it (with a lower --lr and fewer --epochs).
"""
import argparse
import csv
import glob
import os
import random
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, load_peft_weights, set_peft_model_state_dict
from PIL import Image
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from scenealigner.inference import SOFT_ARGMAX_TEMPERATURE, load_model, local_checkpoint, sample_descriptors
from scenealigner.training import (DensityFloorplanDataset, collate, denormalize, distributed_feature_matching_loss,
                                   feature_matching_loss, geometry_loss, regression_loss, topology_loss)
from scenealigner.visualization import draw_training_sample, estimate_camera_on_plan

LAMBDA_REGR, LAMBDA_TOPO, LAMBDA_GEO = 50.0, 10.0, 10.0  # the weight of the feature matching loss is 1


class FeatureExtractor(torch.nn.Module):
    """Patch features (B, C, S / 16, S / 16) of images (B, 3, S, S): the forward wrapped by DistributedDataParallel."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, img):
        tokens = self.model.forward_features(img)["x_norm_patchtokens"]
        grid_dim = img.shape[-1] // 16
        return tokens.permute(0, 2, 1).reshape(len(img), -1, grid_dim, grid_dim)


def world():
    """Rank of this process and number of processes (0, 1 without torchrun)."""
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def same_scene_mask(scene_ids, negatives):
    """same_scene (B, W * B) of feature_matching_loss: pairs of a local sample and another sample of the batches of
    all GPUs (rank-major) of one scene, whose points are not negatives (none with negatives="batch")."""
    rank, num_gpus = world()
    B = len(scene_ids)
    mask = np.zeros((B, num_gpus * B), dtype=bool)
    if negatives == "scene":
        all_ids = [scene_ids]
        if num_gpus > 1:
            all_ids = [None] * num_gpus
            dist.all_gather_object(all_ids, list(scene_ids))
        mask = np.array(scene_ids)[:, None] == np.concatenate(all_ids)[None]
        mask[np.arange(B), rank * B + np.arange(B)] = False
    return torch.tensor(mask).cuda()


def epoch_folders(root):
    """Data folders of the epochs: <root>/epoch_* (train/lamar/prepare_lamar.py: other windows at every epoch, taken
    in turn), else root itself (the same samples at every epoch)."""
    return sorted(glob.glob(os.path.join(root, "epoch_*"))) or [root]


def mean_over_gpus(values):
    """Mean over the GPUs of scalar tensors {name: tensor} (the same names on every GPU), for the logs."""
    if not dist.is_initialized():
        return values
    stacked = torch.stack([v.detach().float().cuda() for v in values.values()])
    dist.all_reduce(stacked)
    return dict(zip(values, stacked / dist.get_world_size()))


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

    model: FeatureExtractor, wrapped by DistributedDataParallel under torchrun (the points of all GPUs are negatives).

    negatives: "scene" (the points of other samples of the same scene are not negatives) or "batch" (all points of
    the batch are negatives, e.g. windows at different places of one building)."""
    density_img, plan_img = batch["density_img"].cuda(), batch["plan_img"].cuda()
    pts_density, pts_plan, valid = batch["density_pts"].cuda(), batch["plan_pts"].cuda(), batch["valid"].cuda()
    same_scene = same_scene_mask(batch["scene_id"], negatives)

    # One forward for both images (same resolution, see collate), as DistributedDataParallel expects
    map_density, map_plan = model(torch.cat([density_img, plan_img])).chunk(2)
    grid_dim = map_plan.shape[-1]
    desc_density = sample_descriptors(map_density, pts_density)
    matching_loss = distributed_feature_matching_loss if dist.is_initialized() else feature_matching_loss
    losses = {"feat": matching_loss(desc_density, sample_descriptors(map_plan, pts_plan), same_scene, valid)}
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


def validate(model, extractor, dataset, args, step, total_steps):
    """Mean losses and diagnostics of compute_loss on the val data, averaged over the GPUs. The augmentations of the
    data loader are seeded by --seed: the same at every epoch."""
    rank, num_gpus = world()
    sampler = DistributedSampler(dataset, shuffle=False) if num_gpus > 1 else None
    loader = DataLoader(dataset, batch_size=4, sampler=sampler, num_workers=4, generator=torch.Generator().manual_seed(args.seed),
                        collate_fn=partial(collate, resolutions=args.resolutions))
    totals = {}
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="val", disable=rank != 0):
            loss, losses, _ = compute_loss(extractor, batch, step, total_steps, args.negatives)
            for k, v in {"loss": loss, **losses}.items():
                totals[k] = totals.get(k, 0) + v.float()
    model.train()
    return mean_over_gpus({k: v / len(loader) for k, v in totals.items()})


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
    parser.add_argument("--data", required=True, help="training data (prepare_c3.py, or train/lamar/prepare_lamar.py: "
                        "<data>/epoch_<e> at epoch e)")
    parser.add_argument("--output", required=True, help="directory of the LoRA layers and the training log")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[512, 768, 1024, 1280], help="one per batch, at random")
    parser.add_argument("--min_points", type=int, default=10000, help="correspondences that a random crop must keep")
    parser.add_argument("--plan_grid", type=int, default=0, help="one correspondence per cell of this many floorplan px (0: all)")
    parser.add_argument("--negatives", default="scene", choices=["scene", "batch"],
                        help="batch: the points of other samples of the same scene are negatives too")
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--init", default=None, help="LoRA layers to start from (local directory or Hugging Face "
                        "repository, as load_model, e.g. jhcho99/SceneAligner); default: new LoRA layers")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", default=None, help="logs the losses and the visualizations to Weights & Biases")
    parser.add_argument("--wandb_name", default=None, help="name of the wandb run")
    parser.add_argument("--val_data", default=None, help="val data (prepare_lamar.py --sessions val --epochs 1), "
                        "evaluated after every epoch")
    parser.add_argument("--resume", action="store_true", help="continues the run of --output after its last epoch (state.pt)")
    parser.add_argument("--vis_every", type=int, default=500, help="steps between two visualizations (0: none)")
    args = parser.parse_args()
    # Resume after the last completed epoch (LoRA layers, optimizer, step, wandb run) only with --resume
    state_path = os.path.join(args.output, "state.pt")
    if os.path.exists(state_path) != args.resume:
        parser.error(f"--resume: no {state_path}" if args.resume else f"{state_path} exists: pass --resume or another --output")
    state = torch.load(state_path, map_location="cpu", weights_only=True) if args.resume else {}
    if "LOCAL_RANK" in os.environ:  # torchrun
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    rank, num_gpus = world()
    wandb = None
    if args.wandb_project and rank == 0:
        import wandb  # optional dependency

        init = dict(project=args.wandb_project, name=args.wandb_name, config=vars(args), id=state.get("wandb_id"), resume="allow")
        try:
            wandb.init(**init)
        except wandb.errors.CommError as error:  # the compute node can not reach the server: log locally, wandb sync later
            print(f"wandb: {error}\nwandb: offline run in {os.path.abspath('wandb')} (upload: wandb sync <run folder>)")
            wandb.init(**init, mode="offline")

    random.seed(args.seed + rank)  # different augmentations and dropout on each GPU
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    image_size = args.resolutions[0] if len(args.resolutions) == 1 else None
    datasets = [DensityFloorplanDataset(folder, min_points=args.min_points, image_size=image_size, plan_grid=args.plan_grid)
                for folder in epoch_folders(args.data)]

    def epoch_loader(epoch):
        """Data loader of an epoch (the dataset of its folder) and its sampler."""
        dataset = datasets[epoch % len(datasets)]
        sampler = DistributedSampler(dataset, seed=args.seed) if num_gpus > 1 else None  # same number of batches on every GPU
        return DataLoader(dataset, batch_size=4, shuffle=sampler is None, sampler=sampler, num_workers=4,
                          collate_fn=partial(collate, resolutions=args.resolutions)), sampler

    total_steps = sum(len(epoch_loader(epoch)[0]) for epoch in range(args.epochs))
    if rank == 0:
        print(f"{len(datasets)} data folders of {sum(len(d) for d in datasets)} samples, {total_steps} steps")
    val_dataset = DensityFloorplanDataset(epoch_folders(args.val_data)[0], min_points=args.min_points, image_size=image_size,
                                          plan_grid=args.plan_grid) if args.val_data else None

    lora = LoraConfig(r=16, lora_alpha=16, lora_dropout=0.1, target_modules=["qkv", "proj", "fc1", "fc2"])
    if rank:  # the first process downloads the DINOv3 weights
        dist.barrier()
    model = get_peft_model(load_model("pre-trained", args.dinov3_weights), lora)
    if args.init:  # fine-tuning: same LoRA configuration as the checkpoint (that of the C3 / Structured3D training)
        result = set_peft_model_state_dict(model, load_peft_weights(local_checkpoint(args.init)))
        mismatch = result.unexpected_keys + [k for k in result.missing_keys if "lora_" in k]
        assert not mismatch, f"--init {args.init}: keys not matching the LoRA layers {mismatch[:5]}"
    if rank == 0 and num_gpus > 1:
        dist.barrier()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    if rank == 0:
        os.makedirs(os.path.join(args.output, "vis"), exist_ok=True)
    first_epoch, step = 0, 0
    if state:
        set_peft_model_state_dict(model, load_peft_weights(args.output))
        optimizer.load_state_dict(state["optimizer"])
        first_epoch, step = state["epoch"] + 1, state["step"]
        if rank == 0:
            print(f"resumed after epoch {first_epoch} (step {step})")
    extractor = FeatureExtractor(model)
    if num_gpus > 1:
        extractor = DistributedDataParallel(extractor, device_ids=[torch.cuda.current_device()])
    columns = ("feat", "regr", "geo", "topo", "acc", "chance")
    if rank == 0:
        log = csv.writer(open(os.path.join(args.output, "log.csv"), "a" if first_epoch else "w", buffering=1))
        if not first_epoch:
            log.writerow(["step", "loss", *columns])
    for epoch in range(first_epoch, args.epochs):
        loader, sampler = epoch_loader(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", disable=rank != 0):
            optimizer.zero_grad()
            loss, losses, pred = compute_loss(extractor, batch, step, total_steps, args.negatives)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            if step % 100 == 0:
                logged = mean_over_gpus({"loss": loss, **losses})
                if rank == 0:
                    log.writerow([step, logged["loss"].item()] + [logged[k].item() if k in logged else "" for k in columns])
                    if wandb:
                        wandb.log({"epoch": epoch, **{k: v.item() for k, v in logged.items()}}, step=step)
            if args.vis_every and step % args.vis_every == 0 and rank == 0:
                image = visualize(batch, pred)
                Image.fromarray(image).save(os.path.join(args.output, "vis", f"{step:06d}.png"))
                if wandb:
                    wandb.log({"sample": wandb.Image(image, caption=f"step {step}: {batch['scene_id'][0]}")}, step=step)
        if rank == 0:
            model.save_pretrained(args.output)
            torch.save({"epoch": epoch, "step": step, "optimizer": optimizer.state_dict(),
                        "wandb_id": wandb.run.id if wandb else None}, state_path)
        if val_dataset is not None:
            val = validate(model, extractor, val_dataset, args, step, total_steps)
            if rank == 0:
                val_path = os.path.join(args.output, "val_log.csv")
                new = not os.path.exists(val_path)
                with open(val_path, "a") as f:
                    if new:
                        csv.writer(f).writerow(["epoch", "step", "loss", *columns])
                    csv.writer(f).writerow([epoch, step, val["loss"].item()] + [val[k].item() if k in val else "" for k in columns])
                print(f"val after epoch {epoch + 1}: " + ", ".join(f"{k} {v.item():.4f}" for k, v in val.items()))
                if wandb:
                    wandb.log({f"val/{k}": v.item() for k, v in val.items()}, step=step)
    if num_gpus > 1:
        dist.barrier()  # the first process saves the last epoch
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
