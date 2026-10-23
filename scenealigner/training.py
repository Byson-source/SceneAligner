"""Training data and losses of the LoRA layers."""
import glob
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .inference import norm_transform


def render_training_map(points, proj_params, gamma=0.5):
    """Density map as rendered for the training data. Its rounding differs from the one at inference: the points
    are assigned to pixels before the vertical flip, and the density is truncated before the inversion."""
    size = proj_params["density_size"]
    u = ((points[:, 0] - proj_params["min_x"]) * proj_params["scale"] + proj_params["cx"]).astype(np.int64)
    v = ((points[:, 2] - proj_params["min_z"]) * proj_params["scale"] + proj_params["cz"]).astype(np.int64)
    density_map = np.bincount((size - 1 - v) * size + u, minlength=size * size).reshape(size, size)
    density_map = np.power(density_map.astype(np.float32), gamma)
    return Image.fromarray(255 - (density_map / density_map.max() * 255.0).astype(np.uint8))


def rotate_points(pts, k, w, h):
    """Points (N, 2) of an image of size (w, h) after its rotation by k * 90 degrees (counter-clockwise)."""
    x, y = pts[:, 0], pts[:, 1]
    return np.stack([(x, y), (y, w - x), (w - x, h - y), (h - y, x)][k], axis=1)


def random_rotate(img, pts, k):
    """Rotates an image by k * 90 degrees (counter-clockwise) together with its points."""
    w, h = img.size
    return img.rotate(k * 90, expand=True), rotate_points(pts, k, w, h)


def parse_point(value):
    """A point stored as "x y" in a CSV column; NaN if the column is missing (C3 and Structured3D data)."""
    if isinstance(value, str):
        return np.array(value.split(), dtype=np.float32)
    return np.full(2, np.nan, dtype=np.float32)


class DensityFloorplanDataset(Dataset):
    """Pairs of a density map and a floorplan with their correspondences, listed in the CSV files of a folder."""

    def __init__(self, root, num_points=1024, min_points=10000, image_size=None, plan_grid=0):
        """plan_grid: if > 0, one correspondence per cell of plan_grid floorplan pixels, so that the positives of a
        sample are distinguishable on the floorplan (LaMAR: the points of a window lie in a few patches of the
        building plan); the free slots are random points, used as negatives only."""
        self.root, self.num_points, self.min_points, self.image_size = root, num_points, min_points, image_size
        self.plan_grid = plan_grid
        files = sorted(glob.glob(os.path.join(root, "*.csv")))
        self.data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        self.jitter_plan = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        self.jitter_density = T.ColorJitter(brightness=0.2, contrast=0.2)

    def __len__(self):
        return len(self.data)

    def random_crop(self, img, pts, other_pts, extra, crop_range=(0.7, 1.0)):
        """Random crop of an image, kept only if at least min_points correspondences stay inside.

        extra (2,): one more point of the image (the camera position, for visualization), NaN if cropped out.
        """
        w, h = img.size
        scale = random.uniform(*crop_range)
        crop_w, crop_h = int(w * scale), int(h * scale)
        left, top = random.randint(0, w - crop_w), random.randint(0, h - crop_h)
        cropped = pts - np.float32([left, top])
        inside = (cropped[:, 0] >= 0) & (cropped[:, 0] < crop_w) & (cropped[:, 1] >= 0) & (cropped[:, 1] < crop_h)
        if inside.sum() < self.min_points:
            return img, pts, other_pts, extra
        extra = extra - np.float32([left, top])
        if not (0 <= extra[0] < crop_w and 0 <= extra[1] < crop_h):
            extra = np.full(2, np.nan, dtype=np.float32)
        return img.crop((left, top, left + crop_w, top + crop_h)), cropped[inside], other_pts[inside], extra

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        density = Image.open(os.path.join(self.root, row.density)).convert("RGB")
        plan = Image.open(os.path.join(self.root, row.floorplan)).convert("RGB")
        pts_plan, pts_density = np.load(os.path.join(self.root, row.correspondences))  # (N, 2) pixels each
        # Camera position of the last frame on both images and its photo (optional columns, for visualization)
        camera_plan, camera_density = parse_point(row.get("camera_plan")), parse_point(row.get("camera_density"))
        rgb_path = row.rgb if isinstance(row.get("rgb"), str) else ""

        # Augmentation: independent rotations and crops of the two images, colour jitter
        k, (w, h) = np.random.randint(0, 4), density.size
        density, pts_density = random_rotate(density, pts_density, k)
        camera_density = rotate_points(camera_density[None], k, w, h)[0]
        k, (w, h) = np.random.randint(0, 4), plan.size
        plan, pts_plan = random_rotate(plan, pts_plan, k)
        camera_plan = rotate_points(camera_plan[None], k, w, h)[0]
        plan, pts_plan, pts_density, camera_plan = self.random_crop(plan, pts_plan, pts_density, camera_plan)
        density, pts_density, pts_plan, camera_density = self.random_crop(density, pts_density, pts_plan, camera_density)
        plan, density = self.jitter_plan(plan), self.jitter_density(density)
        if self.plan_grid:  # one random correspondence per floorplan cell
            perm = np.random.permutation(len(pts_plan))
            _, first = np.unique(np.floor(pts_plan[perm] / self.plan_grid), axis=0, return_index=True)
            pts_plan, pts_density = pts_plan[perm[first]], pts_density[perm[first]]

        # Pixels -> [-1, 1] (grid_sample), without duplicates
        camera_density = 2 * ((camera_density + 0.5) / np.float32(density.size)) - 1
        camera_plan = 2 * ((camera_plan + 0.5) / np.float32(plan.size)) - 1
        pts_density = np.clip(2 * ((pts_density + 0.5) / np.float32(density.size)) - 1, -1, 1)
        pts_plan = np.clip(2 * ((pts_plan + 0.5) / np.float32(plan.size)) - 1, -1, 1)
        if self.image_size:  # training at a single resolution
            density = density.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
            plan = plan.resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        _, unique = np.unique(np.round(np.hstack([pts_density, pts_plan]), decimals=3), axis=0, return_index=True)
        pts_density, pts_plan = pts_density[np.sort(unique)], pts_plan[np.sort(unique)]

        # num_points correspondences; missing ones are random points, used as negatives only
        n = len(pts_density)
        if n >= self.num_points:
            keep = np.random.choice(n, self.num_points, replace=False)
            pts_density, pts_plan = torch.tensor(pts_density[keep]), torch.tensor(pts_plan[keep])
        else:
            pts_density = torch.cat([torch.tensor(pts_density), torch.rand(self.num_points - n, 2) * 2 - 1])
            pts_plan = torch.cat([torch.tensor(pts_plan), torch.rand(self.num_points - n, 2) * 2 - 1])
        valid = torch.arange(self.num_points) < n
        return {"density_img": norm_transform(density), "plan_img": norm_transform(plan), "density_pts": pts_density,
                "plan_pts": pts_plan, "valid": valid, "scene_id": row.scene_id,
                "camera_density": torch.tensor(camera_density), "camera_plan": torch.tensor(camera_plan), "rgb_path": rgb_path}


def collate(batch, resolutions):
    """All images of a batch are resized to one resolution, drawn at random (mixed-resolution training)."""
    size = random.choice(resolutions)
    out = {k: torch.stack([sample[k] for sample in batch]) for k in ("density_pts", "plan_pts", "valid", "camera_density", "camera_plan")}
    for k in ("density_img", "plan_img"):  # the points are normalized: no rescaling
        out[k] = torch.cat([F.interpolate(sample[k][None], size=(size, size), mode="bicubic", antialias=True) for sample in batch])
    out["scene_id"] = [sample["scene_id"] for sample in batch]
    out["rgb_path"] = [sample["rgb_path"] for sample in batch]
    return out


def denormalize(img):
    """Image tensor (3, H, W) of norm_transform -> uint8 array (H, W, 3)."""
    mean, std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
    img = img.detach().float().cpu().permute(1, 2, 0) * std + mean
    return (img.clamp(0, 1) * 255).round().to(torch.uint8).numpy()


def feature_matching_loss(feat_a, feat_b, same_scene, valid, temperature=0.07):
    """Symmetric InfoNCE over all points of the batch (Eq. 2). feat_a, feat_b: (B, N, C).

    same_scene (B, B): pairs of different samples of one scene, whose points are not used as negatives.
    """
    B, N, C = feat_a.shape
    logits = torch.matmul(F.normalize(feat_a, dim=-1).reshape(B * N, C), F.normalize(feat_b, dim=-1).reshape(B * N, C).T) / temperature
    logits = logits.masked_fill(same_scene[:, None, :, None].expand(B, N, B, N).reshape(B * N, B * N), float("-inf"))
    idx = torch.nonzero(valid.reshape(B * N)).squeeze(1)
    return (F.cross_entropy(logits[idx], idx) + F.cross_entropy(logits.T[idx], idx)) / 2


def distributed_feature_matching_loss(feat_a, feat_b, same_scene, valid, temperature=0.07):
    """feature_matching_loss over the points of the samples of all GPUs (DistributedDataParallel): the local points
    are the queries, the points of every GPU (all_gather, with gradients) the candidates. feat_a, feat_b: (B, N, C).

    same_scene (B, W * B): pairs of a local sample and a sample of the global batch (rank-major) of one scene, other
    than itself, whose points are not used as negatives. The gradients averaged by DDP are those of the mean loss.
    """
    import torch.distributed as dist
    import torch.distributed.nn.functional as dist_nn

    B, N, C = feat_a.shape
    a, b = F.normalize(feat_a, dim=-1).reshape(B * N, C), F.normalize(feat_b, dim=-1).reshape(B * N, C)
    all_a, all_b = torch.cat(dist_nn.all_gather(a)), torch.cat(dist_nn.all_gather(b))
    mask = same_scene[:, None, :, None].expand(B, N, same_scene.shape[1], N).reshape(B * N, -1)
    logits_ab = (a @ all_b.T / temperature).masked_fill(mask, float("-inf"))  # density -> floorplan points
    logits_ba = (b @ all_a.T / temperature).masked_fill(mask, float("-inf"))  # floorplan -> density points
    idx = torch.nonzero(valid.reshape(B * N)).squeeze(1)
    target = idx + dist.get_rank() * B * N  # the same point in the global batch
    return (F.cross_entropy(logits_ab[idx], target) + F.cross_entropy(logits_ba[idx], target)) / 2


def regression_loss(pred, gt, weights, delta=0.02):
    """Confidence-weighted Huber loss on the predicted floorplan coordinates (Eq. 3)."""
    dist = torch.norm(pred - gt, p=2, dim=-1)
    return (F.huber_loss(dist, torch.zeros_like(dist), delta=delta, reduction="none") * weights).sum() / (weights.sum() + 1e-6)


def sample_groups(B, N, size, num_groups, device):
    """Random groups of `size` distinct points per sample: indices (B, K, size)."""
    K = min(num_groups, N // size)
    return torch.argsort(torch.rand(B, N, device=device), dim=1)[:, :size * K].view(B, K, size)


def geometry_loss(src, pred, weights, num_pairs=512, delta=0.02, eps=1e-6):
    """Consistent distance ratios between point pairs of the density map and their predictions (Eq. 5)."""
    idx = sample_groups(*src.shape[:2], 2, num_pairs, src.device)
    batch = torch.arange(len(src), device=src.device).unsqueeze(1)
    i, j = idx[:, :, 0], idx[:, :, 1]
    w = weights[batch, i] * weights[batch, j]
    ds = torch.norm(src[batch, i] - src[batch, j], dim=-1).clamp(min=eps)
    dp = torch.norm(pred[batch, i] - pred[batch, j], dim=-1).clamp(min=eps)
    diff = torch.log(dp) - torch.log(ds)
    mean = (diff * w).sum(dim=1, keepdim=True) / (w.sum(dim=1, keepdim=True) + eps)
    return (F.huber_loss(diff, mean.detach().expand_as(diff), delta=delta, reduction="none") * w).sum() / (w.sum() + eps)


def topology_loss(src, pred, weights, num_triplets=512, delta=0.02, eps=1e-6):
    """Consistent angles (cosine and sine) of point triplets of the density map and their predictions (Eq. 4)."""
    idx = sample_groups(*src.shape[:2], 3, num_triplets, src.device)
    batch = torch.arange(len(src), device=src.device).unsqueeze(1)
    i, j, k = idx[:, :, 0], idx[:, :, 1], idx[:, :, 2]
    w = weights[batch, i] * weights[batch, j] * weights[batch, k]
    u_src, v_src = src[batch, j] - src[batch, i], src[batch, k] - src[batch, i]
    u_pred, v_pred = pred[batch, j] - pred[batch, i], pred[batch, k] - pred[batch, i]

    def sine(u, v):
        u, v = F.normalize(u, dim=-1), F.normalize(v, dim=-1)
        return u[:, :, 0] * v[:, :, 1] - u[:, :, 1] * v[:, :, 0]

    cos_src, cos_pred = F.cosine_similarity(u_src, v_src, dim=-1), F.cosine_similarity(u_pred, v_pred, dim=-1)
    loss = F.huber_loss(cos_pred, cos_src.detach(), delta=delta, reduction="none") + \
        F.huber_loss(sine(u_pred, v_pred), sine(u_src, v_src).detach(), delta=delta, reduction="none")
    return (loss * w).sum() / (w.sum() + eps)
