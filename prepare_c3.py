"""Training data from C3: density maps of reconstructed photos and their correspondences with the floorplan.

    python prepare_c3.py --c3_root <C3> --train_csv <cache>/train.csv --output <data>                 # pi3, photo collections
    python prepare_c3.py --c3_root <C3> --train_csv <cache>/train.csv --output <data> --model vggt    # VGGT, photo collections
    python prepare_c3.py --c3_root <C3> --train_csv <cache>/train.csv --output <data> --single_view   # pi3, single photos
"""
import argparse
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from scenealigner.evaluation import load_correspondences, process_gt_correspondences
from scenealigner.inference import reconstruct_projection_geometry
from scenealigner.preprocess import GEOCALIB_SEED, estimate_gravity, load_and_resize_plan, load_models, reconstruct
from scenealigner.training import render_training_map

GAMMAS = (0.25, 0.5, 0.75, 1.0)
XZ_SLICES = (0.0, 2.5, 5.0)  # x/z percentile filters


def cached_gravity(path, photo_path, geocalib):
    """Gravity direction of a photo, estimated once for all runs."""
    if not os.path.exists(path):
        torch.manual_seed(GEOCALIB_SEED)
        tmp_path = f"{path}.{os.getpid()}.npy"
        np.save(tmp_path, estimate_gravity(photo_path, geocalib))
        os.replace(tmp_path, path)
    return np.load(path)


def find_correspondences(c3_root, uid):
    """C3 keeps the annotations of our train split in its train and val folders."""
    for split in ("train", "val"):
        path = os.path.join(c3_root, "geometric", split, "correspondences", str(uid // 1000), f"{uid:06d}.npy")
        if os.path.exists(path):
            return path


def is_consistent(density_px, plan_px, single_view):
    """The correspondences have to agree with a similarity transform (RANSAC). Those of a single photo also need
    40 inliers, 30% of inliers and a scale within [0.2, 3]."""
    M, inliers = cv2.estimateAffinePartial2D(density_px, plan_px, method=cv2.RANSAC, ransacReprojThreshold=10.0)
    if M is None or not single_view:
        return M is not None
    return 0.2 <= np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2) <= 3.0 and inliers.sum() >= 40 and inliers.mean() >= 0.3


def generate(recon, batch_corrs, conf_threshold, single_view, output, name):
    """Saves the density maps of a reconstruction with their correspondences. Returns their paths."""
    saved = []
    for xz_slice_val in ((2.5,) if single_view else XZ_SLICES):
        proj_params, points, mask = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"],
                                                                    conf_threshold, xz_slice_val)
        # Annotated pixels whose 3D points are in the density map; photos with few of them are left out
        pairs = [process_gt_correspondences(corr_data, s_idx, recon["points_cam0"], proj_params, mask)
                 for s_idx, corr_data in enumerate(batch_corrs) if corr_data is not None]
        pairs = [pair for pair in pairs if len(pair[0]) >= (200 if single_view else 8)]
        if not pairs:
            continue
        density_px, plan_px = np.concatenate([pair[0] for pair in pairs]), np.concatenate([pair[1] for pair in pairs])
        if not is_consistent(density_px, plan_px, single_view):
            continue
        corr_path = f"{name}_xz{xz_slice_val}.npy"
        np.save(os.path.join(output, corr_path), np.stack([plan_px, density_px]))
        for gamma in ((0.5,) if single_view else GAMMAS):
            density_path = f"{name}_xz{xz_slice_val}_g{gamma}.png"
            render_training_map(points, proj_params, gamma).save(os.path.join(output, density_path))
            saved.append({"density": density_path, "correspondences": corr_path})
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c3_root", required=True, help="C3 dataset, with visual/ and geometric/")
    parser.add_argument("--train_csv", required=True, help="train.csv of our dataset repository")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="pi3", choices=["pi3", "vggt"], help="3D reconstruction model")
    parser.add_argument("--single_view", action="store_true", help="reconstructs every photo alone")
    parser.add_argument("--conf", type=int, nargs="+", default=[30, 45, 60, 75, 90], help="confidence thresholds (percentiles)")
    parser.add_argument("--recon_checkpoint", default=None, help="local copy of the pi3 or VGGT weights")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    recon_model, geocalib = load_models(args.recon_checkpoint, args.model)
    data = pd.read_csv(args.train_csv)
    for folder in ("floorplan", "gravity"):
        os.makedirs(os.path.join(args.output, folder), exist_ok=True)

    # One run per confidence threshold and photo preprocessing mode: every scene is reconstructed again
    for conf_threshold in args.conf:
        for mode in (("pad",) if args.single_view else ("crop", "pad")):
            run = f"{args.model}{'_single-view' if args.single_view else ''}_conf{conf_threshold}_{mode}"
            os.makedirs(os.path.join(args.output, run), exist_ok=True)
            random.seed(f"{args.seed} {run}")
            pairs = []
            for scene_id, rows in tqdm(data.groupby("scene_id", sort=False), desc=run):
                visual_dir = os.path.join(args.c3_root, "visual", rows.scene_name.iloc[0])
                plan_img, plan_transform_params = load_and_resize_plan(os.path.join(visual_dir, rows.plan_path.iloc[0]))
                plan_path = os.path.join("floorplan", f"{scene_id}.png")
                if not os.path.exists(os.path.join(args.output, plan_path)):
                    plan_img.save(os.path.join(args.output, plan_path))

                if args.single_view:
                    subsets = [rows.iloc[[i]] for i in range(len(rows))]
                else:  # a random limit of 50 to 150 photos per reconstruction
                    max_photos = random.randint(50, 150)
                    subsets = [rows if len(rows) <= max_photos else
                               rows.iloc[random.sample(range(len(rows)), max_photos)].sort_values("photo_path")]
                for photos in subsets:
                    photo_paths = [os.path.join(visual_dir, p) for p in photos.photo_path]
                    gravities = [cached_gravity(os.path.join(args.output, "gravity", f"{uid:06d}.npy"), photo_path, geocalib)
                                 for uid, photo_path in zip(photos.uid, photo_paths)]
                    recon = reconstruct(photo_paths, recon_model, geocalib, mode, gravities)
                    batch_corrs = [load_correspondences(find_correspondences(args.c3_root, uid), photo_path, plan_transform_params, mode)
                                   for uid, photo_path in zip(photos.uid, photo_paths)]
                    name = os.path.join(run, scene_id + (f"_{photos.uid.iloc[0]:06d}" if args.single_view else ""))
                    for saved in generate(recon, batch_corrs, conf_threshold, args.single_view, args.output, name):
                        pairs.append({"scene_id": scene_id, "floorplan": plan_path, **saved})
            columns = ["scene_id", "density", "floorplan", "correspondences"]
            pd.DataFrame(pairs, columns=columns).to_csv(os.path.join(args.output, f"{run}.csv"), index=False)


if __name__ == "__main__":
    main()
