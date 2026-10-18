"""Evaluation on the Structured3D test scenes: single-view camera localization (Table 3).

    python evaluate_s3d.py --s3d_root <Structured3D>
"""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from scenealigner import structured3d as s3d
from scenealigner.evaluation import compute_pose_error_analytic, get_camera_center_on_density_px
from scenealigner.inference import S3D_MODEL_REPO, align_to_floorplan, load_model, reconstruct_projection_geometry
from scenealigner.preprocess import load_models, reconstruct


def pose_errors(recon, alignment, plan_params, view_dir):
    """Positional (m) and angular (degrees) errors of the camera of a view aligned to its floorplan."""
    eye, forward = s3d.load_camera(view_dir)[:2]
    center_px = get_camera_center_on_density_px(recon["extrinsics"], alignment["proj_params"])[0]
    t_err, r_err = compute_pose_error_analytic(alignment["M"], recon["extrinsics"][0], alignment["proj_params"], center_px,
                                               s3d.world_to_plan(eye, plan_params), forward[:2] * (1, -1))
    return float(t_err * s3d.MM_PER_PIXEL / plan_params[1] / 1000), r_err


def compute_metrics(t_errs, r_errs):
    """Table 3: recalls (%) over the views."""
    t_errs, r_errs = np.array(t_errs), np.array(r_errs)
    metrics = {f"Pos@{thr}m": np.mean(t_errs < thr) for thr in (0.1, 0.5, 1.0)}
    metrics["Ang&Pos@(30,1m)"] = np.mean((t_errs < 1.0) & (r_errs < 30.0))
    return {k: float(v) * 100 for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3d_root", required=True, help="Structured3D dataset: the directory of the scene_* folders")
    parser.add_argument("--checkpoint_dir", default=S3D_MODEL_REPO, help="LoRA layers")
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--output", default="outputs/s3d_results.json")
    args = parser.parse_args()

    np.random.seed(42)
    recon_model, geocalib = load_models(recon_model="pi3-original")
    model = load_model(args.checkpoint_dir, args.dinov3_weights)

    t_errs, r_errs, failed = [], [], 0
    for scene in tqdm(s3d.TEST_SCENES):
        scene_dir = os.path.join(args.s3d_root, f"scene_{scene:05d}")
        plan_img, plan_params = s3d.load_floorplan(scene_dir)
        for view_dir in s3d.list_views(scene_dir):
            recon = reconstruct([os.path.join(view_dir, "rgb_rawlight.png")], recon_model, geocalib)
            try:
                alignment = align_to_floorplan(recon, plan_img, model)
            except ValueError:  # no similarity transform: the identity is used
                proj_params = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"])[0]
                alignment = {"proj_params": proj_params, "M": np.eye(2, 3)}
                failed += 1
            t_err, r_err = pose_errors(recon, alignment, plan_params, view_dir)
            t_errs.append(t_err)
            r_errs.append(r_err)

    metrics = compute_metrics(t_errs, r_errs)
    print(f"{len(t_errs)} views, {failed} without a similarity transform")
    print("  ".join(f"{k:>16}" for k in metrics))
    print("  ".join(f"{v:16.2f}" for v in metrics.values()))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
