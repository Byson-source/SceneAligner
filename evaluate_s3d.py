"""Evaluation on the Structured3D test scenes: single-view camera localization (Table 3).

    python evaluate_s3d.py --s3d_root data/Structured3D
    python evaluate_s3d.py --s3d_root data/Structured3D --scenes 3200-3220 --output outputs/s3d_results_3200-3220.json

The test scenes are scene_03250 to scene_03499; --scenes FIRST-LAST evaluates another inclusive range of scenes
instead (scenes missing from --s3d_root are skipped).  The floorplans of the scenes and the pi3 reconstructions of the views are cached in --recon_cache and reused by
later runs.  Per view, --image_dir/<scene>/<room>_<position>.png shows the RGB image next to the floorplan with the
aligned density map and the estimated (green) and ground-truth (blue) cameras, each a dot with a 1 m arrow.
"""
import argparse
import json
import os

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from icecream import ic, install
install()

from scenealigner import structured3d as s3d
from scenealigner.evaluation import compute_pose_error_analytic, get_camera_center_on_density_px
from scenealigner.inference import (S3D_MODEL_REPO, align_to_floorplan, load_model, reconstruct_projection_geometry,
                                    render_density_map)
from scenealigner.preprocess import load_models, reconstruct

CACHED_KEYS = ("points_cam0", "conf", "extrinsics", "est_gravity_cam")  # what the alignment and the errors use


def pose_errors(recon, alignment, plan_params, view_dir):
    """Positional (m) and angular (degrees) errors of the camera of a view aligned to its floorplan."""
    eye, forward = s3d.load_camera(view_dir)[:2]
    center_px = get_camera_center_on_density_px(recon["extrinsics"], alignment["proj_params"])[0]
    t_err, r_err = compute_pose_error_analytic(alignment["M"], recon["extrinsics"][0], alignment["proj_params"], center_px,
                                               s3d.world_to_plan(eye, plan_params), forward[:2] * (1, -1))
    return float(t_err * s3d.MM_PER_PIXEL / plan_params[1] / 1000), r_err


def cameras_on_plan(recon, alignment, plan_params, view_dir):
    """Estimated and ground-truth camera of a view on the floorplan: (position (2,), viewing direction (2,)) each, in
    pixels, as compute_pose_error_analytic places them."""
    proj_params, M = alignment["proj_params"], alignment["M"]
    center_px = get_camera_center_on_density_px(recon["extrinsics"], proj_params)[0]
    forward_ground = proj_params["R_rect"] @ (recon["extrinsics"][0, :3, :3] @ np.array([0.0, 0.0, 1.0]))
    pred = (M @ np.array([*center_px, 1.0]), M[:2, :2] @ np.array([forward_ground[0], -forward_ground[2]]))
    eye, forward = s3d.load_camera(view_dir)[:2]
    return pred, (s3d.world_to_plan(eye, plan_params), forward[:2] * (1, -1))


def draw_result(view_dir, alignment, plan_img, plan_params, cameras, out_path):
    """RGB image next to the floorplan with the density map warped by M and the cameras: dot + 1 m arrow, estimate
    green, ground truth blue."""
    plan = np.array(plan_img)
    density = np.array(alignment["density_img"])
    canvas = cv2.addWeighted(plan, 0.6, cv2.warpAffine(density, alignment["M"], plan.shape[1::-1]), 0.4, 0)
    arrow_px = 1000.0 / s3d.MM_PER_PIXEL * plan_params[1]  # 1 m on the resized floorplan
    for (center, forward), color in zip(cameras, ((0, 255, 0), (0, 0, 255))):
        start = tuple(np.round(center).astype(int))
        if np.linalg.norm(forward) > 1e-6:  # looking straight up or down otherwise
            end = tuple(np.round(center + forward / np.linalg.norm(forward) * arrow_px).astype(int))
            cv2.arrowedLine(canvas, start, end, color, 2, tipLength=0.3, line_type=cv2.LINE_AA)
        cv2.circle(canvas, start, 4, color, -1, lineType=cv2.LINE_AA)
    rgb = Image.open(os.path.join(view_dir, "rgb_rawlight.png")).convert("RGB")
    rgb = rgb.resize((max(1, round(rgb.width * canvas.shape[0] / rgb.height)), canvas.shape[0]))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(np.hstack([np.array(rgb), canvas])).save(out_path)


def load_or_draw_floorplan(scene_dir, cache_dir):
    """Floorplan of a scene and its world_to_plan parameters, from the cache if there, else drawn and cached."""
    img_path, params_path = os.path.join(cache_dir, "floorplan.png"), os.path.join(cache_dir, "floorplan.json")
    if os.path.exists(params_path):  # written after the image
        with open(params_path) as f:
            origin, scale, offsets = json.load(f)
        return Image.open(img_path).convert("RGB"), (tuple(origin), scale, tuple(offsets))
    plan_img, plan_params = s3d.load_floorplan(scene_dir)
    os.makedirs(cache_dir, exist_ok=True)
    plan_img.save(img_path)  # lossless
    with open(params_path + ".tmp", "w") as f:
        json.dump(plan_params, f)
    os.replace(params_path + ".tmp", params_path)
    return plan_img, plan_params


def load_or_reconstruct(view_dir, cache_path, models):
    """pi3 reconstruction of a view, from the cache if there, else computed and cached. models: loaded on first use."""
    if os.path.exists(cache_path):
        return dict(np.load(cache_path))
    if not models:
        tqdm.write(f"{cache_path} not cached: loading pi3 and GeoCalib to reconstruct the views")
        models.extend(load_models(recon_model="pi3-original"))
    recon = reconstruct([os.path.join(view_dir, "rgb_rawlight.png")], *models)
    recon = {k: recon[k] for k in CACHED_KEYS}
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path + ".tmp.npz", **recon)
    os.replace(cache_path + ".tmp.npz", cache_path)  # no partial file if interrupted
    return recon


def compute_metrics(t_errs, r_errs):
    """Table 3: recalls (%) over the views."""
    t_errs, r_errs = np.array(t_errs), np.array(r_errs)
    metrics = {f"Pos@{thr}m": np.mean(t_errs < thr) for thr in (0.1, 0.5, 1.0)}
    metrics["Ang&Pos@(30,1m)"] = np.mean((t_errs < 1.0) & (r_errs < 30.0))
    return {k: float(v) * 100 for k, v in metrics.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3d_root", default="data/Structured3D", help="Structured3D test scenes: the directory of the scene_* folders")
    parser.add_argument("--checkpoint_dir", default=S3D_MODEL_REPO, help="LoRA layers")
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--recon_cache", default="outputs/s3d_reconstructions", help="cached floorplans and pi3 reconstructions")
    parser.add_argument("--output", default="outputs/s3d_results.json")
    parser.add_argument("--image_dir", default="outputs/s3d_views", help="per-view images of the cameras on the floorplan")
    parser.add_argument("--scenes", default=f"{s3d.TEST_SCENES[0]}-{s3d.TEST_SCENES[-1]}",
                        help="inclusive scene index range FIRST-LAST, e.g. 3200-3220 (default: the test scenes)")
    args = parser.parse_args()
    first, last = (int(v) for v in args.scenes.split("-"))

    np.random.seed(42)
    recon_models = []  # pi3 and GeoCalib, loaded only if a view is not cached
    model = load_model(args.checkpoint_dir, args.dinov3_weights)

    t_errs, r_errs, failed = [], [], 0
    for scene in tqdm(range(first, last + 1)):
        scene_id = f"scene_{scene:05d}"
        ic(scene_id)
        scene_dir = os.path.join(args.s3d_root, scene_id)
        if not os.path.isdir(os.path.join(scene_dir, "2D_rendering")):
            tqdm.write(f"{scene_id}: not in {args.s3d_root}, skipped")
            continue
        plan_img, plan_params = load_or_draw_floorplan(scene_dir, os.path.join(args.recon_cache, scene_id))
        for view_dir in s3d.list_views(scene_dir):
            
            room_id, position_id = view_dir.split(os.sep)[-4], view_dir.split(os.sep)[-1]
            cache_path = os.path.join(args.recon_cache, scene_id, f"{room_id}_{position_id}.npz")
            recon = load_or_reconstruct(view_dir, cache_path, recon_models)
            ic()
            try:
                alignment = align_to_floorplan(recon, plan_img, model)
            except ValueError:  # no similarity transform: the identity is used
                proj_params, points, _ = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"])
                alignment = {"proj_params": proj_params, "density_img": render_density_map(points, proj_params), "M": np.eye(2, 3)}
                failed += 1
            ic()    
            
            t_err, r_err = pose_errors(recon, alignment, plan_params, view_dir)
            t_errs.append(t_err)
            r_errs.append(r_err)
            draw_result(view_dir, alignment, plan_img, plan_params, cameras_on_plan(recon, alignment, plan_params, view_dir),
                        os.path.join(args.image_dir, scene_id, f"{room_id}_{position_id}.png"))

    metrics = compute_metrics(t_errs, r_errs)
    print(f"{len(t_errs)} views, {failed} without a similarity transform")
    print("  ".join(f"{k:>16}" for k in metrics))
    print("  ".join(f"{v:16.2f}" for v in metrics.values()))
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
