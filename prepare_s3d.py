"""Training data from Structured3D: density maps of single views and their correspondences with the floorplan.

Every view gives two samples: one from its pi3 reconstruction, one from its ground-truth depth map and camera.

    python prepare_s3d.py --s3d_root <Structured3D> --output <data>
"""
import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from scenealigner import structured3d as s3d
from scenealigner.evaluation import process_gt_correspondences
from scenealigner.inference import reconstruct_projection_geometry
from scenealigner.preprocess import load_models, reconstruct
from scenealigner.training import render_training_map

MIN_CORRESPONDENCES = 10


def depth_sample(world_points, plan_xy, plan_gray, size=512, padding=0.1):
    """Density map of the ground-truth points of a view, seen from above in the world frame (z up), and its
    correspondences: the pixels of the points that lie on walls (dark pixels) of the floorplan."""
    valid = ~np.isnan(world_points).any(axis=-1)
    points, plan_xy = world_points[valid], plan_xy[valid]

    # Height (lowest 20% and highest 5% removed) and x/y percentile filters
    keep = (points[:, 2] >= np.percentile(points[:, 2], 20.0)) & (points[:, 2] <= np.percentile(points[:, 2], 95.0))
    points, plan_xy = points[keep], plan_xy[keep]
    x, y = points[:, 0], points[:, 1]
    keep = (x >= np.percentile(x, 2.5)) & (x <= np.percentile(x, 97.5)) & (y >= np.percentile(y, 2.5)) & (y <= np.percentile(y, 97.5))
    x, y, plan_xy = x[keep], y[keep], plan_xy[keep]

    # Top-down projection, centered in the map with a margin
    width, height = x.max() - x.min(), y.max() - y.min()
    scale = (size * (1 - padding * 2)) / max(width, height)
    u = ((x - x.min()) * scale + (size - width * scale) / 2).astype(np.int32)
    v = size - 1 - ((y - y.min()) * scale + (size - height * scale) / 2).astype(np.int32)
    density_map = np.power(np.bincount(v * size + u, minlength=size * size).reshape(size, size).astype(np.float32), 0.5)
    density_img = Image.fromarray((255 - density_map / density_map.max() * 255.0).astype(np.uint8))

    inside = ((plan_xy >= 0) & (plan_xy < size)).all(axis=1)
    density_px, plan_px = np.stack([u, v], axis=1)[inside].astype(np.float32), plan_xy[inside].astype(np.float32)
    plan_idx = np.clip(plan_px.astype(int), 0, size - 1)
    on_wall = plan_gray[plan_idx[:, 1], plan_idx[:, 0]] < 50
    return density_img, density_px[on_wall], plan_px[on_wall]


def pi3_sample(recon, plan_xy, plan_gray):
    """Density map of the pi3 reconstruction of a view and its correspondences: the pixels of the reconstructed
    points whose ground-truth positions (plan_xy) lie on walls of the floorplan."""
    inside = ~np.isnan(plan_xy).any(axis=-1) & ((plan_xy >= 0) & (plan_xy < plan_gray.shape[0])).all(axis=-1)
    plan_idx = plan_xy[inside].astype(int)
    on_wall = inside.copy()
    on_wall[inside] = plan_gray[plan_idx[:, 1], plan_idx[:, 0]] < 50
    v, u = np.nonzero(on_wall)
    corr_data = {"tensor_coords": np.stack([u, v], axis=1).astype(np.float32), "plan_coords": plan_xy[on_wall].astype(np.float32)}

    proj_params, points, mask = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"])
    density_px, plan_px = process_gt_correspondences(corr_data, 0, recon["points_cam0"], proj_params, mask)
    return render_training_map(points, proj_params), density_px, plan_px


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s3d_root", required=True, help="Structured3D dataset: the directory of the scene_* folders")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenes", type=int, nargs=2, default=[s3d.TRAIN_SCENES.start, s3d.TRAIN_SCENES.stop], help="first train scene and the one after the last")
    args = parser.parse_args()

    recon_model, geocalib = load_models(recon_model="pi3-original")
    os.makedirs(os.path.join(args.output, "floorplan"), exist_ok=True)
    pairs = []
    for scene in tqdm(range(*args.scenes)):
        scene_id = f"scene_{scene:05d}"
        scene_dir = os.path.join(args.s3d_root, scene_id)
        # Scenes 1600 to 1799 have no perspective views
        if scene not in s3d.TRAIN_SCENES or scene in s3d.OFF_CANVAS_SCENES or not os.path.isdir(os.path.join(scene_dir, "2D_rendering")):
            continue
        plan_img, plan_params = s3d.load_floorplan(scene_dir)
        plan_path = os.path.join("floorplan", f"{scene_id}.png")
        plan_img.save(os.path.join(args.output, plan_path))
        plan_gray = np.array(plan_img.convert("L"))

        for view_dir in s3d.list_views(scene_dir):
            world_points = s3d.load_world_points(view_dir)
            plan_xy = s3d.world_to_plan(world_points, plan_params)
            recon = reconstruct([os.path.join(view_dir, "rgb_rawlight.png")], recon_model, geocalib)
            samples = {"depth": depth_sample(world_points, plan_xy, plan_gray), "pi3": pi3_sample(recon, plan_xy, plan_gray)}
            room_id, position_id = view_dir.split(os.sep)[-4], view_dir.split(os.sep)[-1]
            for source, sample in samples.items():
                if len(sample[1]) < MIN_CORRESPONDENCES:
                    continue
                density_img, density_px, plan_px = sample
                name = os.path.join(source, scene_id, f"{room_id}_{position_id}")
                os.makedirs(os.path.join(args.output, source, scene_id), exist_ok=True)
                density_img.save(os.path.join(args.output, f"{name}.png"))
                np.save(os.path.join(args.output, f"{name}.npy"), np.stack([plan_px, density_px]))
                pairs.append({"scene_id": scene_id, "density": f"{name}.png", "floorplan": plan_path, "correspondences": f"{name}.npy"})
    columns = ["scene_id", "density", "floorplan", "correspondences"]
    pd.DataFrame(pairs, columns=columns).to_csv(os.path.join(args.output, f"scenes_{args.scenes[0]}_{args.scenes[1]}.csv"), index=False)


if __name__ == "__main__":
    main()
