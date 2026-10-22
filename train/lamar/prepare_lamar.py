"""Training data from LaMAR HGE: density maps of windows of consecutive frames and their correspondences with the
floorplan. Every window gives samples of its MapAnything reconstruction (ground-truth intrinsics and relative
poses as input, as prepare_c3.py varies the confidence threshold, the x/z slice and the gamma) and samples of its
LiDAR depth maps unprojected with the ground-truth poses (as the depth samples of prepare_s3d.py).

    python train/lamar/prepare_lamar.py --map_dir <HGE/sessions/map> --raw_dir <raw sessions> --floorplan hge.png --output <data>

Correspondences: the pixels of the density map of a 3D point and the floorplan pixel of that point, placed in the
world with the ground-truth pose of the first camera and the affine transform of the floorplan (lamar.py); only
points on walls of the floorplan are kept. With --plan_source recon (default), the points are those of the
reconstruction, gated by the LiDAR depth where it exists; with --plan_source lidar, the LiDAR points (as
prepare_s3d.py). The CSV rows also hold the photo and the position of the last camera, for the visualizations of
train.py.
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lamar import (SEQ_LENGTHS, Session, filter_ground_points, ground_rotation, iter_train_windows,  # noqa: E402
                   lidar_points_cam0, load_floorplan, load_session, load_split, on_wall, top_down_projection,
                   train_frames, world_to_plan)
from reconstruction import load_mapanything, load_recon, reconstruct, save_recon  # noqa: E402
from scenealigner.evaluation import get_camera_center_on_density_px, process_gt_correspondences  # noqa: E402
from scenealigner.inference import get_density_coordinates, reconstruct_projection_geometry  # noqa: E402
from scenealigner.training import render_training_map  # noqa: E402

GAMMAS = (0.25, 0.5, 0.75, 1.0)
XZ_SLICES = (0.0, 2.5, 5.0)  # x/z percentile filters
DENSITY_SIZE = 512  # of the density maps, as for C3 (the floorplan is larger: the points are normalized in training)
MIN_VIEW_CORRESPONDENCES = 8  # as prepare_c3.py
RANSAC_THRESHOLD = 10.0  # px of a 512 px floorplan (prepare_c3.is_consistent)


def is_consistent(density_px: np.ndarray, plan_px: np.ndarray, plan_size: int) -> bool:
    """
    The correspondences agree with a similarity transform without reflection (RANSAC, prepare_c3.is_consistent).
    Args:
        density_px (np.ndarray): tensor with shape [N, 2]
        plan_px (np.ndarray): tensor with shape [N, 2]
        plan_size (int): floorplan resolution, which scales the inlier threshold

    Returns:
        bool: consistent
    """
    M, _ = cv2.estimateAffinePartial2D(density_px, plan_px, method=cv2.RANSAC,
                                       ransacReprojThreshold=RANSAC_THRESHOLD * plan_size / 512)
    return M is not None and np.linalg.det(M[:2, :2]) > 0


def world_points(points_cam0: np.ndarray, c2w0: np.ndarray) -> np.ndarray:
    """
    Points of the frame of camera 0 -> world.
    Args:
        points_cam0 (np.ndarray): tensor with shape [..., 3]
        c2w0 (np.ndarray): camera-to-world of camera 0 with shape [4, 4]

    Returns:
        np.ndarray: tensor with shape [..., 3]
    """
    return points_cam0 @ c2w0[:3, :3].T + c2w0[:3, 3]


def lidar_gate(recon: dict, lidar_cam0: np.ndarray, rel_tol: float = 0.1, abs_tol: float = 0.3) -> np.ndarray:
    """
    Reconstructed points that agree with the LiDAR where it is valid: within max(abs_tol, rel_tol * range).
    Args:
        recon (dict): of reconstruction.reconstruct
        lidar_cam0 (np.ndarray): tensor with shape [S, H, W, 3], LiDAR points in camera 0 (NaN where missing)
        rel_tol (float): tolerance relative to the distance of the LiDAR point from its camera
        abs_tol (float): tolerance [m]

    Returns:
        np.ndarray: boolean tensor with shape [S, H, W], True where kept
    """
    centres = recon["extrinsics"][:, None, None, :3, 3]  # [S, 1, 1, 3] camera centres in camera 0
    ranges = np.linalg.norm(lidar_cam0 - centres, axis=-1)
    diff = np.linalg.norm(recon["points_cam0"] - lidar_cam0, axis=-1)
    has_lidar = np.isfinite(lidar_cam0).all(axis=-1)
    return ~has_lidar | (diff <= np.maximum(abs_tol, rel_tol * ranges))


def wall_correspondences(world: np.ndarray, plan_params: dict, max_points: int, rng: np.random.Generator) -> list:
    """
    Per view, the pixels whose 3D points lie on a wall of the floorplan and the floorplan pixels of these points.
    Args:
        world (np.ndarray): tensor with shape [S, H, W, 3], world points of the pixels (NaN where unused)
        plan_params (dict): of lamar.load_floorplan
        max_points (int): at most this many correspondences in the window, drawn at random
        rng (np.random.Generator): random generator

    Returns:
        list: per view, {"tensor_coords": [N, 2] (u, v) pixels, "plan_coords": [N, 2] floorplan pixels} or None
    """
    S, H, W, _ = world.shape
    flat = world.reshape(-1, 3)
    wall = on_wall(flat, plan_params)
    idx = np.nonzero(wall)[0]
    if len(idx) > max_points:
        idx = np.sort(rng.choice(idx, max_points, replace=False))
    plan_px = world_to_plan(flat[idx], plan_params).astype(np.float32)
    view, pixel = np.divmod(idx, H * W)
    v, u = np.divmod(pixel, W)
    corrs = []
    for s in range(S):
        sel = view == s
        if sel.sum() < MIN_VIEW_CORRESPONDENCES:
            corrs.append(None)
        else:
            corrs.append({"tensor_coords": np.stack([u[sel], v[sel]], axis=1).astype(np.float32), "plan_coords": plan_px[sel]})
    return corrs


def mapanything_samples(recon: dict, corrs: list, conf_thresholds: list, plan_params: dict, output: str, name: str,
                        min_correspondences: int) -> list:
    """
    Saves the density maps of a reconstruction (one per confidence threshold, x/z slice and gamma) with their
    correspondences, as prepare_c3.generate. Returns their rows.
    Args:
        recon (dict): of reconstruction.reconstruct
        corrs (list): of wall_correspondences
        conf_thresholds (list): confidence percentiles
        plan_params (dict): of lamar.load_floorplan
        output (str): data directory
        name (str): file stem of the window, relative to output
        min_correspondences (int): samples with fewer correspondences are not saved

    Returns:
        list: rows {"density", "correspondences", "camera_density"}
    """
    rows = []
    for conf in conf_thresholds:
        for xz in XZ_SLICES:
            proj_params, points, mask = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"],
                                                                        conf, xz, density_size=DENSITY_SIZE)
            pairs = [process_gt_correspondences(corr, s, recon["points_cam0"], proj_params, mask)
                     for s, corr in enumerate(corrs) if corr is not None]
            pairs = [pair for pair in pairs if len(pair[0]) >= MIN_VIEW_CORRESPONDENCES]
            if not pairs:
                continue
            density_px, plan_px = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
            if len(density_px) < min_correspondences or not is_consistent(density_px, plan_px, plan_params["plan_size"]):
                continue
            corr_path = f"{name}_conf{conf}_xz{xz}.npy"
            np.save(os.path.join(output, corr_path), np.stack([plan_px, density_px]))
            camera = get_camera_center_on_density_px(recon["extrinsics"], proj_params)[-1]
            for gamma in GAMMAS:
                density_path = f"{name}_conf{conf}_xz{xz}_g{gamma}.png"
                render_training_map(points, proj_params, gamma).save(os.path.join(output, density_path))
                rows.append({"density": density_path, "correspondences": corr_path, "camera_density": camera})
    return rows


def lidar_samples(lidar_cam0: np.ndarray, recon: dict, plan_params: dict, c2w0: np.ndarray, output: str, name: str,
                  min_correspondences: int) -> list:
    """
    Saves the density maps of the LiDAR points of a window (one per gamma), seen from above in the gravity-aligned
    frame of camera 0, with their correspondences (the points on walls). Returns their rows.
    Args:
        lidar_cam0 (np.ndarray): tensor with shape [S, H, W, 3], NaN where missing
        recon (dict): of reconstruction.reconstruct (gravity and extrinsics)
        plan_params (dict): of lamar.load_floorplan
        c2w0 (np.ndarray): camera-to-world of camera 0 with shape [4, 4]
        output (str): data directory
        name (str): file stem of the window, relative to output
        min_correspondences (int): samples with fewer correspondences are not saved

    Returns:
        list: rows {"density", "correspondences", "camera_density"}
    """
    points = lidar_cam0.reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < min_correspondences:
        return []
    R_rect = ground_rotation(recon["est_gravity_cam"])
    ground = points @ R_rect.T
    keep = filter_ground_points(ground)
    ground, points = ground[keep], points[keep]
    proj_params = top_down_projection(ground, R_rect, DENSITY_SIZE)
    wall = on_wall(world_points(points, c2w0), plan_params)
    density_px = get_density_coordinates(ground[wall], proj_params)
    plan_px = world_to_plan(world_points(points[wall], c2w0), plan_params).astype(np.float32)
    if len(density_px) < min_correspondences or not is_consistent(density_px, plan_px, plan_params["plan_size"]):
        return []
    corr_path = f"{name}.npy"
    np.save(os.path.join(output, corr_path), np.stack([plan_px, density_px]))
    camera = get_density_coordinates(recon["extrinsics"][-1:, :3, 3] @ R_rect.T, proj_params)[0]
    rows = []
    for gamma in GAMMAS:
        density_path = f"{name}_g{gamma}.png"
        render_training_map(ground, proj_params, gamma).save(os.path.join(output, density_path))
        rows.append({"density": density_path, "correspondences": corr_path, "camera_density": camera})
    return rows


def window_recon(frames: Session, model, cache_dir: str | None, name: str) -> dict:
    """
    Reconstruction of a window, from the cache if it is there.
    Args:
        frames (Session): the frames of the window
        model: MapAnything
        cache_dir (str | None): directory of the cached reconstructions
        name (str): window name

    Returns:
        dict: of reconstruction.reconstruct
    """
    path = os.path.join(cache_dir, f"{name}.npz") if cache_dir else None
    if path and os.path.exists(path):
        return load_recon(path)
    recon = reconstruct(frames.image_paths, frames.K, frames.c2w, model)
    if path:
        save_recon(path, recon)
    return recon


def scene_id_of(frames: Session, mode: str, cell_m: float) -> str:
    """
    scene_id of a window: samples of one scene_id are not negatives of each other in the feature matching loss.
    Args:
        frames (Session): the frames of the window
        mode (str): "hge" (one scene: only the points of a sample are its negatives) or "cell" (windows of
            different cells of cell_m metres are negatives of each other)
        cell_m (float): cell size [m]

    Returns:
        str: scene_id
    """
    if mode == "hge":
        return "hge"
    cx, cy = np.floor(frames.c2w[:, :2, 3].mean(axis=0) / cell_m).astype(int)
    return f"hge_{cx}_{cy}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map_dir", required=True, help="HGE/sessions/map")
    parser.add_argument("--raw_dir", default=None, help="directory of the raw phone sessions (LiDAR depth); none: no LiDAR")
    parser.add_argument("--floorplan", required=True, help="hge.png")
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.yaml"))
    parser.add_argument("--sessions", nargs="+", default=["train"], help="splits (train, val, test) or session names")
    parser.add_argument("--plan_size", type=int, default=1024, help="resolution of the floorplan")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(SEQ_LENGTHS), help="window lengths")
    parser.add_argument("--stride", type=float, default=0.5, help="stride between windows, as a fraction of their length")
    parser.add_argument("--windows_per_session", type=int, default=None, help="random subset of the windows of a session")
    parser.add_argument("--conf", type=int, nargs="+", default=[45, 75], help="confidence thresholds (percentiles)")
    parser.add_argument("--plan_source", default="recon", choices=["recon", "lidar"], help="3D points of the correspondences")
    parser.add_argument("--no_lidar_gate", action="store_true", help="keeps reconstructed points that disagree with the LiDAR")
    parser.add_argument("--no_lidar_samples", action="store_true", help="no density maps of the LiDAR points")
    parser.add_argument("--scene_id", default="hge", choices=["hge", "cell"])
    parser.add_argument("--cell_m", type=float, default=20.0)
    parser.add_argument("--max_points", type=int, default=20000, help="correspondences kept per window")
    parser.add_argument("--min_correspondences", type=int, default=200, help="samples with fewer are dropped")
    parser.add_argument("--mapanything_model", default="facebook/map-anything")
    parser.add_argument("--recon_cache", default=None, help="directory of cached reconstructions (read and written)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    plan_img, plan_params = load_floorplan(args.floorplan, args.plan_size)
    for folder in ("floorplan", "mapanything", "lidar"):
        os.makedirs(os.path.join(args.output, folder), exist_ok=True)
    if args.recon_cache:
        os.makedirs(args.recon_cache, exist_ok=True)
    plan_path = os.path.join("floorplan", "hge.png")
    plan_img.save(os.path.join(args.output, plan_path))
    with open(os.path.join(args.output, "floorplan", "hge_params.json"), "w") as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in plan_params.items() if k != "wall_dist_px"}, f, indent=2)

    split = load_split(args.split)
    sessions = [name for key in args.sessions for name in (split[key] if key in split else [key])]
    model = load_mapanything(args.mapanything_model)
    rows = {"mapanything": [], "lidar": []}
    for session_name in sessions:
        session = load_session(args.map_dir, args.raw_dir, session_name)
        kept = session.subset(train_frames(session, plan_params))
        windows = iter_train_windows(kept.timestamps, tuple(args.lengths), args.stride)
        if args.windows_per_session and len(windows) > args.windows_per_session:
            windows = [windows[i] for i in sorted(rng.choice(len(windows), args.windows_per_session, replace=False))]
        for start, length in tqdm(windows, desc=session_name):
            frames = kept.subset(np.arange(start, start + length))
            name = f"{session_name}_s{start:05d}_L{length}"
            recon = window_recon(frames, model, args.recon_cache, name)
            H, W = recon["points_cam0"].shape[1:3]
            lidar = None
            if frames.K_depth is not None and any(p is not None for p in frames.depth_paths):
                lidar = lidar_points_cam0(frames.depth_paths, frames.K_depth, recon["intrinsics"][0], (H, W),
                                          recon["extrinsics"], frames.T_depth_to_cam)

            # Correspondences: world points of the pixels, NaN where unused
            if args.plan_source == "lidar":
                if lidar is None:
                    continue
                world = world_points(lidar, frames.c2w[0])
            else:
                world = world_points(recon["points_cam0"], frames.c2w[0])
                unused = ~recon["mask"]
                if lidar is not None and not args.no_lidar_gate:
                    unused |= ~lidar_gate(recon, lidar)
                world[unused] = np.nan
            corrs = wall_correspondences(world, plan_params, args.max_points, rng)

            common = {"scene_id": scene_id_of(frames, args.scene_id, args.cell_m), "floorplan": plan_path,
                      "rgb": os.path.abspath(frames.image_paths[-1]),
                      "camera_plan": world_to_plan(frames.c2w[-1, :3, 3], plan_params)}
            samples = mapanything_samples(recon, corrs, args.conf, plan_params, args.output, os.path.join("mapanything", name),
                                          args.min_correspondences)
            if lidar is not None and not args.no_lidar_samples:
                samples += lidar_samples(lidar, recon, plan_params, frames.c2w[0], args.output, os.path.join("lidar", name),
                                         args.min_correspondences)
            for sample in samples:
                sample.update(common)
                sample["camera_plan"] = "{:.2f} {:.2f}".format(*common["camera_plan"])
                sample["camera_density"] = "{:.2f} {:.2f}".format(*sample["camera_density"])
                rows[sample["density"].split(os.sep)[0]].append(sample)

    columns = ["scene_id", "density", "floorplan", "correspondences", "rgb", "camera_plan", "camera_density"]
    for source, source_rows in rows.items():
        pd.DataFrame(source_rows, columns=columns).to_csv(os.path.join(args.output, f"lamar_{source}.csv"), index=False)
        print(f"{source}: {len(source_rows)} samples")


if __name__ == "__main__":
    main()
