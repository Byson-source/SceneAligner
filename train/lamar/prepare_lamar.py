"""Training data from LaMAR HGE, as the training data of SceneAligner (upstream prepare_c3.py and prepare_s3d.py).
Every window of frames gives samples of its MapAnything reconstruction (ground-truth intrinsics and relative
poses as input), varied as prepare_c3.py does (confidence thresholds 30 to 90, three x/z slices, four gammas),
and one sample of its LiDAR depth maps, as prepare_s3d.py makes one of the ground-truth depth map of each view.

    python train/lamar/prepare_lamar.py --map_dir <HGE/sessions/map> --raw_dir <raw sessions> --floorplan hge.png --output <data>
    # on several GPUs: one process per GPU with --shard i --num_shards n (shells/lamar/prepare.sh), then
    python train/lamar/prepare_lamar.py --merge --output <data>

Windows: 15 to 100 frames, consecutive or every 2nd or 4th frame (C3 reconstructs unordered photo collections).
Correspondences, as prepare_s3d.py: the pixels of the density map of the 3D points whose positions lie on walls of
the floorplan (within --max_wall_dist 0.5 m of a wall pixel of hge.png: the affine transform is accurate to about
0.25 m), and the floorplan pixels of these positions. With --plan_source recon (default) the positions are those
of the reconstructed points placed with the ground-truth pose of the first camera; with --plan_source lidar,
those of the LiDAR points (prepare_s3d.py takes the ground-truth depth). The CSV rows also hold the photo and the
position of the last camera, for the visualizations of train.py.
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
from lamar import (SEQ_LENGTHS, TRAIN_STEPS, Session, filter_ground_points, ground_rotation, iter_train_windows,  # noqa: E402
                   lidar_points_cam0, load_floorplan, load_session, load_split, on_plan, top_down_projection,
                   train_frames, window_name, world_to_plan)
from reconstruction import load_mapanything, load_recon, reconstruct, save_recon  # noqa: E402
from scenealigner.evaluation import get_camera_center_on_density_px, process_gt_correspondences  # noqa: E402
from scenealigner.inference import get_density_coordinates, reconstruct_projection_geometry  # noqa: E402
from scenealigner.training import render_training_map  # noqa: E402

GAMMAS = (0.25, 0.5, 0.75, 1.0)  # prepare_c3.py
XZ_SLICES = (0.0, 2.5, 5.0)  # x/z percentile filters, prepare_c3.py
CONF_THRESHOLDS = (30, 45, 60, 75, 90)  # prepare_c3.py
DEPTH_GAMMA = 0.5  # of the depth samples, prepare_s3d.depth_sample
DENSITY_SIZE = 512  # of the density maps, as for C3 (the floorplan is larger: the points are normalized in training)
MIN_VIEW_CORRESPONDENCES = 8  # per frame, prepare_c3.generate
MIN_DEPTH_CORRESPONDENCES = 10  # per depth sample, prepare_s3d.MIN_CORRESPONDENCES
RANSAC_THRESHOLD = 10.0  # px of a 512 px floorplan (prepare_c3.is_consistent)
MAX_WALL_DIST_M = 0.5  # correspondences: points within this distance of a wall of hge.png
SOURCES = ("mapanything", "lidar")  # sample folders and CSV files lamar_<source>.csv
CSV_COLUMNS = ["scene_id", "density", "floorplan", "correspondences", "rgb", "camera_plan", "camera_density"]


def is_consistent(density_px: np.ndarray, plan_px: np.ndarray, plan_size: int) -> bool:
    """
    The correspondences agree with a similarity transform (RANSAC, prepare_c3.is_consistent of photo collections).
    Args:
        density_px (np.ndarray): tensor with shape [N, 2]
        plan_px (np.ndarray): tensor with shape [N, 2]
        plan_size (int): floorplan resolution, which scales the inlier threshold

    Returns:
        bool: consistent
    """
    M, _ = cv2.estimateAffinePartial2D(density_px, plan_px, method=cv2.RANSAC,
                                       ransacReprojThreshold=RANSAC_THRESHOLD * plan_size / 512)
    return M is not None


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


def window_world(frames: Session, recon: dict, plan_source: str) -> tuple:
    """
    World points of the pixels of a window, from which the correspondences are drawn, and its LiDAR points.
    Args:
        frames (Session): the frames of the window
        recon (dict): of reconstruction.reconstruct
        plan_source (str): "recon" (reconstructed points) or "lidar" (LiDAR points)

    Returns:
        tuple: world (np.ndarray [S, H, W, 3], NaN where unused; None for "lidar" without depth) and lidar
            (np.ndarray [S, H, W, 3] in the frame of camera 0, NaN where missing; None without depth)
    """
    H, W = recon["points_cam0"].shape[1:3]
    lidar = None
    if any(p is not None for p in frames.depth_paths):
        lidar = lidar_points_cam0(frames.depth_paths, frames.K_depth, recon["intrinsics"], (H, W), recon["extrinsics"])
    if plan_source == "lidar":
        return (None if lidar is None else world_points(lidar, frames.c2w[0])), lidar
    world = world_points(recon["points_cam0"], frames.c2w[0])
    world[~recon["mask"]] = np.nan
    return world, lidar


def plan_correspondences(world: np.ndarray, plan_params: dict, max_points: int, rng: np.random.Generator,
                         max_wall_dist_m: float | None = MAX_WALL_DIST_M) -> list:
    """
    Per view, the pixels whose 3D points lie on walls of the floorplan (lamar.on_plan) and the floorplan pixels of these points.
    Args:
        world (np.ndarray): tensor with shape [S, H, W, 3], world points of the pixels (NaN where unused)
        plan_params (dict): of lamar.load_floorplan
        max_points (int): at most this many correspondences in the window, drawn at random
        rng (np.random.Generator): random generator
        max_wall_dist_m (float | None): only the points within this distance of a wall [m], None: all inside the floorplan

    Returns:
        list: per view, {"tensor_coords": [N, 2] (u, v) pixels, "plan_coords": [N, 2] floorplan pixels} or None
    """
    S, H, W, _ = world.shape
    flat = world.reshape(-1, 3)
    idx = np.nonzero(on_plan(flat, plan_params, max_wall_dist_m))[0]
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


def mapanything_samples(recon: dict, corrs: list, conf_thresholds: list, plan_params: dict, output: str, name: str) -> list:
    """
    Saves the density maps of a reconstruction (one per confidence threshold, x/z slice and gamma) with their
    correspondences, as prepare_c3.generate. Returns their rows.
    Args:
        recon (dict): of reconstruction.reconstruct
        corrs (list): of plan_correspondences
        conf_thresholds (list): confidence percentiles
        plan_params (dict): of lamar.load_floorplan
        output (str): data directory
        name (str): file stem of the window, relative to output

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
            if not is_consistent(density_px, plan_px, plan_params["plan_size"]):
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
                  max_wall_dist_m: float | None = MAX_WALL_DIST_M) -> list:
    """
    Saves the density map of the LiDAR points of a window, seen from above in the gravity-aligned frame of camera 0,
    with its correspondences (the points on walls), as prepare_s3d.depth_sample: height and x/z percentile filters,
    gamma 0.5, at least 10 correspondences, no consistency test. Returns its row.
    Args:
        lidar_cam0 (np.ndarray): tensor with shape [S, H, W, 3], NaN where missing
        recon (dict): of reconstruction.reconstruct (gravity and extrinsics)
        plan_params (dict): of lamar.load_floorplan
        c2w0 (np.ndarray): camera-to-world of camera 0 with shape [4, 4]
        output (str): data directory
        name (str): file stem of the window, relative to output
        max_wall_dist_m (float | None): only the points within this distance of a wall [m], None: all inside the floorplan

    Returns:
        list: [] or [row {"density", "correspondences", "camera_density"}]
    """
    points = lidar_cam0.reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < MIN_DEPTH_CORRESPONDENCES:
        return []
    R_rect = ground_rotation(recon["est_gravity_cam"])
    ground = points @ R_rect.T
    keep = filter_ground_points(ground)  # highest 5 % and lowest 20 %, x/z 2.5 %: as depth_sample
    ground, points = ground[keep], points[keep]
    proj_params = top_down_projection(ground, R_rect, DENSITY_SIZE)
    keep = on_plan(world_points(points, c2w0), plan_params, max_wall_dist_m)
    if keep.sum() < MIN_DEPTH_CORRESPONDENCES:
        return []
    density_px = get_density_coordinates(ground[keep], proj_params)
    plan_px = world_to_plan(world_points(points[keep], c2w0), plan_params).astype(np.float32)
    corr_path, density_path = f"{name}.npy", f"{name}.png"
    np.save(os.path.join(output, corr_path), np.stack([plan_px, density_px]))
    render_training_map(ground, proj_params, DEPTH_GAMMA).save(os.path.join(output, density_path))
    camera = get_density_coordinates(recon["extrinsics"][-1:, :3, 3] @ R_rect.T, proj_params)[0]
    return [{"density": density_path, "correspondences": corr_path, "camera_density": camera}]


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


def merge_shards(output: str) -> None:
    """
    Concatenates the CSV files of the shards (output/shards/lamar_<source>_<shard>.csv) into output/lamar_<source>.csv.
    Args:
        output (str): data directory
    """
    for source in SOURCES:
        files = sorted(Path(output, "shards").glob(f"lamar_{source}_*.csv"))
        data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(columns=CSV_COLUMNS)
        data.to_csv(os.path.join(output, f"lamar_{source}.csv"), index=False)
        print(f"{source}: {len(data)} samples from {len(files)} shards")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map_dir", help="HGE/sessions/map")
    parser.add_argument("--raw_dir", default=None, help="directory of the raw phone sessions (LiDAR depth); none: no LiDAR")
    parser.add_argument("--floorplan", help="hge.png")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", type=int, default=0, help="this process handles the windows shard, shard + num_shards, ...")
    parser.add_argument("--num_shards", type=int, default=1, help="processes (one per GPU); their CSV files go to <output>/shards")
    parser.add_argument("--merge", action="store_true", help="only concatenates the CSV files of the shards")
    parser.add_argument("--split", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.yaml"))
    parser.add_argument("--sessions", nargs="+", default=["train"], help="splits (train, val, test) or session names")
    parser.add_argument("--plan_size", type=int, default=1024, help="resolution of the floorplan")
    parser.add_argument("--lengths", type=int, nargs="+", default=list(SEQ_LENGTHS), help="window lengths")
    parser.add_argument("--steps", type=int, nargs="+", default=list(TRAIN_STEPS), help="frame intervals of the windows")
    parser.add_argument("--stride", type=float, default=0.5, help="stride between windows, as a fraction of their span")
    parser.add_argument("--windows_per_session", type=int, default=None, help="random subset of the windows of a session")
    parser.add_argument("--conf", type=int, nargs="+", default=list(CONF_THRESHOLDS), help="confidence thresholds (percentiles)")
    parser.add_argument("--plan_source", default="recon", choices=["recon", "lidar"], help="3D points of the correspondences")
    parser.add_argument("--no_lidar_samples", action="store_true", help="no density maps of the LiDAR points")
    parser.add_argument("--scene_id", default="hge", choices=["hge", "cell"])
    parser.add_argument("--cell_m", type=float, default=20.0)
    parser.add_argument("--max_points", type=int, default=20000, help="correspondences kept per window")
    parser.add_argument("--max_wall_dist", type=float, default=MAX_WALL_DIST_M,
                        help="correspondences: points within this distance of a wall [m]; negative: all points inside the floorplan")
    parser.add_argument("--mapanything_model", default="facebook/map-anything")
    parser.add_argument("--recon_cache", default=None, help="directory of cached reconstructions (read and written)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.merge:
        merge_shards(args.output)
        return

    random.seed(args.seed + args.shard)
    np.random.seed(args.seed + args.shard)
    rng = np.random.default_rng(args.seed)  # the same windows in every shard
    rng_points = np.random.default_rng([args.seed, args.shard])  # sampling of the correspondences
    max_wall_dist = None if args.max_wall_dist < 0 else args.max_wall_dist
    plan_img, plan_params = load_floorplan(args.floorplan, args.plan_size)
    for folder in ("floorplan", "shards", *SOURCES):
        os.makedirs(os.path.join(args.output, folder), exist_ok=True)
    if args.recon_cache:
        os.makedirs(args.recon_cache, exist_ok=True)
    plan_path = os.path.join("floorplan", "hge.png")
    if args.shard == 0:  # one writer
        plan_img.save(os.path.join(args.output, plan_path))
        with open(os.path.join(args.output, "floorplan", "hge_params.json"), "w") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in plan_params.items() if k != "wall_dist_px"}, f, indent=2)

    # All windows of the sessions, split between the shards (round robin: similar loads)
    split = load_split(args.split)
    sessions, jobs = {}, []
    for session_name in [name for key in args.sessions for name in (split[key] if key in split else [key])]:
        session = load_session(args.map_dir, args.raw_dir, session_name)
        sessions[session_name] = session.subset(train_frames(session, plan_params))
        windows = iter_train_windows(sessions[session_name].timestamps, tuple(args.lengths), tuple(args.steps), args.stride)
        if args.windows_per_session and len(windows) > args.windows_per_session:
            windows = [windows[i] for i in sorted(rng.choice(len(windows), args.windows_per_session, replace=False))]
        jobs += [(session_name, *window) for window in windows]
    jobs = jobs[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(jobs)} windows")

    model = load_mapanything(args.mapanything_model)
    rows = {source: [] for source in SOURCES}
    progress = tqdm(jobs, desc=f"shard {args.shard}")
    for session_name, start, length, step in progress:
        frames = sessions[session_name].subset(start + step * np.arange(length))
        name = window_name(session_name, frames.timestamps[0], length, step)
        recon = window_recon(frames, model, args.recon_cache, name)
        # deviation of the poses predicted by MapAnything from the ground truth (the points use the ground truth)
        progress.set_postfix(pose_err_deg=f"{recon['pose_error_deg'].max():.2f}", pose_err_m=f"{recon['pose_error_m'].max():.3f}")
        world, lidar = window_world(frames, recon, args.plan_source)
        if world is None:
            continue
        corrs = plan_correspondences(world, plan_params, args.max_points, rng_points, max_wall_dist)

        common = {"scene_id": scene_id_of(frames, args.scene_id, args.cell_m), "floorplan": plan_path,
                  "rgb": os.path.abspath(frames.image_paths[-1]),
                  "camera_plan": world_to_plan(frames.c2w[-1, :3, 3], plan_params)}
        samples = mapanything_samples(recon, corrs, args.conf, plan_params, args.output, os.path.join("mapanything", name))
        if lidar is not None and not args.no_lidar_samples:
            samples += lidar_samples(lidar, recon, plan_params, frames.c2w[0], args.output, os.path.join("lidar", name),
                                     max_wall_dist)
        for sample in samples:
            sample.update(common)
            sample["camera_plan"] = "{:.2f} {:.2f}".format(*common["camera_plan"])
            sample["camera_density"] = "{:.2f} {:.2f}".format(*sample["camera_density"])
            rows[sample["density"].split(os.sep)[0]].append(sample)

    for source, source_rows in rows.items():
        path = os.path.join(args.output, "shards", f"lamar_{source}_{args.shard}.csv") if args.num_shards > 1 else \
            os.path.join(args.output, f"lamar_{source}.csv")
        pd.DataFrame(source_rows, columns=CSV_COLUMNS).to_csv(path, index=False)
        print(f"{source}: {len(source_rows)} samples -> {path}")


if __name__ == "__main__":
    main()
