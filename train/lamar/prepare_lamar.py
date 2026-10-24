"""Training data from LaMAR HGE, as the training data of SceneAligner (upstream prepare_c3.py and prepare_s3d.py).
Every window of frames gives one sample of its MapAnything reconstruction (ground-truth intrinsics and relative
poses as input) with a random variation of prepare_c3.py (confidence threshold 30 to 90, x/z slice, gamma), and one
sample of its LiDAR depth maps, as prepare_s3d.py makes one of the ground-truth depth map of each view.

    # 1. MapAnything on chunks of consecutive frames of every session (GPU; --workers: one process per GPU)
    python train/lamar/prepare_lamar.py --stage recon --map_dir <HGE/sessions/map> --raw_dir <raw sessions> \
        --floorplan hge.png --session_cache <data>/session_cache
    # 2. the windows of every epoch, assembled from the cached chunks (CPU; --workers processes)
    python train/lamar/prepare_lamar.py --stage samples ... --session_cache <data>/session_cache --output <data>

The kept frames of a session are reconstructed once, in chunks of at most --chunk_size consecutive frames; the
points of every frame are cached in its camera frame, and a window takes those of its frames, placed with the
ground-truth poses relative to its first frame (reconstruction.window_from_cache).
Windows (--windows_per_epoch per epoch, --epochs sets in <data>/epoch_<e>, every one a different combination of
frames; lamar.sample_epoch_windows): a session drawn proportionally to its kept frames, 50 to 500 frames drawn at
random among n to 8 n consecutive kept frames of a run (C3 reconstructs 50 to 150 random photos of a scene); in
half of the windows, a quarter to a half of the frames replaced with frames of one or two other sessions within 2 m
of its cameras. Every frame keeps a random fraction in [--min_keep 0.25, 1] of its pixels and LiDAR points.
Correspondences, as prepare_s3d.py: the pixels of the density map of the 3D points whose positions lie on walls of
the floorplan (within --max_wall_dist 0.2 m of a wall pixel of hge.png), and the floorplan pixels of these positions. With --plan_source recon (default) the positions are those
of the reconstructed points placed with the ground-truth pose of the first camera; with --plan_source lidar,
those of the LiDAR points (prepare_s3d.py takes the ground-truth depth). The CSV rows also hold the photo and the
position of the last camera, for the visualizations of train.py; <epoch>/meta/windows.csv lists the frames of the
windows (visualize_correspondences.py).
"""
import argparse
import json
import multiprocessing
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lamar import (CHUNK_SIZE, MIX_PROB, MIX_RADIUS_M, TRAIN_FRAMES, TRAIN_MAX_SPREAD, Session, concat_sessions,  # noqa: E402
                   filter_ground_points, ground_rotation, lidar_points, lidar_points_cam0, load_floorplan, load_session,
                   load_split, on_plan, sample_epoch_windows, session_chunks,
                   top_down_projection, train_frames, window_name, world_to_plan)
from reconstruction import SessionCache, load_mapanything, reconstruct, save_chunk, window_from_cache  # noqa: E402
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
MAX_WALL_DIST_M = 0.2  # correspondences: points within this distance of a wall of hge.png
SOURCES = ("mapanything", "lidar")  # sample folders and CSV files lamar_<source>.csv
PLAN_PATH = os.path.join("floorplan", "hge.png")  # of every data folder
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
    World points of the pixels of a window from which the correspondences are drawn: only the pixels with a point.
    Args:
        frames (Session): the frames of the window
        recon (dict): of reconstruction.reconstruct
        plan_source (str): "recon" (reconstructed points) or "lidar" (LiDAR points on the pixel grid)

    Returns:
        tuple: world (np.ndarray [K, 3]) and pixels (np.ndarray [K], increasing indices into the S x H x W pixels);
            (None, None) for "lidar" without depth
    """
    if plan_source == "lidar":
        if all(p is None for p in frames.depth_paths):
            return None, None
        points = lidar_points_cam0(frames.depth_paths, frames.K_depth, recon["intrinsics"], recon["mask"].shape[1:],
                                   recon["extrinsics"]).reshape(-1, 3)
        pixels = np.nonzero(np.isfinite(points).all(axis=1))[0]
    else:
        points = recon["points_cam0"].reshape(-1, 3)
        pixels = np.nonzero(recon["mask"].reshape(-1))[0]
    return world_points(points[pixels], frames.c2w[0]), pixels


def world_grid(world: np.ndarray, pixels: np.ndarray, shape: tuple) -> np.ndarray:
    """
    World points of window_world on the pixel grid (visualizations).
    Args:
        world (np.ndarray): tensor with shape [K, 3]
        pixels (np.ndarray): indices of the pixels with shape [K]
        shape (tuple): (S, H, W)

    Returns:
        np.ndarray: tensor with shape [S, H, W, 3], NaN where there is no point
    """
    grid = np.full((int(np.prod(shape)), 3), np.nan, dtype=np.float32)
    grid[pixels] = world
    return grid.reshape(*shape, 3)


def plan_correspondences(world: np.ndarray, pixels: np.ndarray, shape: tuple, plan_params: dict, max_points: int,
                         rng: np.random.Generator, max_wall_dist_m: float | None = MAX_WALL_DIST_M) -> list:
    """
    Per view, the pixels whose 3D points lie on walls of the floorplan (lamar.on_plan) and the floorplan pixels of these points.
    Args:
        world (np.ndarray): tensor with shape [K, 3], world points of the pixels (window_world)
        pixels (np.ndarray): increasing indices of these pixels into the S x H x W pixels, with shape [K]
        shape (tuple): (S, H, W)
        plan_params (dict): of lamar.load_floorplan
        max_points (int): at most this many correspondences in the window, drawn at random
        rng (np.random.Generator): random generator
        max_wall_dist_m (float | None): only the points within this distance of a wall [m], None: all inside the floorplan

    Returns:
        list: per view, {"tensor_coords": [N, 2] (u, v) pixels, "plan_coords": [N, 2] floorplan pixels} or None
    """
    S, H, W = shape
    keep = np.nonzero(on_plan(world, plan_params, max_wall_dist_m))[0]
    if len(keep) > max_points:
        keep = np.sort(rng.choice(keep, max_points, replace=False))
    plan_px = world_to_plan(world[keep], plan_params).astype(np.float32)
    view, pixel = np.divmod(pixels[keep], H * W)
    v, u = np.divmod(pixel, W)
    corrs = []
    for s in range(S):
        sel = view == s
        if sel.sum() < MIN_VIEW_CORRESPONDENCES:
            corrs.append(None)
        else:
            corrs.append({"tensor_coords": np.stack([u[sel], v[sel]], axis=1).astype(np.float32), "plan_coords": plan_px[sel]})
    return corrs


def projection_geometry(recon: dict, conf: float, xz: float, keep: np.ndarray | None) -> tuple:
    """
    scenealigner.inference.reconstruct_projection_geometry of the kept pixels only (their percentiles).
    Args:
        recon (dict): of reconstruction.reconstruct
        conf (float): confidence threshold (percentile)
        xz (float): x/z slice
        keep (np.ndarray | None): boolean tensor with shape [S, H, W], the pixels kept; None: all

    Returns:
        tuple: projection parameters, the points of the density map [N, 3] (ground frame) and their mask [S, H, W]
    """
    if keep is None:
        return reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"], conf, xz,
                                               density_size=DENSITY_SIZE)
    proj_params, points, kept_mask = reconstruct_projection_geometry(recon["points_cam0"][keep], recon["conf"][keep],
                                                                     recon["est_gravity_cam"], conf, xz, density_size=DENSITY_SIZE)
    mask = np.zeros(keep.shape, dtype=bool)
    mask[keep] = kept_mask
    return proj_params, points, mask


def mapanything_sample(recon: dict, corrs: list, conf_thresholds: list, plan_params: dict, output: str, name: str,
                       rng: np.random.Generator, keep: np.ndarray | None = None) -> list:
    """
    Saves one density map of a reconstruction with its correspondences, as prepare_c3.generate makes each of its
    samples, for a random confidence threshold, x/z slice and gamma: the (threshold, slice) pairs are tried in random
    order until one keeps at least 8 correspondences in a frame and they agree with a similarity transform.
    Args:
        recon (dict): of reconstruction.reconstruct
        corrs (list): of plan_correspondences
        conf_thresholds (list): confidence percentiles
        plan_params (dict): of lamar.load_floorplan
        output (str): data directory
        name (str): file stem of the window, relative to output
        rng (np.random.Generator): random generator
        keep (np.ndarray | None): boolean tensor with shape [S, H, W], the pixels kept (dropped at random); None: all

    Returns:
        list: [] or [row {"density", "correspondences", "camera_density"}]
    """
    pairs_conf_xz = [(conf, xz) for conf in conf_thresholds for xz in XZ_SLICES]
    for i in rng.permutation(len(pairs_conf_xz)):
        conf, xz = pairs_conf_xz[i]
        proj_params, points, mask = projection_geometry(recon, conf, xz, keep)
        pairs = [process_gt_correspondences(corr, s, recon["points_cam0"], proj_params, mask)
                 for s, corr in enumerate(corrs) if corr is not None]
        pairs = [pair for pair in pairs if len(pair[0]) >= MIN_VIEW_CORRESPONDENCES]
        if not pairs:
            continue
        density_px, plan_px = np.concatenate([p[0] for p in pairs]), np.concatenate([p[1] for p in pairs])
        if not is_consistent(density_px, plan_px, plan_params["plan_size"]):
            continue
        gamma = GAMMAS[rng.integers(len(GAMMAS))]
        corr_path, density_path = f"{name}_conf{conf}_xz{xz}.npy", f"{name}_conf{conf}_xz{xz}_g{gamma}.png"
        np.save(os.path.join(output, corr_path), np.stack([plan_px, density_px]))
        render_training_map(points, proj_params, gamma).save(os.path.join(output, density_path))
        camera = get_camera_center_on_density_px(recon["extrinsics"], proj_params)[-1]
        return [{"density": density_path, "correspondences": corr_path, "camera_density": camera}]
    return []


def lidar_samples(points: np.ndarray, recon: dict, plan_params: dict, c2w0: np.ndarray, output: str, name: str,
                  max_points: int, rng: np.random.Generator, max_wall_dist_m: float | None = MAX_WALL_DIST_M) -> list:
    """
    Saves the density map of the LiDAR points of a window, seen from above in the gravity-aligned frame of camera 0,
    with its correspondences (the points on walls, at most max_points at random), as prepare_s3d.depth_sample: height
    and x/z percentile filters, gamma 0.5, at least 10 correspondences, no consistency test. Returns its row.
    Args:
        points (np.ndarray): tensor with shape [M, 3], LiDAR points in the frame of camera 0 (lamar.lidar_points)
        recon (dict): of reconstruction.reconstruct (gravity and extrinsics)
        plan_params (dict): of lamar.load_floorplan
        c2w0 (np.ndarray): camera-to-world of camera 0 with shape [4, 4]
        output (str): data directory
        name (str): file stem of the window, relative to output
        max_points (int): most correspondences, drawn at random
        rng (np.random.Generator): random generator
        max_wall_dist_m (float | None): only the points within this distance of a wall [m], None: all inside the floorplan

    Returns:
        list: [] or [row {"density", "correspondences", "camera_density"}]
    """
    if len(points) < MIN_DEPTH_CORRESPONDENCES:
        return []
    R_rect = ground_rotation(recon["est_gravity_cam"])
    ground = points @ R_rect.T
    keep = filter_ground_points(ground)  # highest 5 % and lowest 20 %, x/z 2.5 %: as depth_sample
    ground, points = ground[keep], points[keep]
    proj_params = top_down_projection(ground, R_rect, DENSITY_SIZE)
    keep = np.nonzero(on_plan(world_points(points, c2w0), plan_params, max_wall_dist_m))[0]
    if len(keep) < MIN_DEPTH_CORRESPONDENCES:
        return []
    if len(keep) > max_points:
        keep = np.sort(rng.choice(keep, max_points, replace=False))
    density_px = get_density_coordinates(ground[keep], proj_params)
    plan_px = world_to_plan(world_points(points[keep], c2w0), plan_params).astype(np.float32)
    corr_path, density_path = f"{name}.npy", f"{name}.png"
    np.save(os.path.join(output, corr_path), np.stack([plan_px, density_px]))
    render_training_map(ground, proj_params, DEPTH_GAMMA).save(os.path.join(output, density_path))
    camera = get_density_coordinates(recon["extrinsics"][-1:, :3, 3] @ R_rect.T, proj_params)[0]
    return [{"density": density_path, "correspondences": corr_path, "camera_density": camera}]


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




def epoch_dir(output: str, epoch: int) -> str:
    """Data folder of the windows of one epoch (train.py takes <data>/epoch_<e> at epoch e)."""
    return os.path.join(output, f"epoch_{epoch:03d}")


WORKER = {}  # state of the worker processes (reconstruct_sessions, sample_windows)


def chunk_directory(args, name: str, frames: Session) -> str:
    """Folder of a cached chunk: --session_cache/<session>/t<first timestamp>_n<frames>."""
    return os.path.join(args.session_cache, name, f"t{int(frames.timestamps[0])}_n{len(frames)}")


def init_recon_worker(gpus, args, sessions: dict) -> None:
    """A MapAnything worker on its own GPU (taken from the queue gpus), before CUDA is initialized."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus.get())
    WORKER.update(args=args, sessions=sessions, model=None)


def chunk_recon(job: tuple) -> tuple[float, float]:
    """
    MapAnything on one chunk (session name, kept-frame indices), cached with save_chunk.
    Returns:
        tuple[float, float]: largest deviation of the poses predicted by MapAnything from the ground truth [deg, m]
            (the points use the ground truth)
    """
    name, chunk = job
    frames = WORKER["sessions"][name].subset(chunk)
    if WORKER["model"] is None:
        WORKER["model"] = load_mapanything(WORKER["args"].mapanything_model)
    recon = reconstruct(frames.image_paths, frames.K, frames.c2w, WORKER["model"])
    save_chunk(chunk_directory(WORKER["args"], name, frames), recon, frames.timestamps)
    return float(recon["pose_error_deg"].max()), float(recon["pose_error_m"].max())


def reconstruct_sessions(sessions: dict, args) -> None:
    """
    MapAnything on chunks of consecutive kept frames of every session (lamar.session_chunks), cached in
    --session_cache; --workers processes, one per GPU, with one progress bar; cached chunks are skipped.
    Args:
        sessions (dict): {name: Session of the kept frames}
        args: of main
    """
    jobs = [(name, chunk) for name, session in sessions.items() for chunk in session_chunks(session.timestamps, args.chunk_size)]
    todo = [(name, chunk) for name, chunk in jobs if not os.path.exists(
        os.path.join(chunk_directory(args, name, sessions[name].subset(chunk)), "timestamps.npy"))]
    print(f"{len(todo)} of {len(jobs)} chunks to do ({sum(len(c) for _, c in todo)} frames), {args.workers} GPUs", flush=True)
    if not todo:
        return
    context = multiprocessing.get_context("spawn")  # CUDA: no fork
    gpus = context.Queue()
    for gpu in range(args.workers):
        gpus.put(gpu)
    with context.Pool(args.workers, init_recon_worker, (gpus, args, sessions)) as pool:
        progress = tqdm(pool.imap_unordered(chunk_recon, todo), total=len(todo), desc="chunks")
        for err_deg, err_m in progress:
            progress.set_postfix(pose_err_deg=f"{err_deg:.2f}", pose_err_m=f"{err_m:.3f}")




def window_rows(index: int) -> tuple[int, dict]:
    """
    Saves the samples of one window of the jobs of sample_windows (in WORKER) and returns their CSV rows. The pixels
    of every frame (and its LiDAR points) are kept with a random fraction in [--min_keep, 1].
    Args:
        index (int): of the window in the jobs (its random generator: the samples do not depend on the workers)

    Returns:
        tuple[int, dict]: index and {source: rows, "windows": [row of the frames of the window]} (empty without points)
    """
    args, sessions, plan_params, caches = WORKER["args"], WORKER["sessions"], WORKER["plan_params"], WORKER["caches"]
    epoch, parts = WORKER["jobs"][index]
    rows = {kind: [] for kind in (*SOURCES, "windows")}
    output = epoch_dir(args.output, epoch)
    frames = concat_sessions([sessions[session].subset(idx) for session, idx in parts])
    part_timestamps = [(session, sessions[session].timestamps[idx]) for session, idx in parts]
    name = window_name(part_timestamps)
    for session, _ in parts:
        if session not in caches:  # memory-mapped, one per worker
            caches[session] = SessionCache(os.path.join(args.session_cache, session))
    recon = window_from_cache([(caches[session], ts) for session, ts in part_timestamps], frames.c2w)
    rng = np.random.default_rng([args.seed, index])  # dropped pixels, correspondences and density map
    fractions = rng.uniform(args.min_keep, 1.0, len(frames))
    keep = rng.random(recon["mask"].shape) < fractions[:, None, None]
    world, pixels = window_world(frames, recon, args.plan_source)
    if world is None:
        return index, rows
    kept = keep.reshape(-1)[pixels]
    max_wall_dist = None if args.max_wall_dist < 0 else args.max_wall_dist
    corrs = plan_correspondences(world[kept], pixels[kept], recon["mask"].shape, plan_params, args.max_points, rng,
                                 max_wall_dist)
    samples = mapanything_sample(recon, corrs, args.conf, plan_params, output, os.path.join("mapanything", name), rng, keep)
    if not args.no_lidar_samples and any(p is not None for p in frames.depth_paths):
        lidar = lidar_points(frames.depth_paths, frames.K_depth, recon["extrinsics"], fractions, rng)
        samples += lidar_samples(lidar, recon, plan_params, frames.c2w[0], output, os.path.join("lidar", name),
                                 args.max_points, rng, max_wall_dist)
    camera_plan = world_to_plan(frames.c2w[-1, :3, 3], plan_params)
    for sample in samples:
        sample.update({"scene_id": scene_id_of(frames, args.scene_id, args.cell_m), "floorplan": PLAN_PATH,
                       "rgb": os.path.abspath(frames.image_paths[-1]), "camera_plan": "{:.2f} {:.2f}".format(*camera_plan),
                       "camera_density": "{:.2f} {:.2f}".format(*sample["camera_density"])})
        rows[sample["density"].split(os.sep)[0]].append(sample)
    rows["windows"].append({"name": name, "frames": " ".join(f"{session}:{int(t)}" for session, ts in part_timestamps for t in ts),
                            "keep": " ".join(f"{f:.3f}" for f in fractions)})
    return index, rows


def sample_windows(sessions: dict, args, plan_img, plan_params: dict) -> None:
    """
    The samples of --epochs sets of --windows_per_epoch random windows (lamar.sample_epoch_windows), every window a
    different combination of frames (over all epochs), assembled from the cached chunks: one density map of the reconstruction
    (mapanything_sample) and one of the LiDAR points (lidar_samples). Epoch e goes to <output>/epoch_<e>; the windows
    are shared by --workers processes (one progress bar), and the CSV files of an epoch are written once all its
    windows are done (meta/windows.csv last: the epoch is skipped on a rerun).
    Args:
        sessions (dict): {name: Session of the kept frames}
        args: of main
        plan_img: floorplan image of lamar.load_floorplan
        plan_params (dict): of lamar.load_floorplan
    """
    rng = np.random.default_rng(args.seed)
    seen = set()
    timestamps = {name: session.timestamps for name, session in sessions.items()}
    positions = {name: session.c2w[:, :3, 3] for name, session in sessions.items()}
    jobs = [(epoch, parts) for epoch in range(args.epochs)
            for parts in sample_epoch_windows(timestamps, positions, args.windows_per_epoch, rng, seen, tuple(args.num_frames),
                                              args.max_spread, args.mix_prob, args.mix_radius)]
    done = {e for e in range(args.epochs) if os.path.exists(os.path.join(epoch_dir(args.output, e), "meta", "windows.csv"))}
    todo = [i for i, job in enumerate(jobs) if job[0] not in done]
    for epoch in set(range(args.epochs)) - done:
        folder = epoch_dir(args.output, epoch)
        for sub in ("floorplan", "meta", *SOURCES):
            os.makedirs(os.path.join(folder, sub), exist_ok=True)
        plan_img.save(os.path.join(folder, PLAN_PATH))
        with open(os.path.join(folder, "floorplan", "hge_params.json"), "w") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in plan_params.items() if k != "wall_dist_px"}, f, indent=2)
    print(f"{len(todo)} of {len(jobs)} windows to do ({len(done)} epochs done), {args.workers} workers", flush=True)
    if not todo:
        return

    left = {e: sum(jobs[i][0] == e for i in todo) for e in set(range(args.epochs)) - done}
    rows = {e: {} for e in left}  # epoch -> index -> rows
    WORKER.update(args=args, sessions=sessions, plan_params=plan_params, jobs=jobs, caches={})
    with multiprocessing.get_context("fork").Pool(args.workers) as pool:
        for index, window in tqdm(pool.imap_unordered(window_rows, todo), total=len(todo), desc="windows", smoothing=0.05):
            epoch = jobs[index][0]
            rows[epoch][index] = window
            left[epoch] -= 1
            if left[epoch] == 0:  # the CSV files of the epoch, in the order of the windows
                ordered = [rows[epoch][i] for i in sorted(rows[epoch])]
                folder = epoch_dir(args.output, epoch)
                for source in SOURCES:
                    pd.DataFrame([r for w in ordered for r in w[source]], columns=CSV_COLUMNS).to_csv(
                        os.path.join(folder, f"lamar_{source}.csv"), index=False)
                pd.DataFrame([r for w in ordered for r in w["windows"]], columns=["name", "frames", "keep"]).to_csv(
                    os.path.join(folder, "meta", "windows.csv"), index=False)
                tqdm.write(f"epoch {epoch}: " + ", ".join(f"{k} {sum(len(w[k]) for w in ordered)}" for k in (*SOURCES, "windows")))
                del rows[epoch]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["recon", "samples"], help="recon: MapAnything on the chunks of the sessions "
                        "(GPU); samples: the windows of every epoch from the cached chunks (CPU)")
    parser.add_argument("--map_dir", help="HGE/sessions/map")
    parser.add_argument("--raw_dir", default=None, help="directory of the raw phone sessions (LiDAR depth); none: no LiDAR")
    parser.add_argument("--floorplan", help="hge.png")
    parser.add_argument("--session_cache", help="cached reconstructions of the chunks of the sessions (written by recon)")
    parser.add_argument("--output", help="data directory of the samples: <output>/epoch_<e>")
    parser.add_argument("--workers", type=int, default=1, help="processes: recon, one per GPU; samples, CPU processes")
    parser.add_argument("--split", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.yaml"))
    parser.add_argument("--sessions", nargs="+", default=["train"], help="splits (train, val, test) or session names")
    parser.add_argument("--plan_size", type=int, default=1024, help="resolution of the floorplan")
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE, help="most frames reconstructed together "
                        "(delete --session_cache after changing it)")
    parser.add_argument("--epochs", type=int, default=20, help="sets of windows, one per training epoch")
    parser.add_argument("--windows_per_epoch", type=int, default=17728, help="random windows per epoch (two samples "
                        "each: 35456, 1108 steps of 32 as the 600 windows x 60 density maps before)")
    parser.add_argument("--num_frames", type=int, nargs=2, default=list(TRAIN_FRAMES), help="min and max frames of a window")
    parser.add_argument("--max_spread", type=int, default=TRAIN_MAX_SPREAD,
                        help="the n frames of a window are drawn among n to max_spread * n consecutive frames")
    parser.add_argument("--mix_prob", type=float, default=MIX_PROB, help="probability of frames of other sessions in a window")
    parser.add_argument("--mix_radius", type=float, default=MIX_RADIUS_M, help="other sessions: frames within this distance [m]")
    parser.add_argument("--min_keep", type=float, default=0.25, help="pixels kept per frame: a random fraction in [min_keep, 1]")
    parser.add_argument("--conf", type=int, nargs="+", default=list(CONF_THRESHOLDS), help="confidence thresholds (percentiles)")
    parser.add_argument("--plan_source", default="recon", choices=["recon", "lidar"], help="3D points of the correspondences")
    parser.add_argument("--no_lidar_samples", action="store_true", help="no density maps of the LiDAR points")
    parser.add_argument("--scene_id", default="hge", choices=["hge", "cell"])
    parser.add_argument("--cell_m", type=float, default=20.0)
    parser.add_argument("--max_points", type=int, default=20000, help="correspondences kept per window")
    parser.add_argument("--max_wall_dist", type=float, default=MAX_WALL_DIST_M,
                        help="correspondences: points within this distance of a wall [m] (< 0: all points inside the floorplan)")
    parser.add_argument("--mapanything_model", default="facebook/map-anything")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.stage is None:
        parser.error("--stage recon or --stage samples")

    plan_img, plan_params = load_floorplan(args.floorplan, args.plan_size)
    split = load_split(args.split)
    sessions = {}
    for session_name in [name for key in args.sessions for name in (split[key] if key in split else [key])]:
        session = load_session(args.map_dir, args.raw_dir, session_name)
        sessions[session_name] = session.subset(train_frames(session, plan_params))
    if args.stage == "recon":
        reconstruct_sessions(sessions, args)
    else:
        sample_windows(sessions, args, plan_img, plan_params)


if __name__ == "__main__":
    main()
