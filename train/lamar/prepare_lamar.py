"""Training data from LaMAR HGE, as the training data of SceneAligner (upstream prepare_c3.py and prepare_s3d.py).
Every window of frames gives one sample of its MapAnything reconstruction (ground-truth intrinsics and relative
poses as input) with a random variation of prepare_c3.py (confidence threshold 30 to 90, x/z slice, gamma), and one
sample of its LiDAR depth maps, as prepare_s3d.py makes one of the ground-truth depth map of each view.

    # 1. MapAnything on chunks of consecutive frames of every session (GPU; one process per GPU with --shard i --num_shards n)
    python train/lamar/prepare_lamar.py --stage recon --map_dir <HGE/sessions/map> --raw_dir <raw sessions> \
        --floorplan hge.png --session_cache <data>/session_cache
    # 2. the windows of every epoch, assembled from the cached chunks (CPU; processes with --shard, then --merge)
    python train/lamar/prepare_lamar.py --stage samples ... --session_cache <data>/session_cache --output <data>
    python train/lamar/prepare_lamar.py --merge --output <data>

The kept frames of a session are reconstructed once, in chunks of at most --chunk_size consecutive frames; the
points of every frame are cached in its camera frame, and a window takes those of its frames, placed with the
ground-truth poses relative to its first frame (reconstruction.window_from_cache).
Windows (--windows_per_session per session and epoch, --epochs sets in <data>/epoch_<e>, every one a different
combination of frames): 50 to 500 frames from a random start, every 1 to 4 frames (random intervals), inside a run
of kept frames (C3 reconstructs 50 to 150 random photos of a scene).
Correspondences, as prepare_s3d.py: the pixels of the density map of the 3D points whose positions lie on walls of
the floorplan (within --max_wall_dist 0.2 m of a wall pixel of hge.png), and the floorplan pixels of these positions. With --plan_source recon (default) the positions are those
of the reconstructed points placed with the ground-truth pose of the first camera; with --plan_source lidar,
those of the LiDAR points (prepare_s3d.py takes the ground-truth depth). The CSV rows also hold the photo and the
position of the last camera, for the visualizations of train.py; <epoch>/meta/windows.csv lists the frames of the
windows (visualize_correspondences.py).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lamar import (TRAIN_FRAMES, TRAIN_MAX_STEP, Session, filter_ground_points, ground_rotation, lidar_points_cam0,  # noqa: E402
                   load_floorplan, load_session, load_split, on_plan, sample_distinct_windows, session_chunks,
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
    lidar = recon.get("lidar_cam0")  # window_from_cache
    if lidar is None and any(p is not None for p in frames.depth_paths):
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


def mapanything_sample(recon: dict, corrs: list, conf_thresholds: list, plan_params: dict, output: str, name: str,
                       rng: np.random.Generator) -> list:
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

    Returns:
        list: [] or [row {"density", "correspondences", "camera_density"}]
    """
    pairs_conf_xz = [(conf, xz) for conf in conf_thresholds for xz in XZ_SLICES]
    for i in rng.permutation(len(pairs_conf_xz)):
        conf, xz = pairs_conf_xz[i]
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
        gamma = GAMMAS[rng.integers(len(GAMMAS))]
        corr_path, density_path = f"{name}_conf{conf}_xz{xz}.npy", f"{name}_conf{conf}_xz{xz}_g{gamma}.png"
        np.save(os.path.join(output, corr_path), np.stack([plan_px, density_px]))
        render_training_map(points, proj_params, gamma).save(os.path.join(output, density_path))
        camera = get_camera_center_on_density_px(recon["extrinsics"], proj_params)[-1]
        return [{"density": density_path, "correspondences": corr_path, "camera_density": camera}]
    return []


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


def merge_shards(output: str) -> None:
    """
    Concatenates the CSV files of the shards of every epoch (<epoch>/shards/lamar_<source>_<shard>.csv and
    windows_<shard>.csv) into <epoch>/lamar_<source>.csv and <epoch>/meta/windows.csv.
    Args:
        output (str): data directory
    """
    for folder in sorted(Path(output).glob("epoch_*")):
        for kind, path in [*((f"lamar_{s}", folder / f"lamar_{s}.csv") for s in SOURCES), ("windows", folder / "meta" / "windows.csv")]:
            files = sorted((folder / "shards").glob(f"{kind}_*.csv"))
            data = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
            data.to_csv(path, index=False)
            print(f"{folder.name} {kind}: {len(data)} rows from {len(files)} shards")


def reconstruct_sessions(sessions: dict, args) -> None:
    """
    MapAnything on chunks of consecutive kept frames of every session (lamar.session_chunks), cached in
    --session_cache/<session>/t<first timestamp>_n<frames> with the LiDAR points of the frames; this shard takes the
    chunks shard, shard + num_shards, ...; cached chunks are skipped.
    Args:
        sessions (dict): {name: Session of the kept frames}
        args: of main
    """
    jobs = [(name, chunk) for name, session in sessions.items() for chunk in session_chunks(session.timestamps, args.chunk_size)]
    jobs = jobs[args.shard::args.num_shards]
    print(f"shard {args.shard}/{args.num_shards}: {len(jobs)} chunks of {sum(len(c) for _, c in jobs)} frames")
    model = None
    progress = tqdm(jobs, desc=f"shard {args.shard}")
    for name, chunk in progress:
        frames = sessions[name].subset(chunk)
        directory = os.path.join(args.session_cache, name, f"t{int(frames.timestamps[0])}_n{len(frames)}")
        if os.path.exists(os.path.join(directory, "timestamps.npy")):
            continue
        model = model or load_mapanything(args.mapanything_model)
        recon = reconstruct(frames.image_paths, frames.K, frames.c2w, model)
        # deviation of the poses predicted by MapAnything from the ground truth (the points use the ground truth)
        progress.set_postfix(pose_err_deg=f"{recon['pose_error_deg'].max():.2f}", pose_err_m=f"{recon['pose_error_m'].max():.3f}")
        lidar = None
        if any(p is not None for p in frames.depth_paths):  # in the frame of every camera
            lidar = lidar_points_cam0(frames.depth_paths, frames.K_depth, recon["intrinsics"], recon["mask"].shape[1:],
                                      np.tile(np.eye(4), (len(frames), 1, 1)))
        save_chunk(directory, recon, frames.timestamps, lidar)


def sample_windows(sessions: dict, args, plan_img, plan_params: dict) -> None:
    """
    The samples of --epochs sets of --windows_per_session random windows per session, every window a different
    combination of frames (over all epochs), assembled from the cached chunks: one density map of the reconstruction
    (mapanything_sample) and one of the LiDAR points (lidar_samples). Epoch e goes to <output>/epoch_<e>; this shard
    takes the windows shard, shard + num_shards, ...
    Args:
        sessions (dict): {name: Session of the kept frames}
        args: of main
        plan_img: floorplan image of lamar.load_floorplan
        plan_params (dict): of lamar.load_floorplan
    """
    rng = np.random.default_rng(args.seed)  # the same windows in every shard
    seen = {name: set() for name in sessions}
    jobs = [(epoch, name, window) for epoch in range(args.epochs) for name, session in sessions.items()
            for window in sample_distinct_windows(session.timestamps, args.windows_per_session, rng, seen[name],
                                                  tuple(args.num_frames), args.max_step)]
    max_wall_dist = None if args.max_wall_dist < 0 else args.max_wall_dist
    plan_path = os.path.join("floorplan", "hge.png")
    for epoch in range(args.epochs):
        for folder in ("floorplan", "shards", "meta", *SOURCES):
            os.makedirs(os.path.join(epoch_dir(args.output, epoch), folder), exist_ok=True)
        if args.shard == 0:  # one writer
            plan_img.save(os.path.join(epoch_dir(args.output, epoch), plan_path))
            with open(os.path.join(epoch_dir(args.output, epoch), "floorplan", "hge_params.json"), "w") as f:
                json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in plan_params.items() if k != "wall_dist_px"}, f, indent=2)
    print(f"shard {args.shard}/{args.num_shards}: {len(jobs[args.shard::args.num_shards])} of {len(jobs)} windows")

    caches = {}  # memory-mapped
    rows = {epoch: {source: [] for source in (*SOURCES, "windows")} for epoch in range(args.epochs)}
    for index in tqdm(range(args.shard, len(jobs), args.num_shards), desc=f"shard {args.shard}"):
        epoch, session_name, window = jobs[index]
        output = epoch_dir(args.output, epoch)
        frames = sessions[session_name].subset(window)
        name = window_name(session_name, frames.timestamps)
        if session_name not in caches:
            caches[session_name] = SessionCache(os.path.join(args.session_cache, session_name))
        recon = window_from_cache(caches[session_name], frames.timestamps, frames.c2w)
        world, lidar = window_world(frames, recon, args.plan_source)
        if world is None:
            continue
        rng_window = np.random.default_rng([args.seed, index])  # correspondences and density map, whatever the shards
        corrs = plan_correspondences(world, plan_params, args.max_points, rng_window, max_wall_dist)

        common = {"scene_id": scene_id_of(frames, args.scene_id, args.cell_m), "floorplan": plan_path,
                  "rgb": os.path.abspath(frames.image_paths[-1]),
                  "camera_plan": world_to_plan(frames.c2w[-1, :3, 3], plan_params)}
        samples = mapanything_sample(recon, corrs, args.conf, plan_params, output, os.path.join("mapanything", name), rng_window)
        if lidar is not None and not args.no_lidar_samples:
            samples += lidar_samples(lidar, recon, plan_params, frames.c2w[0], output, os.path.join("lidar", name),
                                     max_wall_dist)
        for sample in samples:
            sample.update(common)
            sample["camera_plan"] = "{:.2f} {:.2f}".format(*common["camera_plan"])
            sample["camera_density"] = "{:.2f} {:.2f}".format(*sample["camera_density"])
            rows[epoch][sample["density"].split(os.sep)[0]].append(sample)
        rows[epoch]["windows"].append({"name": name, "session": session_name,
                                       "timestamps": " ".join(str(int(t)) for t in frames.timestamps)})

    for epoch, epoch_rows in rows.items():
        for kind, kind_rows in epoch_rows.items():
            columns = ["name", "session", "timestamps"] if kind == "windows" else CSV_COLUMNS
            stem = kind if kind == "windows" else f"lamar_{kind}"
            path = os.path.join(epoch_dir(args.output, epoch), "shards", f"{stem}_{args.shard}.csv")
            pd.DataFrame(kind_rows, columns=columns).to_csv(path, index=False)
    if args.num_shards == 1:
        merge_shards(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["recon", "samples"], help="recon: MapAnything on the chunks of the sessions "
                        "(GPU); samples: the windows of every epoch from the cached chunks (CPU)")
    parser.add_argument("--map_dir", help="HGE/sessions/map")
    parser.add_argument("--raw_dir", default=None, help="directory of the raw phone sessions (LiDAR depth); none: no LiDAR")
    parser.add_argument("--floorplan", help="hge.png")
    parser.add_argument("--session_cache", help="cached reconstructions of the chunks of the sessions (written by recon)")
    parser.add_argument("--output", help="data directory of the samples: <output>/epoch_<e>")
    parser.add_argument("--shard", type=int, default=0, help="this process handles the jobs shard, shard + num_shards, ...")
    parser.add_argument("--num_shards", type=int, default=1, help="processes; their CSV files go to <epoch>/shards")
    parser.add_argument("--merge", action="store_true", help="only concatenates the CSV files of the shards")
    parser.add_argument("--split", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.yaml"))
    parser.add_argument("--sessions", nargs="+", default=["train"], help="splits (train, val, test) or session names")
    parser.add_argument("--plan_size", type=int, default=1024, help="resolution of the floorplan")
    parser.add_argument("--chunk_size", type=int, default=TRAIN_FRAMES[1], help="most frames reconstructed together "
                        "(delete --session_cache after changing it)")
    parser.add_argument("--epochs", type=int, default=20, help="sets of windows, one per training epoch")
    parser.add_argument("--windows_per_session", type=int, default=50, help="random windows per session and epoch")
    parser.add_argument("--num_frames", type=int, nargs=2, default=list(TRAIN_FRAMES), help="min and max frames of a window")
    parser.add_argument("--max_step", type=int, default=TRAIN_MAX_STEP, help="largest interval between the frames of a window")
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
    if args.merge:
        merge_shards(args.output)
        return
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
