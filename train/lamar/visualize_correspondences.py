"""Correspondences of the training data of prepare_lamar.py drawn on the photo, the density map of one MapAnything
sample (--conf, --xz, gamma 0.5) and hge.png (full resolution, cropped around them): the pixels whose reconstructed
points lie on walls of the floorplan and are on the density map, all of them (no subsampling besides that of
prepare_lamar.py, --max_points per window). First row: the correspondences of one frame, coloured by the column of
the photo (seven colours, red on the left); second row: those of the whole window, coloured by frame (red: first,
violet: last), next to the same photo without points.
Reads the windows of one epoch of prepare_lamar.py (--data <data>/epoch_<e>, meta/windows.csv) and the cached
reconstructions of their sessions (--session_cache): no GPU.

    python train/lamar/visualize_correspondences.py --map_dir <HGE/sessions/map> --raw_dir <raw sessions> \
        --floorplan hge.png --session_cache <data>/session_cache --data <data>/epoch_000 --output <data>/vis_correspondences
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lamar import load_floorplan, load_session, train_frames, world_to_plan_full  # noqa: E402
from prepare_lamar import plan_correspondences, window_world  # noqa: E402
from reconstruction import SessionCache, window_from_cache  # noqa: E402
from prepare_lamar import DENSITY_SIZE, MAX_WALL_DIST_M  # noqa: E402
from scenealigner.evaluation import get_camera_center_on_density_px, process_gt_correspondences  # noqa: E402
from scenealigner.inference import reconstruct_projection_geometry  # noqa: E402
from scenealigner.training import render_training_map  # noqa: E402
from scenealigner.visualization import draw_correspondence_rows, rainbow  # noqa: E402

def filter_masks(recon: dict, R_rect: np.ndarray, conf_percent: float, xz_slice_val: float) -> tuple[dict, tuple]:
    """
    The successive filters of scenealigner.inference.reconstruct_projection_geometry, to tell which one drops points.
    Args:
        recon (dict): of reconstruction.reconstruct
        R_rect (np.ndarray): camera 0 -> ground frame (+Y along gravity) with shape [3, 3]
        conf_percent (float): confidence threshold (percentile)
        xz_slice_val (float): x/z percentile filter

    Returns:
        tuple[dict, tuple]: cumulative masks {"conf", "height", "xz"} with shape [S, H, W], and the kept heights
            (ceiling, floor cutoffs) [m] along gravity from camera 0 (negative: above it)
    """
    points = (recon["points_cam0"] @ R_rect.T).reshape(-1, 3)
    conf = recon["conf"].reshape(-1)
    masks = {"conf": conf > np.percentile(conf, conf_percent)}
    cutoffs = np.percentile(points[masks["conf"], 1], [5.0, 80.0])
    masks["height"] = masks["conf"] & (points[:, 1] >= cutoffs[0]) & (points[:, 1] <= cutoffs[1])
    x_min, x_max = np.percentile(points[masks["height"], 0], [xz_slice_val, 100 - xz_slice_val])
    z_min, z_max = np.percentile(points[masks["height"], 2], [xz_slice_val, 100 - xz_slice_val])
    masks["xz"] = masks["height"] & (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return {k: v.reshape(recon["conf"].shape) for k, v in masks.items()}, tuple(cutoffs)


def breakdown(tc: np.ndarray, s: int, masks: dict) -> np.ndarray:
    """
    Wall correspondences of frame s, then those left after each filter.
    Args:
        tc (np.ndarray): (u, v) pixels with shape [N, 2]
        s (int): frame
        masks (dict): of filter_masks

    Returns:
        np.ndarray: counts with shape [1 + len(masks)] (wall, conf, height, xz)
    """
    u, v = tc[:, 0].astype(int), tc[:, 1].astype(int)
    return np.array([len(tc)] + [masks[k][s, v, u].sum() for k in masks])


def view_points(tc: np.ndarray, s: int, world: np.ndarray, recon: dict, proj_params: dict, proj_mask: np.ndarray,
                plan_params: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Correspondences of frame s that are on the density map, in the three images.
    Args:
        tc (np.ndarray): (u, v) pixels of the frame with shape [N, 2] (plan_correspondences)
        s (int): frame
        world (np.ndarray): world points of the pixels with shape [S, H, W, 3]
        recon (dict): of reconstruction.reconstruct
        proj_params (dict): of reconstruct_projection_geometry
        proj_mask (np.ndarray): points of the density map with shape [S, H, W]
        plan_params (dict): of lamar.load_floorplan

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: pixels [K, 2] in the photo, the density map and hge.png
    """
    tc = tc[proj_mask[s, tc[:, 1].astype(int), tc[:, 0].astype(int)]]
    u, v = tc.astype(int).T
    plan_px = world_to_plan_full(world[s, v, u], plan_params["A"])  # pixels of hge.png
    density_px, _ = process_gt_correspondences({"tensor_coords": tc, "plan_coords": plan_px}, s, recon["points_cam0"],
                                               proj_params, proj_mask)
    return tc, density_px, plan_px


def fmt(counts: np.ndarray, masks: dict) -> str:
    """Counts of breakdown as "wall 164 > conf 120 > height 17 > xz 17"."""
    return " > ".join(f"{k} {n}" for k, n in zip(["wall", *masks], counts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map_dir", required=True, help="HGE/sessions/map")
    parser.add_argument("--raw_dir", default=None, help="directory of the raw phone sessions (LiDAR gate as prepare_lamar.py)")
    parser.add_argument("--floorplan", required=True, help="hge.png")
    parser.add_argument("--session_cache", required=True, help="reconstructions of the sessions cached by prepare_lamar.py")
    parser.add_argument("--data", required=True, help="windows of one epoch of prepare_lamar.py (<data>/epoch_<e>, <data>/val/epoch_000)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", nargs="+", default=None, help="window names (lamar.window_name); default: random")
    parser.add_argument("--num_windows", type=int, default=10, help="random windows of --data without --windows")
    parser.add_argument("--num_frames", type=int, nargs=2, default=None, help="only random windows of min to max frames")
    parser.add_argument("--frames", type=int, nargs="+", default=None,
                        help="frames of a window (indices, -1: last); default: the 2 with most points on the density map")
    parser.add_argument("--conf", type=float, default=45, help="confidence threshold of the density map (percentile)")
    parser.add_argument("--xz", type=float, default=2.5, help="x/z slice of the density map")
    parser.add_argument("--plan_source", default="recon", choices=["recon", "lidar"])
    parser.add_argument("--max_points", type=int, default=20000, help="as prepare_lamar.py")
    parser.add_argument("--max_wall_dist", type=float, default=MAX_WALL_DIST_M, help="as prepare_lamar.py")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    _, plan_params = load_floorplan(args.floorplan)
    plan_full = np.array(Image.open(args.floorplan).convert("RGB"))
    listed = pd.read_csv(os.path.join(args.data, "meta", "windows.csv")).set_index("name")
    windows = args.windows
    if windows is None:
        names = list(listed.index)
        if args.num_frames:
            names = [n for n in names if args.num_frames[0] <= len(listed.timestamps[n].split()) <= args.num_frames[1]]
        windows = [names[i] for i in sorted(rng.choice(len(names), min(args.num_windows, len(names)), replace=False))]
    os.makedirs(args.output, exist_ok=True)

    sessions, caches = {}, {}
    for name in windows:
        session_name = listed.session[name]
        if session_name not in sessions:
            session = load_session(args.map_dir, args.raw_dir, session_name)
            sessions[session_name] = session.subset(train_frames(session, plan_params))
            caches[session_name] = SessionCache(os.path.join(args.session_cache, session_name))
        timestamps = np.array(listed.timestamps[name].split(), dtype=np.int64)
        frames = sessions[session_name].subset(np.searchsorted(sessions[session_name].timestamps, timestamps))
        recon = window_from_cache(caches[session_name], frames.timestamps, frames.c2w)
        length = len(frames)
        world, _ = window_world(frames, recon, args.plan_source)
        if world is None:
            print(f"{name}: no LiDAR")
            continue
        corrs = plan_correspondences(world, plan_params, args.max_points, rng, None if args.max_wall_dist < 0 else args.max_wall_dist)
        proj_params, points, proj_mask = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"],
                                                                         args.conf, args.xz, density_size=DENSITY_SIZE)
        density_img = np.array(render_training_map(points, proj_params, 0.5).convert("RGB"))
        masks, (ceil_m, floor_m) = filter_masks(recon, proj_params["R_rect"], args.conf, args.xz)
        assert (masks["xz"] == proj_mask).all()
        views = [s for s in range(length) if corrs[s] is not None]
        if not views:
            print(f"{name}: too few correspondences")
            continue
        counts = {s: breakdown(corrs[s]["tensor_coords"], s, masks) for s in views}
        print(f"{name}: {fmt(sum(counts.values()), masks)} (height kept: {-ceil_m:.1f} m above to {floor_m:.1f} m below camera 0)")
        trajectory_plan = world_to_plan_full(frames.c2w[:, :3, 3], plan_params["A"])  # pixels of hge.png
        trajectory_density = get_camera_center_on_density_px(recon["extrinsics"], proj_params)
        print(f"  density map: {DENSITY_SIZE / proj_params['scale']:.1f} m wide, trajectory "
              f"{np.linalg.norm(np.diff(frames.c2w[:, :3, 3], axis=0), axis=1).sum():.1f} m long")
        frames_drawn = [f % length for f in args.frames] if args.frames else sorted(views, key=lambda s: -counts[s][-1])[:2]
        # Correspondences of the density map (as in the training data, all of them): per frame, then the window
        per_view = {s: view_points(corrs[s]["tensor_coords"], s, world, recon, proj_params, proj_mask, plan_params) for s in views}
        window_row = {"rgb_px": None, "density_px": np.concatenate([per_view[s][1] for s in views]),
                      "plan_px": np.concatenate([per_view[s][2] for s in views]),
                      "colors": rainbow(np.concatenate([np.full(len(per_view[s][0]), s / max(1, length - 1)) for s in views])),
                      "label": f"window: {sum(len(per_view[s][0]) for s in views)} points, colour = frame (first red, last violet)"}
        W = recon["preprocessed_images"].shape[2]
        for s in frames_drawn:
            print(f"  frame {s}: {fmt(counts[s], masks)}" if s in counts else f"  frame {s}: too few correspondences")
            tc, density_px, plan_px = per_view.get(s, (np.zeros((0, 2)),) * 3)
            frame_row = {"rgb_px": tc, "density_px": density_px, "plan_px": plan_px, "colors": rainbow(tc[:, 0] / W),
                         "label": f"frame {s}: {len(tc)} points, colour = column of the photo"}
            c2w = frames.c2w[s]
            camera_px = world_to_plan_full(c2w[:3, 3], plan_params["A"])
            camera_dir = world_to_plan_full(c2w[:3, 3] + c2w[:3, 2], plan_params["A"]) - camera_px  # optical axis
            image = draw_correspondence_rows(recon["preprocessed_images"][s], [frame_row, window_row], density_img, plan_full,
                                             camera_px, camera_dir, trajectory_density, trajectory_plan)
            Image.fromarray(image).save(os.path.join(args.output, f"{name}_f{s:03d}.png"))


if __name__ == "__main__":
    main()
