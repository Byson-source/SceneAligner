"""Correspondences of the training data of prepare_lamar.py drawn on the photo, the density map and the floorplan:
the pixels of a frame whose reconstructed points lie on walls, these points on the density map of one MapAnything
sample (--conf, --xz, gamma 0.5) and on hge.png (full resolution, cropped around them). A point keeps one of seven
colours in the three images (from the left to the right of the photo). Only the points of the density map are drawn.
Reads the reconstructions cached by prepare_lamar.py (--recon_cache): no GPU.

    python train/lamar/visualize_correspondences.py --map_dir <HGE/sessions/map> --raw_dir <raw sessions> \
        --floorplan hge.png --recon_cache <data>/recon_cache --output <data>/vis_correspondences
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lamar import load_floorplan, load_session, train_frames, world_to_plan_full  # noqa: E402
from prepare_lamar import wall_correspondences, window_world  # noqa: E402
from reconstruction import load_recon  # noqa: E402
from prepare_lamar import DENSITY_SIZE  # noqa: E402
from scenealigner.evaluation import process_gt_correspondences  # noqa: E402
from scenealigner.inference import reconstruct_projection_geometry  # noqa: E402
from scenealigner.training import render_training_map  # noqa: E402
from scenealigner.visualization import draw_rgb_plan_correspondences  # noqa: E402

WINDOW_NAME = re.compile(r"(?P<session>.+)_s(?P<start>\d{5})_L(?P<length>\d+)$")  # prepare_lamar.main


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map_dir", required=True, help="HGE/sessions/map")
    parser.add_argument("--raw_dir", default=None, help="directory of the raw phone sessions (LiDAR gate as prepare_lamar.py)")
    parser.add_argument("--floorplan", required=True, help="hge.png")
    parser.add_argument("--recon_cache", required=True, help="reconstructions cached by prepare_lamar.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--windows", nargs="+", default=None, help="window names (<session>_s<start>_L<length>); default: random")
    parser.add_argument("--num_windows", type=int, default=10, help="random windows of the cache without --windows")
    parser.add_argument("--frames", type=int, nargs="+", default=[0, -1], help="frames of a window (indices, -1: last)")
    parser.add_argument("--num_points", type=int, default=300, help="correspondences drawn per frame")
    parser.add_argument("--conf", type=float, default=45, help="confidence threshold of the density map (percentile)")
    parser.add_argument("--xz", type=float, default=2.5, help="x/z slice of the density map")
    parser.add_argument("--plan_source", default="recon", choices=["recon", "lidar"])
    parser.add_argument("--no_lidar_gate", action="store_true")
    parser.add_argument("--max_points", type=int, default=20000, help="as prepare_lamar.py")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    _, plan_params = load_floorplan(args.floorplan)
    plan_full = np.array(Image.open(args.floorplan).convert("RGB"))
    windows = args.windows
    if windows is None:
        cached = sorted(p.stem for p in Path(args.recon_cache).glob("*.npz"))
        windows = [cached[i] for i in sorted(rng.choice(len(cached), min(args.num_windows, len(cached)), replace=False))]
    os.makedirs(args.output, exist_ok=True)

    sessions = {}
    for name in windows:
        match = WINDOW_NAME.match(name)
        session_name, start, length = match["session"], int(match["start"]), int(match["length"])
        if session_name not in sessions:
            session = load_session(args.map_dir, args.raw_dir, session_name)
            sessions[session_name] = session.subset(train_frames(session, plan_params))
        frames = sessions[session_name].subset(np.arange(start, start + length))
        recon = load_recon(os.path.join(args.recon_cache, f"{name}.npz"))
        world, _ = window_world(frames, recon, args.plan_source, not args.no_lidar_gate)
        if world is None:
            print(f"{name}: no LiDAR")
            continue
        corrs = wall_correspondences(world, plan_params, args.max_points, rng)
        proj_params, points, proj_mask = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"],
                                                                         args.conf, args.xz, density_size=DENSITY_SIZE)
        density_img = np.array(render_training_map(points, proj_params, 0.5).convert("RGB"))
        for frame in args.frames:
            s = frame % length
            corr = corrs[s]
            if corr is None:
                print(f"{name} frame {s}: too few correspondences")
                continue
            tc = corr["tensor_coords"]
            tc = tc[proj_mask[s, tc[:, 1].astype(int), tc[:, 0].astype(int)]]  # points of the density map
            if len(tc) == 0:
                print(f"{name} frame {s}: no correspondence on the density map")
                continue
            tc = tc[rng.choice(len(tc), min(args.num_points, len(tc)), replace=False)]
            u, v = tc.astype(int).T
            plan_px = world_to_plan_full(world[s, v, u], plan_params["A"])  # pixels of hge.png
            density_px, _ = process_gt_correspondences({"tensor_coords": tc, "plan_coords": plan_px}, s, recon["points_cam0"],
                                                       proj_params, proj_mask)
            c2w = frames.c2w[s]
            camera_px = world_to_plan_full(c2w[:3, 3], plan_params["A"])
            camera_dir = world_to_plan_full(c2w[:3, 3] + c2w[:3, 2], plan_params["A"]) - camera_px  # optical axis
            image = draw_rgb_plan_correspondences(recon["preprocessed_images"][s], tc, plan_full, plan_px, camera_px,
                                                  camera_dir, density_img, density_px)
            Image.fromarray(image).save(os.path.join(args.output, f"{name}_f{s:03d}.png"))
            print(f"{name} frame {s}: {len(corr['tensor_coords'])} correspondences, {len(tc)} drawn")


if __name__ == "__main__":
    main()
