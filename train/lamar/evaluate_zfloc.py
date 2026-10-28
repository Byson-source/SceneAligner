"""Evaluation of SceneAligner on the inputs of the Z-FLoc LaMAR HGE experiment: the point clouds of Z-FLoc
(tools/metadata_generation/generate_pcd.py: MapAnything with the ground-truth intrinsics and poses, one window of L
kept frames per file, points in the camera frame of its last frame), its windows and its metric (the positions of the
last 10 frames all within 1 m of poses_map.npy, 0.055 m/px, seeds 42 to 46, a window without a model fails).
The gravity is the ground truth (LaMAR world z-up); the RANSAC hypotheses are restricted to the scale band of Z-FLoc
(--scale_tol, negative for cv2.estimateAffinePartial2D as in evaluate_lamar.py).

    python train/lamar/evaluate_zfloc.py --pcd_root <dir of hge_<L>/> --zfloc_data <create_lamar_hge.py output> \\
        --map_dir <HGE/sessions/map> --floorplan hge.png --checkpoint_dir <run>
    # on several GPUs: one process per GPU with --shard i --num_shards n (shells/lamar/zfloc_eval.sh), then
    python train/lamar/evaluate_zfloc.py --merge --output <json> [--seq_length ...] [--seeds ...]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate_lamar import LAST_FRAMES, SEEDS, SUCCESS_M, align, summarize, write_results  # noqa: E402
from lamar import (HGE_INDEX_SHIFT, METRES_PER_PIXEL, SEQ_LENGTHS, gravity_cam0, kept_indices, load_floorplan,  # noqa: E402
                   load_split, load_trajectory)
from ransac import SCALE_TOL  # noqa: E402
from scenealigner.evaluation import compute_pose_error_analytic, get_camera_center_on_density_px  # noqa: E402
from scenealigner.inference import MODEL_REPO, load_model  # noqa: E402
from scenealigner.visualization import draw_alignment  # noqa: E402


def window_stem(index: int, seq_length: int) -> str:
    """
    File stem of a Z-FLoc window: the kept-frame index of its last frame.
    Args:
        index (int): window index k in the scene, frames [k L, (k + 1) L)
        seq_length (int): window length L

    Returns:
        str: e.g. "00019-0"
    """
    return f"{(index + 1) * seq_length - 1:05d}-0"


def load_zfloc_window(path: str | Path, c2w: np.ndarray) -> dict:
    """
    A point cloud of Z-FLoc as the reconstruction dict of SceneAligner (frame 0 = the last frame of the window).
    Args:
        path (str | Path): <pcd_root>/hge_<L>/<scene>/pcd/<stem>.npz of generate_pcd.py
        c2w (np.ndarray): tensor with shape [L, 4, 4], ground-truth camera-to-world poses of the window (LaMAR
            world, z-up), in time order

    Returns:
        dict: "points_cam0" [N, 3], "conf" None, "est_gravity_cam" [3] (ground truth), "extrinsics" [L, 4, 4]
            (camera-to-last-camera, in time order)
    """
    with np.load(path) as data:
        points, poses = data["all_pcds"], data["gt_camera_poses"]  # poses: last frame first, then frames 0 to L-2
    poses = np.concatenate((poses[1:], poses[:1]))  # time order
    extrinsics = np.linalg.inv(poses[-1]) @ poses
    expected = np.linalg.inv(c2w[-1]) @ c2w
    if len(poses) != len(c2w) or np.abs(extrinsics[:, :3, 3] - expected[:, :3, 3]).max() > 1e-3:
        raise ValueError(f"{path}: the poses differ from trajectories.txt (frames or index shift)")
    return {"points_cam0": points, "conf": None, "est_gravity_cam": gravity_cam0(c2w[-1]), "extrinsics": extrinsics}


def window_errors(alignment: dict, recon: dict, c2w: np.ndarray, poses_map: np.ndarray, plan_params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Errors of the last cameras of a window with the metric of Z-FLoc (zfloc/offline.py).
    Args:
        alignment (dict): of evaluate_lamar.align
        recon (dict): of load_zfloc_window
        c2w (np.ndarray): tensor with shape [L, 4, 4], ground-truth camera-to-world poses of the window
        poses_map (np.ndarray): tensor with shape [L, 2], ground-truth positions of the window, hge.png px
        plan_params (dict): of lamar.load_floorplan

    Returns:
        tuple[np.ndarray, np.ndarray]: position errors [m] and angular errors [deg] of the last LAST_FRAMES cameras
    """
    proj_params, M = alignment["proj_params"], alignment["M"]
    centres = get_camera_center_on_density_px(recon["extrinsics"], proj_params)
    gt_plan = poses_map * plan_params["scale"] + plan_params["off"]  # hge.png px -> resized px
    forward_plan = c2w[:, :3, 2][:, :2] @ plan_params["A"][:, :2].T  # viewing directions on the floorplan
    t_errs, r_errs = [], []
    for i in range(len(c2w) - LAST_FRAMES, len(c2w)):
        t_err_px, r_err = compute_pose_error_analytic(M, recon["extrinsics"][i], proj_params, centres[i], gt_plan[i], forward_plan[i])
        t_errs.append(t_err_px / plan_params["scale"] * METRES_PER_PIXEL)  # resized px -> hge.png px -> metres
        r_errs.append(r_err)
    return np.array(t_errs), np.array(r_errs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pcd_root", help="directory of hge_<L>/<scene>/pcd/ (generate_pcd.py --evol_path <pcd_root>/hge_<L>)")
    parser.add_argument("--zfloc_data", help="output of Z-FLoc create_lamar_hge.py: split.yaml, <scene>/poses_map.npy")
    parser.add_argument("--map_dir", help="HGE/sessions/map (ground-truth gravity)")
    parser.add_argument("--floorplan", help="hge.png")
    parser.add_argument("--checkpoint_dir", default=MODEL_REPO, help='LoRA layers, or "pre-trained" for DINOv3')
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--plan_size", type=int, default=1024, help="resolution of the floorplan (density maps: 512)")
    parser.add_argument("--seq_length", type=int, nargs="+", default=list(SEQ_LENGTHS), help="window lengths")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--ransac_thr", type=float, default=0.04, help="RANSAC threshold, fraction of the floorplan resolution")
    parser.add_argument("--scale_tol", type=float, default=SCALE_TOL, help="scale band of Z-FLoc; negative: unrestricted RANSAC")
    parser.add_argument("--save_vis", action="store_true", help="saves the alignments of the first seed next to the output")
    parser.add_argument("--output", default="outputs/zfloc_results.json")
    parser.add_argument("--shard", type=int, default=0, help="this process evaluates the windows shard, shard + num_shards, ...")
    parser.add_argument("--num_shards", type=int, default=1, help="processes (one per GPU); their records go to <output>.shard<i>.json")
    parser.add_argument("--merge", action="store_true", help="only merges the records of the shards into the results")
    args = parser.parse_args()
    shard_path = os.path.splitext(args.output)[0] + ".shard{}.json"
    if args.merge:
        records = sum((json.load(open(shard_path.format(i))) for i in range(args.num_shards)), [])
        write_results(summarize(records, args.seq_length, args.seeds), args.seq_length, args.seeds, args.output)
        return

    plan_img, plan_params = load_floorplan(args.floorplan, args.plan_size)
    model = load_model(args.checkpoint_dir, args.dinov3_weights)
    scale_tol = args.scale_tol if args.scale_tol >= 0 else None
    vis_dir = os.path.splitext(args.output)[0] + "_vis"
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # Kept frames of the test scenes: ground-truth poses (trajectories.txt + index shift) and positions (poses_map.npy)
    scenes = {}
    for name in load_split(os.path.join(args.zfloc_data, "split.yaml"))["test"]:
        _, _, c2w = load_trajectory(os.path.join(args.map_dir, "trajectories.txt"), name)
        poses_map = np.load(os.path.join(args.zfloc_data, name, "poses_map.npy"))
        c2w = c2w[kept_indices(len(c2w), HGE_INDEX_SHIFT[name])]
        if len(c2w) != len(poses_map):
            raise ValueError(f"{name}: {len(c2w)} kept frames, {len(poses_map)} rows in poses_map.npy")
        scenes[name] = (c2w, poses_map)
        print(f"{name}: {len(c2w)} frames")

    # Windows of all lengths and scenes; window_index (per length) seeds the alignment, whatever the shard
    jobs = []
    for seq_length in args.seq_length:
        window_index = 0
        for name, (c2w, _) in scenes.items():
            for k in range(len(c2w) // seq_length):
                jobs.append((seq_length, window_index, name, k))
                window_index += 1
    jobs = jobs[args.shard::args.num_shards]

    records = []
    for seq_length, window_index, name, k in tqdm(jobs, desc=f"shard {args.shard}/{args.num_shards}"):
        window = slice(k * seq_length, (k + 1) * seq_length)
        c2w, poses_map = scenes[name][0][window], scenes[name][1][window]
        stem = window_stem(k, seq_length)
        path = os.path.join(args.pcd_root, f"hge_{seq_length}", name, "pcd", f"{stem}.npz")
        recon = load_zfloc_window(path, c2w) if os.path.exists(path) else None
        if recon is None:
            print(f"missing {path}: counted as a failure")
        for seed in args.seeds:  # the randomness of the alignment (query sampling, RANSAC), per window
            np.random.seed(seed * 1_000_000 + window_index)
            torch.manual_seed(seed * 1_000_000 + window_index)
            alignment = None if recon is None else align(recon, plan_img, model, args.ransac_thr, plan_params, scale_tol,
                                                         np.random.default_rng(seed * 1_000_000 + window_index))
            record = {"L": seq_length, "seed": seed, "success": False, "failed": alignment is None, "t_errs": [], "r_errs": []}
            if alignment is not None:
                t_errs, r_errs = window_errors(alignment, recon, c2w, poses_map, plan_params)
                record.update(success=bool(np.all(t_errs < SUCCESS_M)), t_errs=t_errs.tolist(), r_errs=r_errs.tolist())
                if args.save_vis and seed == args.seeds[0]:
                    image = draw_alignment(np.array(alignment["density_img"]), np.array(plan_img), alignment["M"], alignment["src_px"])
                    Image.fromarray(image).save(os.path.join(vis_dir, f"{name}_L{seq_length}_{stem}.png"))
            records.append(record)

    if args.num_shards > 1:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(shard_path.format(args.shard), "w") as f:
            json.dump(records, f)
        return
    write_results(summarize(records, args.seq_length, args.seeds), args.seq_length, args.seeds, args.output)


if __name__ == "__main__":
    main()
