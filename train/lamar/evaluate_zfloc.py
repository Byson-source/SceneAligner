"""Evaluation of SceneAligner on the inputs of the Z-FLoc LaMAR HGE experiment: the point clouds of Z-FLoc
(tools/metadata_generation/generate_pcd.py: MapAnything with the ground-truth intrinsics and poses, one window of L
kept frames per file, points in the camera frame of its last frame), its windows and its metric (the positions of the
last 10 frames all within 1 m of poses_map.npy, 0.055 m/px, seeds 42 to 46, a window without a model fails).
--solver zfloc (default): only the correspondences come from SceneAligner; all their pairs (nC2) are similarity
hypotheses for the back end of Z-FLoc (scale band, coarse score on the BEV wall lines, alignment/penetration score, LM;
zfloc_solver.py) with the BEV of Z-FLoc extract_primitives.py, whose gravity is also used for the density map.
--solver ransac: the gravity is the ground truth (LaMAR world z-up) and the RANSAC hypotheses are restricted to the
scale band of Z-FLoc (--scale_tol, negative for cv2.estimateAffinePartial2D as in evaluate_lamar.py).

    python train/lamar/evaluate_zfloc.py --pcd_root <dir of hge_<L>/> --zfloc_data <create_lamar_hge.py output> \\
        --map_dir <HGE/sessions/map> --floorplan hge.png --checkpoint_dir <run> --zfloc_repo <z-floc>
    # on several GPUs: one process per GPU with --shard i --num_shards n (shells/lamar/zfloc_eval.sh), then
    python train/lamar/evaluate_zfloc.py --merge --output <json> [--seq_length ...] [--seeds ...]
"""
import argparse
import json
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import zfloc_solver  # noqa: E402
from evaluate_lamar import LAST_FRAMES, SEEDS, SUCCESS_M, align, correspondences, summarize, write_results  # noqa: E402
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


def localize_zfloc(recon: dict, bev: dict, plan_img: Image.Image, model, refs: dict, param: dict, plan_params: dict,
                   c2w: np.ndarray, poses_map: np.ndarray) -> tuple[dict, dict]:
    """
    One window and seed with --solver zfloc: SceneAligner correspondences, then the back end of Z-FLoc.
    Args:
        recon (dict): of load_zfloc_window, "est_gravity_cam" replaced by the gravity of the BEV
        bev (dict): of zfloc_solver.load_bev
        plan_img (Image.Image): resized floorplan given to SceneAligner
        model: DINOv3 with the LoRA layers
        refs (dict): of zfloc_solver.setup_refs
        param (dict): match_param of Z-FLoc
        plan_params (dict): of lamar.load_floorplan
        c2w (np.ndarray): tensor with shape [L, 4, 4], ground-truth camera-to-world poses of the window
        poses_map (np.ndarray): tensor with shape [L, 2], ground-truth positions, hge.png px

    Returns:
        tuple[dict, dict]: record fields ("success", "failed", "t_errs", "r_errs", "n_corr") and the visualizations
            ("density" PIL image, "overlay" BGR array or None)
    """
    result = correspondences(recon, plan_img, model)
    src = zfloc_solver.density_to_bev(result["src_px"], result["proj_params"], bev["gravity"], bev["bev_img"].shape)
    dst = zfloc_solver.plan_to_full(result["dst_px"], plan_params)
    model_bev = zfloc_solver.localize(src, dst, bev, refs, param)
    record = {"success": False, "failed": model_bev is None, "t_errs": [], "r_errs": [], "n_corr": len(src)}
    vis = {"density": result["density_img"], "overlay": None}
    if model_bev is not None:
        t_errs, r_errs, est = zfloc_solver.window_errors(model_bev, bev, c2w, poses_map, plan_params, param, LAST_FRAMES)
        record.update(success=bool(np.all(t_errs < SUCCESS_M)), t_errs=t_errs.tolist(), r_errs=r_errs.tolist())
        vis["overlay"] = zfloc_solver.overlay(bev, refs, model_bev, est, poses_map[-LAST_FRAMES:], src, dst)
    return record, vis


def window_summary(records: list) -> str:
    """
    One-line summary of the records of one window (all seeds) for the log.
    Args:
        records (list): of main, one per seed

    Returns:
        str: e.g. "SR 3/5, median 0.62 m, failed 0" (", 412 correspondences" with --solver zfloc)
    """
    t_errs = sum((r["t_errs"] for r in records), [])
    median = f"{np.median(t_errs):.2f} m" if t_errs else "-"
    n_corr = [r["n_corr"] for r in records if "n_corr" in r]
    corr = f", {np.mean(n_corr):.0f} correspondences" if n_corr else ""
    return f"SR {sum(r['success'] for r in records)}/{len(records)}, median {median}, failed {sum(r['failed'] for r in records)}{corr}"


def elapsed(seconds: float) -> str:
    """Seconds -> h:mm:ss."""
    return str(timedelta(seconds=int(seconds)))


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
    parser.add_argument("--solver", choices=["zfloc", "ransac"], default="zfloc",
                        help="zfloc: nC2 hypotheses + scoring and LM of Z-FLoc; ransac: RANSAC of the correspondences")
    parser.add_argument("--zfloc_repo", help="Z-FLoc repository (--solver zfloc)")
    parser.add_argument("--ransac_thr", type=float, default=0.04, help="RANSAC threshold, fraction of the floorplan resolution")
    parser.add_argument("--scale_tol", type=float, default=SCALE_TOL, help="scale band of Z-FLoc; negative: unrestricted RANSAC")
    parser.add_argument("--save_vis", action="store_true", help="saves the density maps and alignments of the first seed next to the output")
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
    if args.solver == "zfloc":
        zfloc_solver.import_zfloc(args.zfloc_repo)
        param, refs = zfloc_solver.match_param(), zfloc_solver.setup_refs(args.floorplan)
        print(f"model: {args.checkpoint_dir}, solver: nC2 hypotheses + Z-FLoc scoring and LM", flush=True)
    else:
        print(f"model: {args.checkpoint_dir}, solver: RANSAC, scale band: {args.scale_tol if args.scale_tol >= 0 else 'off'}", flush=True)
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
        print(f"{name}: {len(c2w)} frames", flush=True)

    # Windows of all lengths and scenes; window_index (per length) seeds the alignment, whatever the shard
    jobs = []
    for seq_length in args.seq_length:
        window_index = 0
        for name, (c2w, _) in scenes.items():
            for k in range(len(c2w) // seq_length):
                jobs.append((seq_length, window_index, name, k))
                window_index += 1
    jobs = jobs[args.shard::args.num_shards]
    for seq_length in args.seq_length:
        print(f"L={seq_length}: {sum(j[0] == seq_length for j in jobs)} windows", flush=True)
    print(f"{len(jobs)} windows x {len(args.seeds)} seeds (shard {args.shard}/{args.num_shards})", flush=True)

    records = []
    start = time.time()
    for job_index, (seq_length, window_index, name, k) in enumerate(jobs):
        window_start = time.time()
        window = slice(k * seq_length, (k + 1) * seq_length)
        c2w, poses_map = scenes[name][0][window], scenes[name][1][window]
        stem = window_stem(k, seq_length)
        path = os.path.join(args.pcd_root, f"hge_{seq_length}", name, "pcd", f"{stem}.npz")
        recon = load_zfloc_window(path, c2w) if os.path.exists(path) else None
        if args.solver == "zfloc":  # the BEV of Z-FLoc; the density map with its gravity
            bev = zfloc_solver.load_bev(os.path.dirname(os.path.dirname(path)), stem)
            recon = None if recon is None or bev is None else {**recon, "est_gravity_cam": bev["gravity"]}
        for seed in args.seeds:  # the randomness of the alignment (query sampling, RANSAC), per window
            np.random.seed(seed * 1_000_000 + window_index)
            torch.manual_seed(seed * 1_000_000 + window_index)
            if args.solver == "zfloc":
                record = {"L": seq_length, "seed": seed, "success": False, "failed": True, "t_errs": [], "r_errs": []}
                if recon is not None:
                    fields, vis = localize_zfloc(recon, bev, plan_img, model, refs, param, plan_params, c2w, poses_map)
                    record.update(fields)
                    if args.save_vis and seed == args.seeds[0]:
                        vis_path = os.path.join(vis_dir, f"{name}_L{seq_length}_{stem}")
                        vis["density"].save(f"{vis_path}_density.png")
                        if vis["overlay"] is not None:
                            cv2.imwrite(f"{vis_path}.png", vis["overlay"])
                records.append(record)
                continue
            alignment = None if recon is None else align(recon, plan_img, model, args.ransac_thr, plan_params, scale_tol,
                                                         np.random.default_rng(seed * 1_000_000 + window_index))
            M = None if alignment is None else alignment["M"]
            record = {"L": seq_length, "seed": seed, "success": False, "failed": M is None, "t_errs": [], "r_errs": []}
            if args.save_vis and seed == args.seeds[0] and alignment is not None:  # the density map also of failed windows
                vis_path = os.path.join(vis_dir, f"{name}_L{seq_length}_{stem}")
                alignment["density_img"].save(f"{vis_path}_density.png")
                if M is not None:
                    image = draw_alignment(np.array(alignment["density_img"]), np.array(plan_img), M, alignment["src_px"])
                    Image.fromarray(image).save(f"{vis_path}.png")
            if M is not None:
                t_errs, r_errs = window_errors(alignment, recon, c2w, poses_map, plan_params)
                record.update(success=bool(np.all(t_errs < SUCCESS_M)), t_errs=t_errs.tolist(), r_errs=r_errs.tolist())
            records.append(record)

        # Progress: this window, the SR@1m of its length so far, the time
        done = time.time() - start
        length_records = [r for r in records if r["L"] == seq_length]
        status = window_summary(records[-len(args.seeds):]) if recon is not None else f"missing {path} or its BEV: failure"
        print(f"[{job_index + 1}/{len(jobs)}] L={seq_length} {name} {stem}: {status} | "
              f"L={seq_length} SR so far {100 * np.mean([r['success'] for r in length_records]):.1f} % | "
              f"{time.time() - window_start:.1f} s (elapsed {elapsed(done)}, "
              f"ETA {elapsed(done / (job_index + 1) * (len(jobs) - job_index - 1))})", flush=True)
        if job_index + 1 == len(jobs) or jobs[job_index + 1][0] != seq_length:
            print(f"L={seq_length} finished: SR@1m {100 * np.mean([r['success'] for r in length_records]):.1f} % "
                  f"({len(length_records) // len(args.seeds)} windows, mean over the seeds)", flush=True)

    if args.num_shards > 1:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(shard_path.format(args.shard), "w") as f:
            json.dump(records, f)
        return
    write_results(summarize(records, args.seq_length, args.seeds), args.seq_length, args.seeds, args.output)


if __name__ == "__main__":
    main()
