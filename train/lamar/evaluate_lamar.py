"""Evaluation on the LaMAR HGE test sessions with the protocol of Z-FLoc: non-overlapping windows of L consecutive
frames, reconstructed with MapAnything (ground-truth intrinsics and relative poses) and aligned to the floorplan;
a window succeeds when the positions of its last 10 cameras are all within 1 m of the ground truth (SR@1m), with
the random seeds 42 to 46.

    python train/lamar/evaluate_lamar.py --map_dir <HGE/sessions/map> --floorplan hge.png --checkpoint_dir <run>
    python train/lamar/evaluate_lamar.py ... --checkpoint_dir pre-trained          # DINOv3 without LoRA layers
    python train/lamar/evaluate_lamar.py ... --checkpoint_dir jhcho99/SceneAligner  # the LoRA layers trained on C3
    python train/lamar/evaluate_lamar.py ... --sessions val --seq_length 20 --seeds 42   # model selection
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lamar import (METRES_PER_PIXEL, SEQ_LENGTHS, iter_test_windows, load_floorplan, load_session, load_split,  # noqa: E402
                   test_frames, train_frames, world_to_plan)
from reconstruction import load_mapanything, load_recon, reconstruct, save_recon  # noqa: E402
from scenealigner.evaluation import compute_pose_error_analytic, get_camera_center_on_density_px  # noqa: E402
from scenealigner.inference import (MODEL_REPO, estimate_correspondences, load_model, reconstruct_projection_geometry,  # noqa: E402
                                    render_density_map)
from scenealigner.visualization import draw_alignment  # noqa: E402

SEEDS = (42, 43, 44, 45, 46)
LAST_FRAMES = 10  # cameras of a window that are evaluated
SUCCESS_M = 1.0


def align(recon: dict, plan_img: Image.Image, model, ransac_thr: float) -> dict | None:
    """
    Alignment of a reconstruction to the floorplan (scenealigner.inference.align_to_floorplan), or None when RANSAC
    finds no similarity transform (the window counts as a failure, as in Z-FLoc).
    Args:
        recon (dict): of reconstruction.reconstruct
        plan_img (Image.Image): square floorplan
        model: DINOv3 with the LoRA layers
        ransac_thr (float): RANSAC threshold, as a fraction of the floorplan resolution

    Returns:
        dict | None: "proj_params", "density_img", "src_px", "dst_px", "M" (density map px -> floorplan px)
    """
    proj_params, points, _ = reconstruct_projection_geometry(recon["points_cam0"], recon["conf"], recon["est_gravity_cam"],
                                                             density_size=plan_img.size[0])
    density_img = render_density_map(points, proj_params)
    src_px, dst_px = estimate_correspondences(model, density_img, plan_img)
    M, _ = cv2.estimateAffinePartial2D(src_px, dst_px, method=cv2.RANSAC, ransacReprojThreshold=ransac_thr * plan_img.size[0])
    if M is None:
        return None
    return {"proj_params": proj_params, "density_img": density_img, "src_px": src_px, "dst_px": dst_px, "M": M}


def window_errors(alignment: dict, recon: dict, c2w: np.ndarray, plan_params: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Errors of the last cameras of a window on the floorplan.
    Args:
        alignment (dict): of align
        recon (dict): of reconstruction.reconstruct
        c2w (np.ndarray): tensor with shape [S, 4, 4], ground-truth camera-to-world poses of the window
        plan_params (dict): of lamar.load_floorplan

    Returns:
        tuple[np.ndarray, np.ndarray]: position errors [m] and angular errors [deg] of the last LAST_FRAMES cameras
    """
    proj_params, M = alignment["proj_params"], alignment["M"]
    centres = get_camera_center_on_density_px(recon["extrinsics"], proj_params)
    gt_plan = world_to_plan(c2w[:, :3, 3], plan_params)  # resized floorplan px
    forward_plan = c2w[:, :3, 2][:, :2] @ plan_params["A"][:, :2].T  # viewing directions on the floorplan
    t_errs, r_errs = [], []
    for i in range(max(0, len(c2w) - LAST_FRAMES), len(c2w)):
        t_err_px, r_err = compute_pose_error_analytic(M, recon["extrinsics"][i], proj_params, centres[i], gt_plan[i], forward_plan[i])
        t_errs.append(t_err_px / plan_params["scale"] * METRES_PER_PIXEL)  # resized px -> hge.png px -> metres
        r_errs.append(r_err)
    return np.array(t_errs), np.array(r_errs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map_dir", required=True, help="HGE/sessions/map")
    parser.add_argument("--floorplan", required=True, help="hge.png")
    parser.add_argument("--checkpoint_dir", default=MODEL_REPO, help='LoRA layers, or "pre-trained" for DINOv3')
    parser.add_argument("--dinov3_weights", default=None, help="Meta's DINOv3 ViT-B/16 weights")
    parser.add_argument("--split", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.yaml"))
    parser.add_argument("--sessions", nargs="+", default=["test"], help="splits (test, val) or session names")
    parser.add_argument("--plan_size", type=int, default=1024, help="resolution of the floorplan and the density maps")
    parser.add_argument("--seq_length", type=int, nargs="+", default=list(SEQ_LENGTHS), help="window lengths")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--ransac_thr", type=float, default=0.04, help="RANSAC threshold, fraction of the floorplan resolution")
    parser.add_argument("--zfloc_npz", default=None, help="Z-FLoc assets/metadata/hge.npz: evaluates exactly its frames")
    parser.add_argument("--mapanything_model", default="facebook/map-anything")
    parser.add_argument("--recon_cache", default=None, help="directory of cached reconstructions (read and written)")
    parser.add_argument("--save_vis", action="store_true", help="saves the alignments of the first seed next to the output")
    parser.add_argument("--output", default="outputs/lamar_results.json")
    args = parser.parse_args()

    plan_img, plan_params = load_floorplan(args.floorplan, args.plan_size)
    model = load_model(args.checkpoint_dir, args.dinov3_weights)
    recon_model = load_mapanything(args.mapanything_model)
    if args.recon_cache:
        os.makedirs(args.recon_cache, exist_ok=True)
    vis_dir = os.path.splitext(args.output)[0] + "_vis"
    if args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    split = load_split(args.split)
    sessions = {}
    for key in args.sessions:
        for name in (split[key] if key in split else [key]):
            session = load_session(args.map_dir, None, name)
            kept = test_frames(session, plan_params, args.zfloc_npz) if name in split["test"] else train_frames(session, plan_params)
            sessions[name] = session.subset(kept)
            print(f"{name}: {len(session)} frames, {len(kept)} evaluated")

    results = {}
    for seq_length in args.seq_length:
        records = {seed: {"success": [], "t_errs": [], "r_errs": [], "failed": 0} for seed in args.seeds}
        window_index = 0
        for name, session in sessions.items():
            for window in tqdm(iter_test_windows(len(session), seq_length), desc=f"{name} L={seq_length}"):
                frames = session.subset(np.arange(len(session))[window])
                cache = os.path.join(args.recon_cache, f"{name}_L{seq_length}_s{window.start:05d}.npz") if args.recon_cache else None
                if cache and os.path.exists(cache):
                    recon = load_recon(cache)
                else:
                    recon = reconstruct(frames.image_paths, frames.K, frames.c2w, recon_model)
                    if cache:
                        save_recon(cache, recon)
                for seed in args.seeds:  # the randomness of the alignment (query sampling, RANSAC), per window
                    np.random.seed(seed * 1_000_000 + window_index)
                    torch.manual_seed(seed * 1_000_000 + window_index)
                    cv2.setRNGSeed(seed * 1_000_000 + window_index)
                    alignment = align(recon, plan_img, model, args.ransac_thr)
                    record = records[seed]
                    if alignment is None:
                        record["failed"] += 1
                        record["success"].append(False)
                        continue
                    t_errs, r_errs = window_errors(alignment, recon, frames.c2w, plan_params)
                    record["success"].append(bool(np.all(t_errs < SUCCESS_M)))
                    record["t_errs"] += t_errs.tolist()
                    record["r_errs"] += r_errs.tolist()
                    if args.save_vis and seed == args.seeds[0]:
                        image = draw_alignment(np.array(alignment["density_img"]), np.array(plan_img), alignment["M"], alignment["src_px"])
                        Image.fromarray(image).save(os.path.join(vis_dir, f"{name}_L{seq_length}_s{window.start:05d}.png"))
                window_index += 1
        for seed, record in records.items():
            results[f"{seed}_{seq_length}"] = {
                "success_10": float(np.mean(record["success"])) if record["success"] else float("nan"),
                "n_windows": len(record["success"]), "n_failed": record["failed"],
                "median_t_err_m": float(np.median(record["t_errs"])) if record["t_errs"] else float("nan"),
                "median_r_err_deg": float(np.median(record["r_errs"])) if record["r_errs"] else float("nan"),
            }

    results["mean"] = {str(L): float(np.mean([results[f"{seed}_{L}"]["success_10"] for seed in args.seeds])) for L in args.seq_length}
    print("length | seeds | SR@1m (%) | windows | median error (m)")
    for L in args.seq_length:
        rows = [results[f"{seed}_{L}"] for seed in args.seeds]
        print(f"{L:6d} | {len(rows):5d} | {100 * results['mean'][str(L)]:9.1f} | {rows[0]['n_windows']:7d} | "
              f"{np.mean([r['median_t_err_m'] for r in rows]):16.2f}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
