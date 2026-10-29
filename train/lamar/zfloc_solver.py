"""Localization from the SceneAligner correspondences with the back end of Z-FLoc.

Every pair of the N correspondences (density map -> floorplan) gives a similarity hypothesis (nC2 in all). They go
through the tail of the 2C solver of Z-FLoc (zfloc/matching/solvers.py, solve_similarity_2circles): scale band,
coarse score on the wall lines of the BEV (top candidate_num), alignment/penetration score with the scale penalty,
then the LM refinement of the best one (zfloc/localization.py, localize_frame). The BEV, its wall lines and its gravity
come from Z-FLoc tools/metadata_generation/extract_primitives.py; the density map of SceneAligner is made with the same
gravity, so that its pixels map exactly to the BEV pixels.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

BEV_M_PER_PX = 0.02  # resolution_ref of Z-FLoc (make_bev_cuda_dual)
CHUNK = 200_000  # hypotheses coarse-scored at once (memory)


def import_zfloc(zfloc_repo: str | Path) -> None:
    """Puts the Z-FLoc repository on sys.path (zfloc.matching needs numba, scipy, pandas)."""
    if str(zfloc_repo) not in sys.path:
        sys.path.insert(0, str(zfloc_repo))


def match_param() -> dict:
    """Matching parameters of Z-FLoc for HGE (zfloc/config.py)."""
    from zfloc.config import HGE
    return dict(HGE.match_param)


def load_bev(scene_dir: str | Path, stem: str) -> dict | None:
    """
    Primitives of one window from extract_primitives.py.
    Args:
        scene_dir (str | Path): <pcd_root>/hge_<L>/<scene>
        stem (str): window stem, e.g. "00019-0"

    Returns:
        dict | None: "bev_img" [H, W] uint8 (walls 0), "occupancy_mask" [H, W] uint8, "lines" [K, 2, 2] (x, y BEV px),
            "gravity" [3] (camera frame of the last frame, towards the floor); None without primitives (empty point
            cloud or no wall line: a failure, as in Z-FLoc)
    """
    scene_dir = Path(scene_dir)
    paths = {"bev_img": scene_dir / "bev_img" / f"{stem}.png", "occupancy_mask": scene_dir / "occupancy_mask" / f"{stem}.png",
             "lines": scene_dir / "lines" / f"{stem}.npy", "gravity": scene_dir / "gravity" / f"{stem}.npy"}
    if not all(p.exists() for p in paths.values()):
        return None
    bev = {k: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) if p.suffix == ".png" else np.load(p) for k, p in paths.items()}
    return bev


def setup_refs(floorplan: str | Path) -> dict:
    """
    The floorplan of Z-FLoc (hge.png, grayscale) and its distance transform (zfloc/localization.py, setup_matching).
    Args:
        floorplan (str | Path): hge.png

    Returns:
        dict: "src_map_np" [H, W] uint8, "src_dist_map" [H, W] float32, "src_dist_indices" [H, W, 2]
    """
    from zfloc.matching.geometry import setup_distance_map
    src_map_np = cv2.imread(str(floorplan), cv2.IMREAD_GRAYSCALE)
    src_dist_map, src_dist_indices = setup_distance_map(src_map_np, occ_map_option=False)
    return {"src_map_np": src_map_np, "src_dist_map": src_dist_map, "src_dist_indices": src_dist_indices}


def ground_to_bev(gravity: np.ndarray, R_rect: np.ndarray) -> np.ndarray:
    """
    Linear map of the horizontal ground coordinates of SceneAligner to the BEV coordinates of Z-FLoc (metres).
    Args:
        gravity (np.ndarray): tensor with shape [3], the gravity of both (camera frame)
        R_rect (np.ndarray): tensor with shape [3, 3], camera -> ground of SceneAligner (+Y along the gravity)

    Returns:
        np.ndarray: tensor with shape [2, 2], (x, z) of the ground frame -> (x, y) of the BEV
    """
    from zfloc.poses import align_vectors
    rot2bev = align_vectors(np.array([0.0, 0.0, 1.0]), gravity)  # extract_primitives: BEV z along the gravity
    M = rot2bev.T @ R_rect.T  # ground -> BEV; the ground y (gravity) goes to the BEV z
    return M[:2][:, [0, 2]]


def density_to_bev(px: np.ndarray, proj_params: dict, gravity: np.ndarray, bev_shape: tuple) -> np.ndarray:
    """
    Density map px of SceneAligner -> BEV px of Z-FLoc (inverse of get_density_coordinates, then the grid of
    make_bev_cuda_dual: col = x / 0.02 + W // 2, row = y / 0.02 + H // 2).
    Args:
        px (np.ndarray): tensor with shape [N, 2], density map px (u, v)
        proj_params (dict): of reconstruct_projection_geometry with the same gravity
        gravity (np.ndarray): tensor with shape [3], gravity of the BEV (camera frame of the last frame)
        bev_shape (tuple): (H, W) of the BEV

    Returns:
        np.ndarray: tensor with shape [N, 2], BEV px (x, y)
    """
    x = (px[:, 0] - proj_params["cx"]) / proj_params["scale"] + proj_params["min_x"]
    z = (proj_params["density_size"] - 1 - px[:, 1] - proj_params["cz"]) / proj_params["scale"] + proj_params["min_z"]
    xy = np.stack([x, z], axis=1) @ ground_to_bev(gravity, proj_params["R_rect"]).T  # metres in the BEV
    return xy / BEV_M_PER_PX + np.array([bev_shape[1] // 2, bev_shape[0] // 2])


def plan_to_full(px: np.ndarray, plan_params: dict) -> np.ndarray:
    """Resized floorplan px (N, 2) -> hge.png px (N, 2)."""
    return (px - plan_params["off"]) / plan_params["scale"]


def pair_hypotheses(src: np.ndarray, dst: np.ndarray, param: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Similarity hypotheses of all the pairs of correspondences, within the scale band of Z-FLoc.
    Args:
        src (np.ndarray): tensor with shape [N, 2], BEV px
        dst (np.ndarray): tensor with shape [N, 2], hge.png px
        param (dict): match_param

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: scales [H], rotations [H, 2, 2], translations [H, 2] (BEV -> hge.png)
    """
    from zfloc.matching.geometry import estimate_similarity, scale_bounds
    i, j = np.triu_indices(len(src), k=1)
    s, R, t, valid = estimate_similarity(src[i], src[j], dst[i], dst[j])
    scale_min, scale_max = scale_bounds(param)
    keep = (s > scale_min) & (s < scale_max) & valid  # as in solve_similarity_2circles
    return s[keep], R[keep], t[keep]


def localize(src: np.ndarray, dst: np.ndarray, bev: dict, refs: dict, param: dict) -> tuple | None:
    """
    Similarity BEV -> hge.png from the correspondences with the back end of Z-FLoc.
    Args:
        src (np.ndarray): tensor with shape [N, 2], BEV px
        dst (np.ndarray): tensor with shape [N, 2], hge.png px
        bev (dict): of load_bev
        refs (dict): of setup_refs
        param (dict): match_param

    Returns:
        tuple | None: (s, R [2, 2], t [2]) after the LM refinement, or None without a valid hypothesis
    """
    from zfloc.matching import refine_with_lm
    from zfloc.matching.geometry import ParallelError, setup_distance_map
    from zfloc.matching.hypothesis import prune_ref_lines
    from zfloc.matching.solvers import _filter_and_score_candidates, _select_topk_point_candidates

    s, R, t = pair_hypotheses(src, dst, param)
    if len(s) == 0:
        return None
    src_map, src_dist_map = refs["src_map_np"], refs["src_dist_map"]
    h, w = src_map.shape[:2]
    lines = prune_ref_lines(bev["lines"], param)

    # Coarse score (inliers of the points sampled on the BEV wall lines): top candidate_num of every chunk, then overall
    tops = [_select_topk_point_candidates(lines, s[k:k + CHUNK], R[k:k + CHUNK], t[k:k + CHUNK], src_dist_map, w, h, param)
            for k in range(0, len(s), CHUNK)]
    s, R, t = (np.concatenate(x) for x in zip(*tops))
    if len(tops) > 1:
        s, R, t = _select_topk_point_candidates(lines, s, R, t, src_dist_map, w, h, param)

    # Alignment / penetration score with the scale penalty (keypoint solvers: harsh_scale_penalty=False), then LM
    ref_dist_map = setup_distance_map(bev["occupancy_mask"], occ_map_option=True)
    try:
        S, R, t, score, _ = _filter_and_score_candidates(s, R, t, bev["bev_img"], src_map, src_dist_map, ref_dist_map,
                                                         param, harsh_scale_penalty=False)
    except ParallelError:  # no hypothesis inside the floorplan and off the walls (localize_frame skips the solver)
        return None
    best = int(np.argmax(score))
    return refine_with_lm(bev["bev_img"], src_map, (S[best], R[best], t[best]), src_dist_map, refs["src_dist_indices"])


def window_errors(model: tuple, bev: dict, c2w: np.ndarray, poses_map: np.ndarray, plan_params: dict, param: dict,
                  last_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Errors of the last cameras of a window with the metric of Z-FLoc (zfloc/offline.py).
    Args:
        model (tuple): (s, R, t), BEV px -> hge.png px
        bev (dict): of load_bev
        c2w (np.ndarray): tensor with shape [L, 4, 4], ground-truth camera-to-world poses of the window
        poses_map (np.ndarray): tensor with shape [L, 2], ground-truth positions of the window, hge.png px
        plan_params (dict): of lamar.load_floorplan (world -> hge.png affine "A")
        param (dict): match_param (resolution_src: m/px of hge.png in the metric)
        last_frames (int): evaluated cameras

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: position errors [m] and angular errors [deg] of the last cameras,
            and their estimated positions [last_frames, 2] (hge.png px)
    """
    from zfloc.localization import last_frame_pose_estimation
    from zfloc.poses import align_vectors
    s, R, t = model
    L = len(c2w)
    bev_px = last_frame_pose_estimation(bev["bev_img"], c2w, L - 1, BEV_M_PER_PX, num_frames=last_frames,
                                        gravity=bev["gravity"])
    est = s * bev_px @ R.T + t
    t_errs = np.linalg.norm(est - poses_map[L - last_frames:], axis=1) * param["resolution_src"]

    # Viewing directions (camera z-axes): world -> BEV as in last_frame_pose_estimation, then R; ground truth by the affine
    rot = align_vectors(np.array([0.0, 0.0, 1.0]), bev["gravity"]).T @ c2w[-1, :3, :3].T  # world -> BEV
    forward = c2w[L - last_frames:, :3, 2]
    forward_est = (forward @ rot.T)[:, :2] @ R.T
    forward_gt = forward[:, :2] @ plan_params["A"][:, :2].T
    angle = np.arctan2(forward_est[:, 1], forward_est[:, 0]) - np.arctan2(forward_gt[:, 1], forward_gt[:, 0])
    r_errs = np.abs(np.degrees(np.arctan2(np.sin(angle), np.cos(angle))))
    return t_errs, r_errs, est


def overlay(bev: dict, refs: dict, model: tuple, est: np.ndarray, gt: np.ndarray, src: np.ndarray,
            dst: np.ndarray) -> np.ndarray:
    """
    Z-FLoc overlay (zfloc/offline.py: BEV warped onto hge.png, estimated last positions green, ground truth blue) with
    the correspondences (small dots, BEV end mapped by the model -> hge.png end).
    Args:
        bev (dict): of load_bev
        refs (dict): of setup_refs
        model (tuple): (s, R, t)
        est (np.ndarray): tensor with shape [K, 2], estimated positions, hge.png px
        gt (np.ndarray): tensor with shape [K, 2], ground-truth positions, hge.png px
        src (np.ndarray): tensor with shape [N, 2], correspondences, BEV px
        dst (np.ndarray): tensor with shape [N, 2], correspondences, hge.png px

    Returns:
        np.ndarray: BGR image of the size of hge.png
    """
    from zfloc.matching.solvers import overlay_similarity
    s, R, t = model
    image = overlay_similarity(cv2.cvtColor(bev["bev_img"], cv2.COLOR_GRAY2BGR),
                               cv2.cvtColor(refs["src_map_np"], cv2.COLOR_GRAY2BGR), s, R, t)
    for a, b in zip(s * src @ R.T + t, dst):  # orange: where the correspondence lands, line to its floorplan end
        cv2.line(image, tuple(int(v) for v in a), tuple(int(v) for v in b), (0, 165, 255), 1)
        cv2.circle(image, tuple(int(v) for v in b), 3, (0, 165, 255), -1)
    for e, g in zip(est, gt):
        cv2.circle(image, (int(e[0]), int(e[1])), 10, (0, 255, 0), -1)
        cv2.circle(image, (int(g[0]), int(g[1])), 10, (255, 0, 0), -1)
    return image
