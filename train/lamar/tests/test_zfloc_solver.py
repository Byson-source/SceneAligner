"""Unit tests of the Z-FLoc back end on the SceneAligner correspondences (train/lamar/zfloc_solver.py).

Needs the Z-FLoc repository (ZFLOC_REPO, default ../z-floc next to this repository) with numba and scipy.

    python -m pytest train/lamar/tests
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lamar  # noqa: E402
import zfloc_solver  # noqa: E402

ZFLOC_REPO = Path(os.environ.get("ZFLOC_REPO", Path(__file__).resolve().parents[4] / "z-floc"))
if not (ZFLOC_REPO / "zfloc").is_dir():
    pytest.skip(f"Z-FLoc repository not found: {ZFLOC_REPO}", allow_module_level=True)
pytest.importorskip("numba")
pytest.importorskip("scipy")
zfloc_solver.import_zfloc(ZFLOC_REPO)
PARAM = zfloc_solver.match_param()


def test_density_px_to_bev_px():
    from zfloc.poses import align_vectors
    rng = np.random.default_rng(0)
    gravity = np.array([0.1, 0.95, -0.2])  # camera frame, towards the floor
    points = rng.uniform(-4, 4, (500, 3))

    # SceneAligner: ground frame, top-down projection, density map px (get_density_coordinates)
    R_rect = lamar.ground_rotation(gravity)
    ground = points @ R_rect.T
    params = lamar.top_down_projection(ground, R_rect)
    u = (ground[:, 0] - params["min_x"]) * params["scale"] + params["cx"]
    v = params["density_size"] - 1 - ((ground[:, 2] - params["min_z"]) * params["scale"] + params["cz"])

    # Z-FLoc: extract_primitives (rot2bev) and make_bev_cuda_dual (0.02 m/px, centre W // 2, H // 2)
    shape = (801, 600)
    bev = points @ align_vectors(np.array([0.0, 0.0, 1.0]), gravity)  # rot2bev.T @ p
    expected = bev[:, :2] / 0.02 + np.array([shape[1] // 2, shape[0] // 2])
    assert np.allclose(zfloc_solver.density_to_bev(np.stack([u, v], 1), params, gravity, shape), expected, atol=1e-6)


def test_pair_hypotheses_keep_the_truth_in_the_band():
    rng = np.random.default_rng(0)
    s, theta, t = 0.36, 0.7, np.array([400.0, 250.0])
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    src = rng.uniform(0, 800, (30, 2))
    dst = s * src @ R.T + t
    dst[20:] = rng.uniform(0, 3000, (10, 2))  # outliers
    S, Rs, ts = zfloc_solver.pair_hypotheses(src, dst, PARAM)
    assert ((S > 0.236) & (S < 0.491)).all()
    assert len(S) < 30 * 29 // 2
    assert np.sum(np.isclose(S, s, rtol=1e-6) & np.all(np.isclose(ts, t, atol=1e-3), axis=1)) == 20 * 19 // 2


def synthetic_window(rng: np.random.Generator, model: tuple):
    """A floorplan with rooms, the BEV of its walls seen through model (BEV -> plan), and the BEV wall lines."""
    plan = np.full((700, 900), 255, np.uint8)
    segments = [(100, 100, 800, 100), (800, 100, 800, 600), (800, 600, 100, 600), (100, 600, 100, 100),
                (350, 100, 350, 400), (100, 400, 600, 400), (600, 250, 800, 250), (550, 400, 550, 600)]
    for x1, y1, x2, y2 in segments:
        cv2.line(plan, (x1, y1), (x2, y2), 0, 2)
    s, R, t = model
    shape = (1600, 2000)
    A = np.hstack([s * R, t[:, None]])  # BEV px -> plan px
    bev = cv2.warpAffine(plan, A, shape[::-1], flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=255)
    bev = np.where(bev < 120, 0, 255).astype(np.uint8)
    lines = []
    for x1, y1, x2, y2 in segments:
        ends = (np.array([[x1, y1], [x2, y2]], float) - t) @ R / s  # plan px -> BEV px
        if ((ends >= 0) & (ends < shape[::-1])).all():
            lines.append(ends)
    occupancy = np.full(shape, 255, np.uint8)  # everything observed
    return plan, {"bev_img": bev, "occupancy_mask": occupancy, "lines": np.array(lines), "gravity": np.array([0, 1.0, 0])}


@pytest.mark.parametrize("chunk", [zfloc_solver.CHUNK, 300])  # 300: coarse score in chunks
def test_localize_recovers_the_similarity_with_outliers(tmp_path, monkeypatch, chunk):
    monkeypatch.setattr(zfloc_solver, "CHUNK", chunk)
    rng = np.random.default_rng(0)
    s, theta = 0.3636, 0.4
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    t = np.array([450.0, 250.0]) - s * R @ np.array([1000.0, 800.0])  # BEV centre -> a free point of the plan
    plan, bev = synthetic_window(rng, (s, R, t))
    cv2.imwrite(str(tmp_path / "hge.png"), plan)
    refs = zfloc_solver.setup_refs(tmp_path / "hge.png")

    walls = np.argwhere(bev["bev_img"] == 0)[:, ::-1].astype(float)
    src = walls[rng.choice(len(walls), 60, replace=False)]
    dst = s * src @ R.T + t + rng.normal(0, 3, src.shape)  # 3 px noise
    dst[15:] = rng.uniform((0, 0), (900, 700), (45, 2))  # 75 % outliers
    est = zfloc_solver.localize(src, dst, bev, refs, PARAM)
    assert est is not None
    s_est, R_est, t_est = est
    corners = np.array([[0, 0], [2000, 0], [0, 1600], [2000, 1600]], float)
    err = np.linalg.norm(s_est * corners @ R_est.T + t_est - (s * corners @ R.T + t), axis=1)
    assert err.max() < 5.0, err  # plan px


def test_localize_without_hypothesis_in_the_band(tmp_path):
    rng = np.random.default_rng(0)
    R = np.eye(2)
    plan, bev = synthetic_window(rng, (0.3636, R, np.array([100.0, 0.0])))
    cv2.imwrite(str(tmp_path / "hge.png"), plan)
    src = rng.uniform(0, 800, (10, 2))
    assert zfloc_solver.localize(src, 2.0 * src, bev, zfloc_solver.setup_refs(tmp_path / "hge.png"), PARAM) is None
