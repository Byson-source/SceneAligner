"""Unit tests of the scale-band RANSAC (train/lamar/ransac.py) and of the Z-FLoc inputs (train/lamar/evaluate_zfloc.py).

    python -m pytest train/lamar/tests
"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lamar  # noqa: E402
from ransac import estimate_similarity_ransac, scale_band, similarity_matrix  # noqa: E402

TRUE_M = similarity_matrix(0.4 * np.exp(0.3j), 300 + 200j)  # density px -> plan px, scale 0.4
BAND = scale_band(1 / 0.02, 1 / 0.055)  # Z-FLoc HGE: BEV 0.02 m/px, floorplan 0.055 m/px


def transform(M: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ M[:, :2].T + M[:, 2]


def test_scale_band_of_zfloc_hge():
    assert np.allclose(BAND, (0.2364, 0.4909), atol=1e-4)  # zfloc/config.py: "HGE [0.236, 0.491]"


def test_recovers_the_similarity():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 512, (200, 2))
    dst = transform(TRUE_M, src) + rng.normal(0, 0.5, (200, 2))
    outliers = rng.uniform(0, 512, (300, 2))
    src, dst = np.vstack((src, outliers)), np.vstack((dst, rng.uniform(0, 1024, (300, 2))))
    M = estimate_similarity_ransac(src, dst, 10.0, BAND, np.random.default_rng(1))
    assert np.allclose(M[:, :2], TRUE_M[:, :2], atol=2e-3) and np.allclose(M[:, 2], TRUE_M[:, 2], atol=1.0)


def test_rejects_a_larger_cluster_of_the_wrong_scale():
    rng = np.random.default_rng(0)
    src_true, src_wrong = rng.uniform(0, 512, (100, 2)), rng.uniform(0, 512, (150, 2))
    wrong_M = similarity_matrix(1.0 * np.exp(-1.0j), 100 + 50j)  # scale 1: outside the band
    src = np.vstack((src_true, src_wrong))
    dst = np.vstack((transform(TRUE_M, src_true), transform(wrong_M, src_wrong)))
    cv2.setRNGSeed(0)
    M_cv2, _ = cv2.estimateAffinePartial2D(src.astype(np.float32), dst.astype(np.float32), method=cv2.RANSAC,
                                          ransacReprojThreshold=10.0)
    assert np.allclose(M_cv2, wrong_M, atol=1e-2)  # unrestricted: the larger cluster
    M = estimate_similarity_ransac(src, dst, 10.0, BAND, np.random.default_rng(1))
    assert np.allclose(M, TRUE_M, atol=1e-3)


def test_none_without_hypothesis_in_the_band():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 512, (50, 2))
    assert estimate_similarity_ransac(src, src * 2.0, 10.0, BAND, rng) is None
    assert estimate_similarity_ransac(src[:1], src[:1], 10.0, BAND, rng) is None


def test_refit_scale_is_clipped_to_the_band():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 512, (100, 2))
    dst = transform(similarity_matrix(0.5 + 0j, 0j), src)  # scale 0.5, just above the band (0.491)
    M = estimate_similarity_ransac(src, dst, 20.0, BAND, rng)  # hypotheses in the band only with a large threshold
    assert M is None or np.isclose(np.sqrt(abs(np.linalg.det(M[:, :2]))), BAND[1])


def test_zfloc_window_in_time_order(tmp_path):
    evaluate_zfloc = pytest.importorskip("evaluate_zfloc")  # torch, scenealigner
    rng = np.random.default_rng(0)
    L = 12
    c2w = np.tile(np.eye(4), (L, 1, 1))
    c2w[:, :3, 3] = rng.uniform(-5, 5, (L, 3))
    poses_world = np.linalg.inv(c2w[0]) @ c2w  # as poses_world.npy: relative to the first frame of the session
    path = tmp_path / f"{evaluate_zfloc.window_stem(0, L)}.npz"
    np.savez(path, all_pcds=rng.normal(size=(100, 3)), gt_camera_poses=np.concatenate((poses_world[-1:], poses_world[:-1])))
    recon = evaluate_zfloc.load_zfloc_window(path, c2w)
    assert path.name == "00011-0.npz"
    assert np.allclose(recon["extrinsics"], np.linalg.inv(c2w[-1]) @ c2w)
    assert np.allclose(recon["est_gravity_cam"], lamar.gravity_cam0(c2w[-1]))
    with pytest.raises(ValueError):
        evaluate_zfloc.load_zfloc_window(path, c2w[::-1])
