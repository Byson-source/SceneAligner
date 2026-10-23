"""Unit tests of the numpy helpers of train/lamar/lamar.py (no GPU, no LaMAR data).

    python -m pytest train/lamar/tests
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import lamar  # noqa: E402


def synthetic_plan_params(plan_size=1024):
    """plan_params of a floorplan whose walls are the lines u = 1000 and v = 500 of hge.png."""
    A = lamar.floorplan_affine()
    width, height = lamar.PLAN_FULL_SIZE
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    wall_dist_px = np.minimum(np.abs(u - 1000), np.abs(v - 500)).astype(np.float32)
    scale = plan_size / width
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    off = np.array([(plan_size - new_w) // 2, (plan_size - new_h) // 2], dtype=np.float64)
    return {"A": A, "scale": scale, "off": off, "plan_size": plan_size, "full_size": lamar.PLAN_FULL_SIZE,
            "m_per_px": 1.0 / np.sqrt(np.abs(np.linalg.det(A[:, :2]))), "wall_dist_px": wall_dist_px}


def test_affine_reproduces_the_entrances():
    A = lamar.floorplan_affine()
    px = lamar.world_to_plan_full(lamar.TRAJECTORY_CORRESPONDENCES, A)
    assert np.linalg.norm(px - lamar.FLOORPLAN_CORRESPONDENCES, axis=1).max() < 6.0
    assert np.linalg.det(A[:, :2]) < 0  # top-down view in image coordinates: a reflection
    assert abs(1.0 / np.sqrt(abs(np.linalg.det(A[:, :2]))) - 0.0546) < 0.001  # metres per pixel


def test_plan_round_trips():
    params = synthetic_plan_params()
    points = np.array([[-12.6, 10.1, 1.2], [40.0, -50.0, 0.0]])
    px = lamar.world_to_plan(points, params)
    full = lamar.plan_to_full_res(px, params)
    assert np.allclose(full, lamar.world_to_plan_full(points, params["A"]))
    assert np.allclose(lamar.full_res_to_world(full, params), points[:, :2])


def test_on_wall_uses_full_resolution_distance():
    params = synthetic_plan_params()
    on = lamar.full_res_to_world(np.array([[1000.0, 1200.0], [1003.0, 1200.0]]), params)  # 0 px and 3 px from a wall
    off = lamar.full_res_to_world(np.array([[1100.0, 1200.0]]), params)
    nan = np.array([[np.nan, np.nan]])
    result = lamar.on_wall(np.concatenate([on, off, nan]), params, max_dist_m=0.2)
    assert result.tolist() == [True, True, False, False]


def test_on_plan_keeps_all_points_inside_the_plan():
    params = synthetic_plan_params()
    points = lamar.full_res_to_world(np.array([[1000.0, 1200.0], [1100.0, 1200.0], [-10.0, 1200.0]]), params)  # wall, room, outside
    assert lamar.on_plan(points, params).tolist() == [True, True, False]
    assert lamar.on_plan(points, params, max_wall_dist_m=0.2).tolist() == [True, False, False]
    assert not lamar.on_plan(np.array([[np.nan, np.nan]]), params).any()


def test_indoor_and_windows():
    timestamps = np.array([3302315254, 3302315254 + 100_000, 3302315254 + 200_000, 3373150184 + 1])  # us
    assert lamar.is_indoor(timestamps).tolist() == [True, True, True, False]
    gapped = np.concatenate([np.arange(60) * 500_000, 100_000_000 + np.arange(30) * 500_000])
    assert lamar.train_runs(gapped) == [(0, 60), (60, 30)]
    rng = np.random.default_rng(0)
    windows = lamar.sample_train_windows(gapped, 20, rng, num_frames=(10, 40), max_step=4)
    assert len(windows) == 20
    for w in windows:
        steps = np.diff(w)
        assert 10 <= len(w) <= 40 and steps.min() >= 1 and steps.max() <= 4
        assert (w[-1] < 60) or (w[0] >= 60)  # inside one run
    assert not lamar.sample_train_windows(gapped, 5, rng, num_frames=(70, 100))  # no run is long enough
    seen = set()
    distinct = [w for _ in range(3) for w in lamar.sample_distinct_windows(gapped, 20, rng, seen, num_frames=(10, 40))]
    assert len(distinct) == 60 and len({tuple(w) for w in distinct}) == 60  # different combinations over the epochs
    chunks = lamar.session_chunks(gapped, 25)
    assert [len(c) for c in chunks] == [20, 20, 20, 15, 15]  # runs of 60 and 30 frames
    assert np.array_equal(np.concatenate(chunks), np.arange(90))
    name = lamar.window_name("ios_x_000", np.array([123, 456]))
    assert name.startswith("ios_x_000_t123_n2_") and name != lamar.window_name("ios_x_000", np.array([123, 457]))
    assert lamar.iter_test_windows(23, 10) == [slice(0, 10), slice(10, 20)]


def test_index_shift_and_bundled_positions():
    kept = lamar.kept_indices(10, [0, 3, 7])
    assert kept.tolist() == [0, 1, 2, 7, 8, 9]
    A = lamar.floorplan_affine()
    c2w = np.tile(np.eye(4), (6, 1, 1))
    c2w[:, :2, 3] = np.arange(6)[:, None] * [1.0, 0.5]
    poses_map = lamar.world_to_plan_full(c2w[[1, 2, 4], :3, 3], A) + 0.3
    assert lamar.kept_indices_from_poses_map(poses_map, c2w, A).tolist() == [1, 2, 4]
    with pytest.raises(ValueError):
        lamar.kept_indices_from_poses_map(poses_map[[2, 0]], c2w, A)


def test_poses_and_gravity():
    c2w = np.tile(np.eye(4), (3, 1, 1))
    c2w[:, :3, :3] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]  # camera z forward along world +y, camera -y = world +z
    c2w[:, :3, 3] = np.arange(3)[:, None] * [1.0, 2.0, 0.0]
    rel = lamar.relative_extrinsics(c2w)
    assert np.allclose(rel[0], np.eye(4))
    assert np.allclose(rel[1, :3, 3], c2w[0, :3, :3].T @ (c2w[1, :3, 3] - c2w[0, :3, 3]))
    gravity = lamar.gravity_cam0(c2w[0])
    assert np.allclose(gravity, [0.0, 1.0, 0.0])  # down is +y in the camera
    up = lamar.check_world_up(c2w)
    assert up["camera_up_deg"] < 1e-6
    R = lamar.ground_rotation(gravity)
    assert np.allclose(R @ gravity, [0.0, 1.0, 0.0]) and np.allclose(R @ R.T, np.eye(3))


def test_depth_sampling_and_unprojection():
    K_depth = np.array([[100.0, 0, 32.0], [0, 100.0, 24.0], [0, 0, 1]])  # 64 x 48 depth map
    K_out = np.array([[400.0, 0, 128.0], [0, 400.0, 96.0], [0, 0, 1]])  # 256 x 192 grid of the same camera
    depth = np.full((48, 64), 2.0, dtype=np.float32)
    depth[10, 20] = np.nan
    sampled = lamar.sample_depth_on_grid(depth, K_depth, K_out, (192, 256))
    assert sampled.shape == (192, 256) and np.nanmax(sampled) == 2.0
    assert np.isnan(sampled[40, 80])  # the pixel that projects onto the invalid depth pixel
    points = lamar.unproject(sampled, K_out)
    assert np.allclose(points[96, 128], [0.0, 0.0, 2.0]) and np.isnan(points[40, 80]).all()
    assert np.allclose(points[96, 56], [-72 / 400 * 2.0, 0.0, 2.0])


def test_top_down_projection_matches_scenealigner_layout():
    points = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 2.0], [2.0, 0.5, 1.0]])
    proj = lamar.top_down_projection(points, np.eye(3), density_size=512)
    assert proj["scale"] == pytest.approx(512 * 0.8 / 4.0)
    assert proj["cx"] == pytest.approx(51.2) and proj["cz"] == pytest.approx((512 - 2.0 * proj["scale"]) / 2)
    keep = lamar.filter_ground_points(np.random.default_rng(0).normal(size=(1000, 3)), xz_slice_val=0.0)
    assert 0.7 < keep.mean() <= 0.75


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="scenealigner.preprocess needs torch")
def test_load_floorplan(tmp_path):
    from PIL import Image

    width, height = lamar.PLAN_FULL_SIZE
    plan = np.full((height, width), 255, dtype=np.uint8)
    plan[:, 1000:1002] = 0
    path = tmp_path / "hge.png"
    Image.fromarray(plan).save(path)
    img, params = lamar.load_floorplan(path, plan_size=512)
    assert img.size == (512, 512)
    assert params["wall_dist_px"][100, 1010] == pytest.approx(9.0)
    assert np.array(img.convert("L"))[:, int(1000 * params["scale"] + params["off"][0])].min() < 128  # the wall survived the resize


def test_load_session_with_one_sensor_per_frame(tmp_path):
    """Phone sessions with varying intrinsics have one camera and one depth sensor per frame (run_phone_to_capture)."""
    session = "ios_2022-01-12_15.15.53_000"
    map_dir, raw = tmp_path / "map", tmp_path / "ios_2022-01-12_15.15.53"
    map_dir.mkdir()
    (raw / "raw_data" / "depth").mkdir(parents=True)
    timestamps = [100, 200, 300]
    (map_dir / "trajectories.txt").write_text("# timestamp, device_id, qw, qx, qy, qz, tx, ty, tz\n" + "".join(
        f"{ts}, {session}/cam_phone_{ts}, 1, 0, 0, 0, {i}, 0, 0\n" for i, ts in enumerate(timestamps)))
    (map_dir / "sensors.txt").write_text("".join(
        f"{session}/cam_phone_{ts}, phone camera, camera, PINHOLE, 1440, 1920, {1500 + i}, {1500 + i}, 720, 960\n"
        for i, ts in enumerate(timestamps)))
    (map_dir / "images.txt").write_text("".join(f"{ts}, {session}/cam_phone_{ts}, {session}/images/{ts}.jpg\n" for ts in timestamps))
    (raw / "sensors.txt").write_text("".join(
        f"depth_phone_{ts}, depth, depth, PINHOLE, 192, 256, {200 + i}, {200 + i}, 96, 128\n" for i, ts in enumerate(timestamps[:2])))
    (raw / "depths.txt").write_text("".join(f"{ts}, depth_phone_{ts}, depth/{ts}.png\n" for ts in timestamps[:2]))

    frames = lamar.load_session(map_dir, tmp_path, session)
    assert frames.K[:, 0, 0].tolist() == [1500, 1501, 1502]
    assert frames.image_paths[1] == str(map_dir / "raw_data" / session / "images" / "200.jpg")
    assert frames.depth_paths[2] is None and frames.depth_paths[0].endswith("depth/100.png")
    assert frames.K_depth[1, 0, 0] == 201 and np.isnan(frames.K_depth[2]).all()
    sub = frames.subset(np.array([2, 0]))
    assert sub.K[:, 0, 0].tolist() == [1502, 1500] and sub.depth_paths[0] is None


def test_pose_deviation():
    reference = np.tile(np.eye(4), (2, 1, 1))
    poses = reference.copy()
    angle = np.radians(3.0)
    poses[1, :3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    poses[1, :3, 3] = [0.3, 0.4, 0.0]
    deg, m = lamar.pose_deviation(poses, reference)
    assert np.allclose(deg, [0.0, 3.0]) and np.allclose(m, [0.0, 0.5])


def test_window_from_cached_chunks(tmp_path):
    reconstruction = pytest.importorskip("reconstruction")  # torch
    rng = np.random.default_rng(0)
    S, H, W = 4, 2, 3
    c2w = np.tile(np.eye(4), (S, 1, 1))
    for i in range(S):
        c2w[i, :3, :3] = lamar.ground_rotation(rng.normal(size=3) / np.sqrt(3) + np.array([0, 0, -1]))
        c2w[i, :3, 3] = rng.uniform(-5, 5, 3)
    local = rng.uniform(-2, 2, (S, H, W, 3)).astype(np.float32)  # points of every frame in its camera frame
    mask = rng.uniform(size=(S, H, W)) > 0.3
    timestamps = np.arange(S) * 500_000
    for chunk in (slice(0, 2), slice(2, 4)):  # two chunks, reconstructed in the frame of their first camera
        ext = lamar.relative_extrinsics(c2w[chunk])
        recon = {"points_cam0": local[chunk] @ ext[:, None, :3, :3].transpose(0, 1, 3, 2) + ext[:, None, None, :3, 3],
                 "mask": mask[chunk], "extrinsics": ext, "conf": np.ones((2, H, W), np.float32),
                 "preprocessed_images": np.zeros((2, H, W, 3), np.uint8), "intrinsics": np.tile(np.eye(3), (2, 1, 1)),
                 "pose_error_deg": np.zeros(2), "pose_error_m": np.zeros(2)}
        reconstruction.save_chunk(str(tmp_path / f"t{timestamps[chunk][0]}_n2"), recon, timestamps[chunk], local[chunk])
    cache = reconstruction.SessionCache(str(tmp_path))
    window = [1, 2, 3]  # across the chunks
    recon = reconstruction.window_from_cache(cache, timestamps[window], c2w[window])
    ext = np.linalg.inv(c2w[1]) @ c2w[window]  # camera-to-camera-0 of the window
    expected = local[window] @ ext[:, None, :3, :3].transpose(0, 1, 3, 2) + ext[:, None, None, :3, 3]
    assert np.array_equal(recon["mask"], mask[window])
    assert np.abs(recon["points_cam0"] - np.where(mask[window, ..., None], expected, 0)).max() < 0.02  # float16
    assert np.abs(recon["lidar_cam0"] - expected).max() < 0.02
    assert np.allclose(recon["est_gravity_cam"], lamar.gravity_cam0(c2w[1]))
