"""LaMAR HGE: CAPTURE files, the UnLoc split, the floorplan and its world-to-pixel transform, frame filters,
windows of consecutive frames and the LiDAR depth maps of the phone sessions.

    python train/lamar/lamar.py --check --map_dir <HGE/sessions/map> --raw_dir <raw sessions> --floorplan hge.png

Conventions: trajectories.txt stores camera-to-world poses (OpenCV camera axes), the world frame is metric with
+z up. The floorplan hge.png (3278 x 2334 px, walls black) is related to the world by a 2D affine transform fitted
on four building entrances (UnLoc, Appendix A.1.2), as in Z-FLoc.
"""
import argparse
import io
import os
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scenealigner package (imported where needed)

PLAN_FULL_SIZE = (3278, 2334)  # (width, height) of hge.png
METRES_PER_PIXEL = 0.055  # of hge.png, the value Z-FLoc uses for its errors (the affine below gives 0.0546)
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0])  # the world frame of LaMAR is z-up
SEQ_LENGTHS = (15, 20, 35, 50, 100)  # window lengths of the Z-FLoc evaluation
TRAIN_FRAMES = (50, 500)  # frames of a training window, drawn at random (C3 reconstructs 50 to 150 random photos)
TRAIN_MAX_STEP = 4  # intervals between the frames of a training window: random in 1..TRAIN_MAX_STEP
DEPTH_SCALE = 1000.0  # LaMAR depth maps: 16-bit PNG in millimetres
HGE_INTRINSICS = np.array([[1596.080444, 0.0, 716.836365], [0.0, 1596.080444, 939.086121], [0.0, 0.0, 1.0]])  # of the 1440 x 1920 photos (Z-FLoc)

# Four building entrances in the floorplan [px] and in the LaMAR trajectories [m] (UnLoc tools/01_create_lamar_hge.py,
# Z-FLoc tools/lamar_setup/create_lamar_hge.py)
FLOORPLAN_CORRESPONDENCES = np.array([[1516.0, 490.5], [1532.0, 2162.0], [350.0, 1515.0], [391.0, 135.0]])
TRAJECTORY_CORRESPONDENCES = np.array([[-12.600, 10.103], [77.981, 9.882], [42.641, -54.348], [-33.099, -51.229]])

# Timestamp intervals of the indoor and outdoor parts of the sessions (UnLoc tools/01_create_lamar_hge.py)
INDOOR_INTERVALS = [
    (10283474528, 10627820197), (3302315254, 3373150184), (3656323308, 3849919589), (3918622077, 3960020150),
    (78222688174, 78428312296), (1543118709, 1589664520), (2711923398, 2997528776), (3659211240, 3790719011),
    (3927574368, 3961675755), (176666751867, 176945375158), (179572739726, 179744144649), (11200002113, 11308653298),
    (11413422848, 11458735800), (130076928497, 130169603527), (130258047052, 130333096575), (18044116869, 18220819706),
    (19050491968, 19235407759), (19297512805, 19344291501), (19832338000, 19950450833), (150988929516, 151153005932),
    (151254926780, 151298457171),
]
OUTDOOR_INTERVALS = [
    (2377234416, 2613861694), (43545496077, 43574965980), (9271751048, 9633470578), (3373749916, 3476720414),
    (3639114393, 3655723578), (3850519320, 3918022346), (3960669859, 3976479425), (78155268579, 78222288354),
    (3862736754, 3926974637), (176611626918, 176666152140), (179420826024, 179572323250), (11309253029, 11412823118),
    (130170203258, 130257447322), (19235824236, 19296913078), (151153405753, 151254327050),
]

# Frames of the test sessions kept by Z-FLoc (zfloc/config.py): [start, end, restart] keeps [start:end] + [restart:]
HGE_INDEX_SHIFT = {
    "ios_2022-07-01_15.45.08_000": [0, 327, 436],
    "ios_2022-07-03_16.00.37_000": [0, 319, 537],
    "ios_2022-01-18_17.10.39_000": [37, 442, 584],
}
SINGLE_FOCAL_MODELS = ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE")


# ----------------------------------------------------------------------------------------------------------------------
# CAPTURE files


def read_csv(path: str | Path) -> list[list[str]]:
    """
    Rows of a CAPTURE text file: comma-separated values, comments start with #.
    Args:
        path (str | Path): text file

    Returns:
        list[list[str]]: stripped fields of every data row, in file order
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        rows.append([field.strip() for field in line.split(",")])
    return rows


def load_split(path: str | Path) -> dict[str, list[str]]:
    """
    Session names of the train, val and test splits (split.yaml of UnLoc).
    Args:
        path (str | Path): YAML file with the keys train, val, test

    Returns:
        dict[str, list[str]]: split -> session names (ios_<date>_<time>_<NNN>)
    """
    with open(path) as f:
        return {split: list(sessions) for split, sessions in yaml.safe_load(f).items()}


def raw_session_name(session: str) -> str:
    """
    Raw phone session of a split session: the name without its sub-sequence suffix.
    Args:
        session (str): e.g. ios_2022-01-20_16.52.33_001

    Returns:
        str: e.g. ios_2022-01-20_16.52.33
    """
    return session.rsplit("_", 1)[0]


def load_sensors(path: str | Path) -> dict[str, dict]:
    """
    Camera-like sensors of sensors.txt: rows sensor_id, name, sensor_type, model, width, height, params...
    Args:
        path (str | Path): sensors.txt

    Returns:
        dict[str, dict]: sensor_id -> {"name", "type", "model", "width", "height", "K"}; K is the 3x3 intrinsic
            matrix from the first parameters (f, cx, cy or fx, fy, cx, cy), distortion is ignored
    """
    sensors = {}
    for row in read_csv(path):
        sensor_id, name, sensor_type = row[:3]
        if len(row) < 9:  # not a camera (wifi, bluetooth, imu...)
            continue
        model, width, height = row[3], int(row[4]), int(row[5])
        params = [float(p) for p in row[6:]]
        if model in SINGLE_FOCAL_MODELS:
            fx, fy, cx, cy = params[0], params[0], params[1], params[2]
        else:
            fx, fy, cx, cy = params[:4]
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        sensors[sensor_id] = {"name": name, "type": sensor_type, "model": model, "width": width, "height": height, "K": K}
    return sensors


def load_records(path: str | Path, prefix: str = "") -> dict[int, tuple[str, str]]:
    """
    Records of images.txt or depths.txt: rows timestamp, sensor_id, path (relative to raw_data/).
    Args:
        path (str | Path): images.txt or depths.txt
        prefix (str): only the sensors whose id starts with it (a sub-session of the map session)

    Returns:
        dict[int, tuple[str, str]]: timestamp [us] -> (sensor_id, relative path)
    """
    return {int(row[0]): (row[1], row[2]) for row in read_csv(path) if row[1].startswith(prefix)}


def pose_from_qt(qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Rigid transform of a CAPTURE pose row (quaternion w, x, y, z and translation).
    Args:
        qw, qx, qy, qz (float): unit quaternion
        tx, ty, tz (float): translation [m]

    Returns:
        np.ndarray: 4x4 transform [R | t]
    """
    rotation = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ])
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = rotation, (tx, ty, tz)
    return pose


def load_trajectory(path: str | Path, session: str) -> tuple[np.ndarray, list, np.ndarray]:
    """
    Camera-to-world poses of one session in trajectories.txt (rows timestamp, device_id, qw, qx, qy, qz, tx, ty, tz).
    Args:
        path (str | Path): trajectories.txt of HGE/sessions/map
        session (str): session name, the prefix of device_id

    Returns:
        tuple[np.ndarray, list, np.ndarray]: timestamps [us] with shape [N] (strictly increasing), the N device ids
            (the camera sensor of each frame: phones with varying intrinsics have one sensor per frame,
            cam_phone_<timestamp>) and camera-to-world poses with shape [N, 4, 4] in the LaMAR world frame [m]
    """
    timestamps, device_ids, c2w = [], [], []
    for row in read_csv(path):
        if not row[1].startswith(session):
            continue
        timestamps.append(int(row[0]))
        device_ids.append(row[1])
        c2w.append(pose_from_qt(*[float(v) for v in row[2:9]]))
    if not timestamps:
        raise ValueError(f"{path}: no poses of {session}")
    timestamps = np.array(timestamps)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError(f"{path}: the timestamps of {session} are not increasing")
    return timestamps, device_ids, np.stack(c2w)


# ----------------------------------------------------------------------------------------------------------------------
# Floorplan


def fit_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Least-squares 2D affine transform mapping src points onto dst points (Z-FLoc fit_affine).
    Args:
        src (np.ndarray): tensor with shape [M, 2], world positions [m]
        dst (np.ndarray): tensor with shape [M, 2], floorplan positions [px]

    Returns:
        np.ndarray: affine matrix with shape [2, 3], dst = A[:, :2] @ src + A[:, 2]
    """
    design = np.hstack([src, np.ones((len(src), 1))])  # [M, 3]
    return np.linalg.lstsq(design, dst, rcond=None)[0].T


def floorplan_affine() -> np.ndarray:
    """
    World [m] -> hge.png pixels transform from the four entrances.

    Returns:
        np.ndarray: affine matrix with shape [2, 3], about 18.3 px/m with a reflection (det < 0)
    """
    return fit_affine(TRAJECTORY_CORRESPONDENCES, FLOORPLAN_CORRESPONDENCES)


def world_to_plan_full(points: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    World coordinates -> pixels of the full-resolution floorplan.
    Args:
        points (np.ndarray): tensor with shape [..., 2 or 3], x and y in metres (z is ignored)
        A (np.ndarray): affine matrix with shape [2, 3]

    Returns:
        np.ndarray: tensor with shape [..., 2], (u, v) pixels of hge.png
    """
    return points[..., :2] @ A[:, :2].T + A[:, 2]


def load_floorplan(path: str | Path, plan_size: int = 1024, wall_dilate_m: float = 0.5) -> tuple[Image.Image, dict]:
    """
    The floorplan resized for SceneAligner and the parameters of its coordinate transforms.

    The walls of hge.png are 1 to 3 px thick and would vanish when the 3278 px plan is resized: they are dilated to
    wall_dilate_m before resizing. The wall test (on_wall) uses the original walls.
    Args:
        path (str | Path): hge.png, 3278 x 2334 px, walls black (< 128)
        plan_size (int): side of the square, white-padded plan given to the network
        wall_dilate_m (float): thickness of the walls in the resized plan [m]

    Returns:
        tuple[Image.Image, dict]: RGB plan of plan_size x plan_size px and plan_params with the keys
            "A" ([2, 3] world -> full-res px), "scale" and "off" (full-res px -> resized px: px * scale + off),
            "plan_size", "full_size" ((width, height) of hge.png), "m_per_px" (of hge.png) and
            "wall_dist_px" ([height, width] float32, distance of every full-res pixel to the nearest wall pixel)
    """
    from scenealigner.preprocess import resize_plan  # imports pi3 and GeoCalib: not needed by the pure numpy helpers

    full = np.array(Image.open(path).convert("L"))
    if full.shape[::-1] != PLAN_FULL_SIZE:
        raise ValueError(f"{path}: expected {PLAN_FULL_SIZE} (width, height), got {full.shape[::-1]}")
    A = floorplan_affine()
    m_per_px = 1.0 / np.sqrt(np.abs(np.linalg.det(A[:, :2])))  # the affine is a similarity up to the fit error
    walls = (full < 128).astype(np.uint8)
    wall_dist_px = cv2.distanceTransform(1 - walls, cv2.DIST_L2, 5).astype(np.float32)
    kernel = int(np.ceil(wall_dilate_m / m_per_px))
    thick = cv2.dilate(walls, np.ones((kernel, kernel), np.uint8))
    plan_img, (scale, off_x, off_y) = resize_plan(Image.fromarray(255 - 255 * thick).convert("RGB"), plan_size)
    plan_params = {"A": A, "scale": scale, "off": np.array([off_x, off_y], dtype=np.float64), "plan_size": plan_size,
                   "full_size": PLAN_FULL_SIZE, "m_per_px": m_per_px, "wall_dist_px": wall_dist_px}
    return plan_img, plan_params


def world_to_plan(points: np.ndarray, plan_params: dict) -> np.ndarray:
    """
    World coordinates -> pixels of the resized floorplan.
    Args:
        points (np.ndarray): tensor with shape [..., 2 or 3], metres
        plan_params (dict): of load_floorplan

    Returns:
        np.ndarray: tensor with shape [..., 2], (u, v) pixels of the plan_size x plan_size plan
    """
    return world_to_plan_full(points, plan_params["A"]) * plan_params["scale"] + plan_params["off"]


def plan_to_full_res(px: np.ndarray, plan_params: dict) -> np.ndarray:
    """
    Pixels of the resized floorplan -> pixels of hge.png.
    Args:
        px (np.ndarray): tensor with shape [..., 2]
        plan_params (dict): of load_floorplan

    Returns:
        np.ndarray: tensor with shape [..., 2]
    """
    return (px - plan_params["off"]) / plan_params["scale"]


def full_res_to_world(px: np.ndarray, plan_params: dict) -> np.ndarray:
    """
    Pixels of hge.png -> world x, y [m] (inverse of the affine).
    Args:
        px (np.ndarray): tensor with shape [..., 2]
        plan_params (dict): of load_floorplan

    Returns:
        np.ndarray: tensor with shape [..., 2]
    """
    A_inv = np.linalg.inv(np.vstack([plan_params["A"], [0.0, 0.0, 1.0]]))[:2]
    return px @ A_inv[:, :2].T + A_inv[:, 2]


def in_bounds(px_full: np.ndarray, full_size: tuple[int, int] = PLAN_FULL_SIZE) -> np.ndarray:
    """
    Pixels inside the full-resolution floorplan.
    Args:
        px_full (np.ndarray): tensor with shape [..., 2]
        full_size (tuple[int, int]): (width, height)

    Returns:
        np.ndarray: boolean tensor with shape [...]
    """
    return (px_full[..., 0] >= 0) & (px_full[..., 0] < full_size[0]) & (px_full[..., 1] >= 0) & (px_full[..., 1] < full_size[1])


def on_wall(points: np.ndarray, plan_params: dict, max_dist_m: float = 0.2) -> np.ndarray:
    """
    Points that lie on a wall of the floorplan: within max_dist_m of a wall pixel of hge.png.
    Args:
        points (np.ndarray): tensor with shape [N, 2 or 3], world metres (NaN allowed)
        plan_params (dict): of load_floorplan
        max_dist_m (float): distance to the nearest wall pixel [m]

    Returns:
        np.ndarray: boolean tensor with shape [N]
    """
    px = world_to_plan_full(points, plan_params["A"])
    result = np.zeros(len(px), dtype=bool)
    valid = np.isfinite(px).all(axis=1)
    valid[valid] &= in_bounds(px[valid])
    idx = np.round(px[valid]).astype(int)
    idx[:, 0] = np.clip(idx[:, 0], 0, plan_params["full_size"][0] - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, plan_params["full_size"][1] - 1)
    result[valid] = plan_params["wall_dist_px"][idx[:, 1], idx[:, 0]] * plan_params["m_per_px"] <= max_dist_m
    return result


def on_plan(points: np.ndarray, plan_params: dict, max_wall_dist_m: float | None = None) -> np.ndarray:
    """
    Points that give correspondences: inside the floorplan (as the annotations of C3, any structure), or within
    max_wall_dist_m of a wall if it is given.
    Args:
        points (np.ndarray): tensor with shape [N, 2 or 3], world metres (NaN allowed)
        plan_params (dict): of load_floorplan
        max_wall_dist_m (float | None): distance to the nearest wall pixel [m], None: no wall test

    Returns:
        np.ndarray: boolean tensor with shape [N]
    """
    if max_wall_dist_m is not None:
        return on_wall(points, plan_params, max_wall_dist_m)
    px = world_to_plan_full(points, plan_params["A"])
    result = np.isfinite(px).all(axis=1)
    result[result] = in_bounds(px[result], plan_params["full_size"])
    return result


# ----------------------------------------------------------------------------------------------------------------------
# Frames and windows


def is_indoor(timestamps: np.ndarray) -> np.ndarray:
    """
    Frames inside one of the indoor intervals of UnLoc (training sessions only).
    Args:
        timestamps (np.ndarray): integer tensor with shape [N], microseconds

    Returns:
        np.ndarray: boolean tensor with shape [N]
    """
    indoor = np.zeros(len(timestamps), dtype=bool)
    for start, end in INDOOR_INTERVALS:
        indoor |= (timestamps >= start) & (timestamps <= end)
    return indoor


def kept_indices(num_frames: int, shift: list) -> np.ndarray:
    """
    Frames of a test session kept by the Z-FLoc index shift, in order.
    Args:
        num_frames (int): number of session frames N
        shift (list): [start, end, restart]; frames [start:end] and [restart:] are kept

    Returns:
        np.ndarray: integer tensor with shape [K]
    """
    frames = np.arange(num_frames)
    return np.concatenate((frames[shift[0]:shift[1]], frames[shift[2]:]))


def load_zfloc_poses_map(npz_path: str | Path, session: str, seq_length: int = 20) -> np.ndarray:
    """
    Ground-truth floorplan positions of a test session bundled with Z-FLoc (assets/metadata/hge.npz).
    Args:
        npz_path (str | Path): hge.npz, whose members are the bytes of the bundled files
        session (str): test session
        seq_length (int): bundled condition (the positions are the same for every length)

    Returns:
        np.ndarray: tensor with shape [K, 2], (u, v) pixels of hge.png of the kept frames
    """
    with np.load(npz_path, allow_pickle=False) as archive:
        member = archive[f"data/hge_complete/hge_{seq_length}/{session}/poses_map.npy"]
    return np.load(io.BytesIO(member.tobytes()))


def kept_indices_from_poses_map(poses_map: np.ndarray, c2w: np.ndarray, A: np.ndarray, tol_px: float = 1.0) -> np.ndarray:
    """
    Frames of a session that match bundled ground-truth positions one to one (reconciles the frame count of
    ios_2022-01-18_17.10.39_000, where the index shift keeps 516 frames but Z-FLoc evaluates 483).
    Args:
        poses_map (np.ndarray): tensor with shape [K, 2], pixels of hge.png (load_zfloc_poses_map)
        c2w (np.ndarray): tensor with shape [N, 4, 4], all frames of the session
        A (np.ndarray): affine matrix with shape [2, 3]
        tol_px (float): maximum distance between a bundled position and its frame [px]

    Returns:
        np.ndarray: integer tensor with shape [K], increasing frame indices
    """
    px = world_to_plan_full(c2w[:, :3, 3], A)  # [N, 2]
    dist = np.linalg.norm(poses_map[:, None] - px[None], axis=-1)  # [K, N]
    nearest = dist.argmin(axis=1)
    if dist[np.arange(len(poses_map)), nearest].max() > tol_px:
        raise ValueError(f"bundled positions farther than {tol_px} px from every frame")
    if len(np.unique(nearest)) != len(nearest) or not np.all(np.diff(nearest) > 0):
        raise ValueError("bundled positions do not match the frames one to one in order")
    return nearest


def iter_test_windows(num_kept: int, seq_length: int) -> list[slice]:
    """
    Non-overlapping windows of consecutive kept frames (Z-FLoc); the remaining frames at the end are dropped.
    Args:
        num_kept (int): number of kept frames
        seq_length (int): frames per window

    Returns:
        list[slice]: window k covers kept frames [k * L, (k + 1) * L)
    """
    return [slice(k * seq_length, (k + 1) * seq_length) for k in range(num_kept // seq_length)]


def train_runs(timestamps: np.ndarray, max_gap_s: float = 5.0) -> list[tuple[int, int]]:
    """
    Runs of kept frames without a gap of more than max_gap_s (the frames dropped by the filters).
    Args:
        timestamps (np.ndarray): integer tensor with shape [N], microseconds of the kept frames, increasing
        max_gap_s (float): largest gap between consecutive frames of a run [s]

    Returns:
        list[tuple[int, int]]: (first frame, number of frames) of every run
    """
    cuts = np.nonzero(np.diff(timestamps) / 1e6 > max_gap_s)[0] + 1
    starts, ends = np.concatenate([[0], cuts]), np.concatenate([cuts, [len(timestamps)]])
    return [(int(a), int(b - a)) for a, b in zip(starts, ends)]


def sample_train_windows(timestamps: np.ndarray, num_windows: int, rng: np.random.Generator,
                         num_frames: tuple = TRAIN_FRAMES, max_step: int = TRAIN_MAX_STEP) -> list[np.ndarray]:
    """
    Random training windows: a run (train_runs) drawn proportionally to its length, a number of frames drawn in
    num_frames (at most the run), intervals drawn in 1..max_step (fewer if the run is too short), a random start.
    Args:
        timestamps (np.ndarray): integer tensor with shape [N], microseconds of the kept frames, increasing
        num_windows (int): windows to draw
        rng (np.random.Generator): random generator
        num_frames (tuple): (min, max) frames of a window; runs shorter than min are not used
        max_step (int): largest interval between the frames of a window

    Returns:
        list[np.ndarray]: indices of the kept frames of every window, increasing
    """
    runs = [run for run in train_runs(timestamps) if run[1] >= num_frames[0]]
    if not runs:
        return []
    lengths = np.array([n for _, n in runs], dtype=np.float64)
    windows = []
    for r in rng.choice(len(runs), num_windows, p=lengths / lengths.sum()):
        first, run_len = runs[r]
        n = int(rng.integers(num_frames[0], min(num_frames[1], run_len) + 1))
        steps = rng.integers(1, min(max_step, (run_len - 1) // max(1, n - 1)) + 1, n - 1)  # the window fits the run
        offsets = np.concatenate([[0], np.cumsum(steps)])
        windows.append(first + int(rng.integers(0, run_len - offsets[-1])) + offsets)
    return windows


def sample_distinct_windows(timestamps: np.ndarray, num_windows: int, rng: np.random.Generator, seen: set,
                            num_frames: tuple = TRAIN_FRAMES, max_step: int = TRAIN_MAX_STEP) -> list[np.ndarray]:
    """
    Random training windows (sample_train_windows) whose combinations of frames are not in seen; seen is updated.
    Args:
        timestamps (np.ndarray): integer tensor with shape [N], microseconds of the kept frames, increasing
        num_windows (int): windows to draw
        rng (np.random.Generator): random generator
        seen (set): tuples of the frame indices of the windows drawn before (of every epoch)
        num_frames (tuple): (min, max) frames of a window
        max_step (int): largest interval between the frames of a window

    Returns:
        list[np.ndarray]: indices of the kept frames of every window, increasing
    """
    windows = []
    for _ in range(100 * num_windows):  # duplicates are rare: frames, intervals and start are random
        if len(windows) == num_windows:
            break
        drawn = sample_train_windows(timestamps, 1, rng, num_frames, max_step)
        if not drawn:
            break
        if tuple(drawn[0]) not in seen:
            seen.add(tuple(drawn[0]))
            windows.append(drawn[0])
    return windows


def session_chunks(timestamps: np.ndarray, chunk_size: int) -> list[np.ndarray]:
    """
    Consecutive kept frames reconstructed together (multi-view MapAnything, cached for all the windows): every run
    (train_runs) split into the fewest chunks of at most chunk_size frames, of equal lengths.
    Args:
        timestamps (np.ndarray): integer tensor with shape [N], microseconds of the kept frames, increasing
        chunk_size (int): most frames of a chunk

    Returns:
        list[np.ndarray]: indices of the kept frames of every chunk, covering all of them
    """
    chunks = []
    for first, run_len in train_runs(timestamps):
        chunks += np.array_split(first + np.arange(run_len), -(-run_len // chunk_size))
    return chunks


def window_name(session: str, timestamps: np.ndarray) -> str:
    """
    Name of a training window (files and cached reconstructions): its first timestamp, number of frames and a hash
    of its timestamps.
    Args:
        session (str): session name
        timestamps (np.ndarray): integer tensor with shape [S], microseconds of the frames of the window

    Returns:
        str: "<session>_t<first timestamp>_n<S>_<crc32 of the timestamps>"
    """
    return f"{session}_t{int(timestamps[0])}_n{len(timestamps)}_{zlib.crc32(np.asarray(timestamps, np.int64).tobytes()):08x}"


# ----------------------------------------------------------------------------------------------------------------------
# Poses and gravity


def relative_extrinsics(c2w: np.ndarray) -> np.ndarray:
    """
    Camera poses relative to the first camera (camera-to-camera-0), the extrinsics of SceneAligner.
    Args:
        c2w (np.ndarray): tensor with shape [S, 4, 4], camera-to-world

    Returns:
        np.ndarray: tensor with shape [S, 4, 4], the first one is the identity
    """
    return np.linalg.inv(c2w[0]) @ c2w


def pose_deviation(poses: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Rotation and translation differences between two sets of poses.
    Args:
        poses (np.ndarray): tensor with shape [S, 4, 4]
        reference (np.ndarray): tensor with shape [S, 4, 4]

    Returns:
        tuple[np.ndarray, np.ndarray]: angles [deg] and translation distances [m], each with shape [S]
    """
    relative = np.einsum("sji,sjk->sik", reference[:, :3, :3], poses[:, :3, :3])  # R_ref^T R
    cos = np.clip((np.trace(relative, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0)
    return np.degrees(np.arccos(cos)), np.linalg.norm(poses[:, :3, 3] - reference[:, :3, 3], axis=1)


def gravity_cam0(c2w0: np.ndarray) -> np.ndarray:
    """
    Gravity (down) direction in the frame of camera 0, the est_gravity_cam of SceneAligner.
    Args:
        c2w0 (np.ndarray): tensor with shape [4, 4], camera-to-world of camera 0

    Returns:
        np.ndarray: unit vector with shape [3]
    """
    return c2w0[:3, :3].T @ GRAVITY_WORLD


def check_world_up(c2w: np.ndarray) -> dict[str, float]:
    """
    Angles that verify the z-up assumption of the world frame.
    Args:
        c2w (np.ndarray): tensor with shape [N, 4, 4]

    Returns:
        dict[str, float]: "plane_normal_deg" (angle between the normal of the plane through the camera centres and
            +z) and "camera_up_deg" (angle between the mean camera up direction, -y of OpenCV, and +z), degrees
    """
    centres = c2w[:, :3, 3]
    normal = np.linalg.svd(centres - centres.mean(axis=0))[2][-1]
    camera_up = -c2w[:, :3, 1].mean(axis=0)
    camera_up /= np.linalg.norm(camera_up)
    return {"plane_normal_deg": float(np.degrees(np.arccos(abs(normal[2])))),
            "camera_up_deg": float(np.degrees(np.arccos(np.clip(camera_up[2], -1.0, 1.0))))}


def ground_rotation(gravity_cam: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Rotation from a camera frame to the gravity-aligned ground frame of SceneAligner: +Y along gravity, +X close to
    the x-axis of the camera (Gram-Schmidt, as in scenealigner.inference.reconstruct_projection_geometry).
    Args:
        gravity_cam (np.ndarray): gravity direction with shape [3]
        eps (float): numerical guard

    Returns:
        np.ndarray: rotation matrix with shape [3, 3], points_ground = points_cam @ R.T
    """
    y_target = gravity_cam / (np.linalg.norm(gravity_cam) + eps)
    x_ref = np.array([1.0, 0.0, 0.0])
    if np.abs(np.dot(y_target, x_ref)) > 0.9999:  # gravity parallel to the x-axis
        x_ref = np.array([0.0, 0.0, 1.0])
    z_temp = np.cross(x_ref, y_target)
    z_temp /= np.linalg.norm(z_temp)
    x_target = np.cross(y_target, z_temp)
    x_target /= np.linalg.norm(x_target)
    z_target = np.cross(x_target, y_target)
    z_target /= np.linalg.norm(z_target)
    return np.stack([x_target, y_target, z_target], axis=1).T


def filter_ground_points(points: np.ndarray, height_percentiles: tuple = (5.0, 80.0), xz_slice_val: float = 2.5) -> np.ndarray:
    """
    Height and x/z percentile filters of the density map (reconstruct_projection_geometry without confidences).
    Args:
        points (np.ndarray): tensor with shape [N, 3], ground frame (+Y down)
        height_percentiles (tuple): kept range of y (ceiling 5 % and floor 20 % removed)
        xz_slice_val (float): percentile cut on both sides of x and z

    Returns:
        np.ndarray: boolean tensor with shape [N]
    """
    ceil_cutoff, floor_cutoff = np.percentile(points[:, 1], height_percentiles)
    mask = (points[:, 1] >= ceil_cutoff) & (points[:, 1] <= floor_cutoff)
    x_min, x_max = np.percentile(points[mask, 0], [xz_slice_val, 100 - xz_slice_val])
    z_min, z_max = np.percentile(points[mask, 2], [xz_slice_val, 100 - xz_slice_val])
    return mask & (points[:, 0] >= x_min) & (points[:, 0] <= x_max) & (points[:, 2] >= z_min) & (points[:, 2] <= z_max)


def top_down_projection(points: np.ndarray, R_rect: np.ndarray, density_size: int = 512, padding: float = 0.1) -> dict:
    """
    Top-down projection parameters of filtered ground-frame points (the proj_params of SceneAligner).
    Args:
        points (np.ndarray): tensor with shape [N, 3], ground frame (+Y down), already filtered
        R_rect (np.ndarray): rotation with shape [3, 3], camera 0 -> ground
        density_size (int): side of the density map [px]
        padding (float): margin on each side, as a fraction of the map

    Returns:
        dict: "R_rect", "min_x", "min_z", "scale", "cx", "cz", "density_size" (see reconstruct_projection_geometry)
    """
    min_x, min_z = points[:, 0].min(), points[:, 2].min()
    width_m, height_m = points[:, 0].max() - min_x, points[:, 2].max() - min_z
    scale = (density_size * (1 - padding * 2)) / max(width_m, height_m)
    return {"R_rect": R_rect, "min_x": min_x, "min_z": min_z, "scale": scale,
            "cx": (density_size - width_m * scale) / 2, "cz": (density_size - height_m * scale) / 2,
            "density_size": density_size}


# ----------------------------------------------------------------------------------------------------------------------
# LiDAR depth


def load_depth(path: str | Path) -> np.ndarray:
    """
    Depth map of a phone frame.
    Args:
        path (str | Path): 16-bit PNG in millimetres, 0 where invalid

    Returns:
        np.ndarray: float32 tensor with shape [H, W], z-depth [m], NaN where invalid
    """
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED).astype(np.float32) / DEPTH_SCALE
    depth[depth <= 0] = np.nan
    return depth


def sample_depth_on_grid(depth: np.ndarray, K_depth: np.ndarray, K_out: np.ndarray, hw_out: tuple[int, int]) -> np.ndarray:
    """
    Z-depth of a depth map at the pixels of another image of the same camera: the ray of every output pixel
    (K_out) is projected into the depth map (K_depth) and the nearest depth value is taken. No resizing is guessed.
    Args:
        depth (np.ndarray): tensor with shape [Hd, Wd], z-depth [m], NaN where invalid
        K_depth (np.ndarray): intrinsics with shape [3, 3] of the depth map
        K_out (np.ndarray): intrinsics with shape [3, 3] of the output grid
        hw_out (tuple[int, int]): (H, W) of the output grid

    Returns:
        np.ndarray: float32 tensor with shape [H, W], NaN outside the depth map or where invalid
    """
    H, W = hw_out
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    x, y = (u - K_out[0, 2]) / K_out[0, 0], (v - K_out[1, 2]) / K_out[1, 1]  # ray directions at z = 1
    ud = np.round(K_depth[0, 0] * x + K_depth[0, 2]).astype(int)
    vd = np.round(K_depth[1, 1] * y + K_depth[1, 2]).astype(int)
    inside = (ud >= 0) & (ud < depth.shape[1]) & (vd >= 0) & (vd < depth.shape[0])
    out = np.full((H, W), np.nan, dtype=np.float32)
    out[inside] = depth[vd[inside], ud[inside]]
    return out


def unproject(depth_z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Camera-frame 3D points of a z-depth map.
    Args:
        depth_z (np.ndarray): tensor with shape [H, W], NaN where invalid
        K (np.ndarray): intrinsics with shape [3, 3]

    Returns:
        np.ndarray: float32 tensor with shape [H, W, 3], OpenCV camera frame, NaN where invalid
    """
    H, W = depth_z.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    points = np.stack([(u - K[0, 2]) * depth_z / K[0, 0], (v - K[1, 2]) * depth_z / K[1, 1], depth_z], axis=-1)
    return points.astype(np.float32)


def lidar_points(depth_paths: list, K_depth: np.ndarray, extrinsics: np.ndarray) -> np.ndarray:
    """
    LiDAR points of the frames of a window in the frame of camera 0, at the resolution of the depth maps (as the
    ground-truth depth samples of prepare_s3d.py).
    Args:
        depth_paths (list): S depth PNG paths, None for frames without depth
        K_depth (np.ndarray): tensor with shape [S, 3, 3], intrinsics of the depth maps (NaN without depth)
        extrinsics (np.ndarray): tensor with shape [S, 4, 4], camera-to-camera-0

    Returns:
        np.ndarray: float32 tensor with shape [M, 3], the valid depth pixels of all the frames
    """
    points = [np.zeros((0, 3), np.float32)]
    for i, path in enumerate(depth_paths):
        if path is None:
            continue
        local = unproject(load_depth(path), K_depth[i]).reshape(-1, 3)
        local = local[np.isfinite(local).all(axis=1)]
        points.append((local @ extrinsics[i, :3, :3].T + extrinsics[i, :3, 3]).astype(np.float32))
    return np.concatenate(points)


def lidar_points_cam0(depth_paths: list, K_depth: np.ndarray, K_out: np.ndarray, hw_out: tuple[int, int],
                      extrinsics: np.ndarray) -> np.ndarray:
    """
    LiDAR points of the frames of a window in the frame of camera 0, on the pixel grid of the reconstruction. The
    ARKit depth maps are registered to the photos (same pose, intrinsics scaled to the depth resolution).
    Args:
        depth_paths (list): S depth PNG paths, None for frames without depth
        K_depth (np.ndarray): tensor with shape [S, 3, 3], intrinsics of the depth maps (NaN without depth)
        K_out (np.ndarray): tensor with shape [S, 3, 3], intrinsics of the reconstruction grid
        hw_out (tuple[int, int]): (H, W) of the reconstruction grid
        extrinsics (np.ndarray): tensor with shape [S, 4, 4], camera-to-camera-0

    Returns:
        np.ndarray: float32 tensor with shape [S, H, W, 3], NaN where there is no depth
    """
    points = np.full((len(depth_paths), *hw_out, 3), np.nan, dtype=np.float32)
    for i, path in enumerate(depth_paths):
        if path is None:
            continue
        local = unproject(sample_depth_on_grid(load_depth(path), K_depth[i], K_out[i], hw_out), K_out[i])
        points[i] = local @ extrinsics[i, :3, :3].T + extrinsics[i, :3, 3]
    return points


# ----------------------------------------------------------------------------------------------------------------------
# Sessions


@dataclass
class Session:
    """Frames of one split session: images and poses of the map session, LiDAR depth of the raw session."""
    name: str
    timestamps: np.ndarray  # [N] microseconds
    image_paths: list  # N paths
    depth_paths: list  # N paths or None
    c2w: np.ndarray  # [N, 4, 4] camera-to-world [m]
    K: np.ndarray  # [N, 3, 3] of the photos (one camera sensor per frame)
    K_depth: np.ndarray  # [N, 3, 3] of the depth maps, NaN for frames without depth

    def __len__(self) -> int:
        return len(self.timestamps)

    def subset(self, indices: np.ndarray) -> "Session":
        """Frames at the given indices, in that order."""
        return Session(self.name, self.timestamps[indices], [self.image_paths[i] for i in indices],
                       [self.depth_paths[i] for i in indices], self.c2w[indices], self.K[indices], self.K_depth[indices])


def load_session(map_dir: str | Path, raw_dir: str | Path | None, session: str) -> Session:
    """
    One split session: timestamps and camera-to-world poses of HGE/sessions/map/trajectories.txt, the photos of
    HGE/sessions/map/raw_data/<session>/images/, and the LiDAR depth maps of the raw phone session matched by
    timestamp (if raw_dir is given and the raw session exists).
    Args:
        map_dir (str | Path): HGE/sessions/map
        raw_dir (str | Path | None): directory of the raw sessions (ios_<date>_<time>/), or None without depth
        session (str): split session name

    Returns:
        Session: frames in trajectory order
    """
    map_dir = Path(map_dir)
    timestamps, device_ids, c2w = load_trajectory(map_dir / "trajectories.txt", session)
    sensors = load_sensors(map_dir / "sensors.txt") if (map_dir / "sensors.txt").exists() else {}
    K = np.stack([sensors[d]["K"] if d in sensors else HGE_INTRINSICS for d in device_ids])
    images = load_records(map_dir / "images.txt", session) if (map_dir / "images.txt").exists() else {}
    image_paths = [str(map_dir / "raw_data" / images[ts][1]) if ts in images else
                   str(map_dir / "raw_data" / session / "images" / f"{ts}.jpg") for ts in timestamps]

    depth_paths, K_depth = [None] * len(timestamps), np.full((len(timestamps), 3, 3), np.nan)
    raw_session = Path(raw_dir) / raw_session_name(session) if raw_dir is not None else None
    if raw_session is not None and (raw_session / "depths.txt").exists():
        records = load_records(raw_session / "depths.txt")  # timestamp -> (depth_phone[_<timestamp>], depth/<ts>.png)
        raw_sensors = load_sensors(raw_session / "sensors.txt")
        for i, ts in enumerate(timestamps):
            if ts in records:
                depth_paths[i] = str(raw_session / "raw_data" / records[ts][1])
                K_depth[i] = raw_sensors[records[ts][0]]["K"]
    return Session(session, timestamps, image_paths, depth_paths, c2w, K, K_depth)


def train_frames(session: Session, plan_params: dict) -> np.ndarray:
    """
    Frames of a training session: indoor (UnLoc intervals) and inside the floorplan.
    Args:
        session (Session): all frames
        plan_params (dict): of load_floorplan

    Returns:
        np.ndarray: integer tensor with shape [K], increasing frame indices
    """
    keep = is_indoor(session.timestamps) & in_bounds(world_to_plan_full(session.c2w[:, :3, 3], plan_params["A"]))
    return np.nonzero(keep)[0]


def test_frames(session: Session, plan_params: dict, zfloc_npz: str | Path | None = None) -> np.ndarray:
    """
    Frames of a test session evaluated by Z-FLoc: the index shift, or the bundled positions when zfloc_npz is given.
    Args:
        session (Session): all frames
        plan_params (dict): of load_floorplan
        zfloc_npz (str | Path | None): Z-FLoc assets/metadata/hge.npz, to reconcile the frame counts

    Returns:
        np.ndarray: integer tensor with shape [K], increasing frame indices
    """
    if zfloc_npz is not None:
        return kept_indices_from_poses_map(load_zfloc_poses_map(zfloc_npz, session.name), session.c2w, plan_params["A"])
    return kept_indices(len(session), HGE_INDEX_SHIFT[session.name])


def check(args: argparse.Namespace) -> None:
    """Prints what prepare_lamar.py and evaluate_lamar.py will find: sessions, frames, filters, depth, sensors."""
    plan_img, plan_params = load_floorplan(args.floorplan, args.plan_size)
    print(f"floorplan: {plan_params['full_size']} px, {plan_params['m_per_px']:.4f} m/px, resized to {plan_img.size}")
    residual = world_to_plan_full(TRAJECTORY_CORRESPONDENCES, plan_params["A"]) - FLOORPLAN_CORRESPONDENCES
    print(f"affine fit residual: {np.linalg.norm(residual, axis=1).round(1)} px")
    sensors = load_sensors(Path(args.map_dir) / "sensors.txt") if (Path(args.map_dir) / "sensors.txt").exists() else {}
    print(f"map sensors.txt: {len(sensors)} camera-like sensors")
    for split, sessions in load_split(args.split).items():
        for name in sessions:
            raw = Path(args.raw_dir) / raw_session_name(name) if args.raw_dir else None
            if not (Path(args.map_dir) / "raw_data" / name).is_dir():
                print(f"{split} {name}: MISSING in {args.map_dir}/raw_data")
                continue
            session = load_session(args.map_dir, args.raw_dir, name)
            up = check_world_up(session.c2w)
            with_depth = sum(p is not None for p in session.depth_paths)
            if split == "test":
                kept = test_frames(session, plan_params, args.zfloc_npz)
                detail = f"kept by the index shift: {len(kept)}"
            else:
                kept = train_frames(session, plan_params)
                runs = [n for _, n in train_runs(session.timestamps[kept])]
                detail = f"indoor and in bounds: {len(kept)}, runs without gaps: {runs}"
            raw_state = "no raw session" if raw is None or not raw.is_dir() else f"depth for {with_depth}/{len(session)} frames"
            images_found = sum(os.path.exists(p) for p in session.image_paths)
            print(f"{split} {name}: {len(session)} frames ({images_found} images found), {detail}, {raw_state}, "
                  f"plane normal {up['plane_normal_deg']:.1f} deg, camera up {up['camera_up_deg']:.1f} deg from +z")
            fx = session.K[:, 0, 0]
            print(f"    image K[0] = {session.K[0][[0, 1, 0, 1], [0, 1, 2, 2]].round(1)}, fx in [{fx.min():.1f}, {fx.max():.1f}]")
            if with_depth:
                i = next(i for i, p in enumerate(session.depth_paths) if p is not None)
                depth_shape = load_depth(session.depth_paths[i]).shape
                ratio = session.K_depth[i][[0, 1, 0, 1], [0, 1, 2, 2]] / session.K[i][[0, 1, 0, 1], [0, 1, 2, 2]]
                print(f"    depth {depth_shape[1]}x{depth_shape[0]}, K_depth / K of frame {i} = {ratio.round(4)} (fx, fy, cx, cy)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="prints the sessions, frames and sensors found")
    parser.add_argument("--map_dir", required=True, help="HGE/sessions/map (trajectories.txt, sensors.txt, raw_data/)")
    parser.add_argument("--raw_dir", default=None, help="directory of the raw phone sessions (LiDAR depth)")
    parser.add_argument("--floorplan", required=True, help="hge.png")
    parser.add_argument("--split", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "split.yaml"))
    parser.add_argument("--plan_size", type=int, default=1024)
    parser.add_argument("--zfloc_npz", default=None, help="Z-FLoc assets/metadata/hge.npz, to reconcile the test frames")
    check(parser.parse_args())
