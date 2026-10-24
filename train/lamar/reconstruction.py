"""3D reconstruction of a window of LaMAR frames with MapAnything, given the ground-truth intrinsics and relative
camera poses, in the format of scenealigner.preprocess.reconstruct (so the density maps, correspondences and
camera centres of SceneAligner are computed as for pi3).

MapAnything is called as in Z-FLoc (tools/metadata_generation/geometry_utils.py, process_images): the inputs are
metric, the poses are OpenCV camera-to-world relative to the first frame, and the gravity direction comes from
the pose of the first frame instead of GeoCalib.

Training windows are not reconstructed one by one: the kept frames of a session are reconstructed in chunks of
consecutive frames (save_chunk), and every window is assembled from these points (window_from_cache).
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lamar import gravity_cam0, pose_deviation, relative_extrinsics

MAPANYTHING_MODEL = "facebook/map-anything"  # CC BY-NC 4.0; "facebook/map-anything-apache" for commercial use
MAPANYTHING_REVISION = None  # the weights of the Hugging Face Hub at their current revision


def load_mapanything(model_id: str = MAPANYTHING_MODEL, device: str = "cuda"):
    """
    MapAnything in evaluation mode.
    Args:
        model_id (str): Hugging Face model id or local directory
        device (str): torch device

    Returns:
        mapanything.models.MapAnything: the model on device
    """
    from mapanything.models import MapAnything  # optional dependency (train/lamar/README.md)

    return MapAnything.from_pretrained(model_id).eval().to(device)


def load_image_uint8(path: str) -> torch.Tensor:
    """
    A photo as MapAnything expects it (no EXIF orientation applied: the LaMAR poses refer to the stored pixels).
    Args:
        path (str): JPEG or PNG

    Returns:
        torch.Tensor: uint8 tensor with shape [H, W, 3], RGB
    """
    return torch.from_numpy(np.array(Image.open(path).convert("RGB")))


@torch.no_grad()
def reconstruct(image_paths: list, K: np.ndarray, c2w: np.ndarray, model, long_side: int | None = None,
                minibatch_size: int | None = None) -> dict:
    """
    Reconstructs the frames of a window in the frame of the first camera, with the ground-truth geometry as input.
    Args:
        image_paths (list): S photo paths, in chronological order
        K (np.ndarray): intrinsics with shape [S, 3, 3] (or [3, 3] shared) of the photos (rescaled with the images by
            MapAnything)
        c2w (np.ndarray): tensor with shape [S, 4, 4], camera-to-world poses [m] (OpenCV camera axes)
        model: of load_mapanything
        long_side (int | None): inference resolution of the long side [px]; None for MapAnything's default 518
        minibatch_size (int | None): views per minibatch of the memory-efficient inference

    Returns:
        dict: the keys of scenealigner.preprocess.reconstruct and more:
            "points_cam0" float32 [S, H, W, 3] (finite; 0 where masked), "conf" float32 [S, H, W] (0 where masked),
            "mask" bool [S, H, W], "extrinsics" [S, 4, 4] (ground-truth camera-to-camera-0),
            "pose_error_deg", "pose_error_m" [S] (deviation of the poses predicted by MapAnything from the given ones),
            "est_gravity_cam" [3] (down direction in camera 0), "preprocessed_images" uint8 [S, H, W, 3],
            "image_sizes" int [S, 2] ((W, H) of every frame), "intrinsics" float32 [S, 3, 3] (of the H x W grid)
    """
    from mapanything.utils.image import preprocess_inputs

    extrinsics = relative_extrinsics(c2w)
    K = np.broadcast_to(K, (len(image_paths), 3, 3))
    views = [{"img": load_image_uint8(path), "intrinsics": torch.tensor(K[i], dtype=torch.float32),
              "camera_poses": torch.tensor(extrinsics[i], dtype=torch.float32), "is_metric_scale": torch.tensor([True])}
             for i, path in enumerate(image_paths)]
    if long_side:
        views = preprocess_inputs(views, resize_mode="longest_side", size=int(long_side))
    else:
        views = preprocess_inputs(views)
    predictions = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16", apply_mask=True,
                              mask_edges=True, apply_confidence_mask=False, confidence_percentile=10,
                              ignore_pose_scale_inputs=False, ignore_calibration_inputs=False,
                              **({"minibatch_size": minibatch_size} if minibatch_size is not None else {}))

    # MapAnything predicts the poses again from the given ones: they deviate slightly. The points of each view (its
    # camera frame, i.e. its predicted depth) are placed with the ground-truth pose, consistent with "extrinsics".
    points, conf, mask, images, intrinsics = [], [], [], [], []
    for i, pred in enumerate(predictions):
        view_mask = pred["mask"][0, ..., 0].bool().cpu().numpy()
        view_points = pred["pts3d_cam"][0].float().cpu().numpy() @ extrinsics[i, :3, :3].T.astype(np.float32) \
            + extrinsics[i, :3, 3].astype(np.float32)
        view_points[~view_mask] = 0.0  # finite everywhere: reconstruct_projection_geometry takes percentiles
        view_conf = pred["conf"][0].float().cpu().numpy()
        view_conf[~view_mask] = 0.0
        points.append(view_points)
        conf.append(view_conf)
        mask.append(view_mask)
        images.append(pred["img_no_norm"][0].float().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy())
        intrinsics.append(pred["intrinsics"][0].float().cpu().numpy())
    poses = np.stack([pred["camera_poses"][0].float().cpu().numpy() for pred in predictions])
    pose_error_deg, pose_error_m = pose_deviation(poses, extrinsics)
    H, W = points[0].shape[:2]
    return {
        "points_cam0": np.stack(points).astype(np.float32),
        "conf": np.stack(conf).astype(np.float32),
        "mask": np.stack(mask),
        "extrinsics": extrinsics.astype(np.float64),
        "est_gravity_cam": gravity_cam0(c2w[0]),
        "preprocessed_images": np.stack(images),
        "image_sizes": np.tile(np.array([[W, H]]), (len(image_paths), 1)),
        "intrinsics": np.stack(intrinsics).astype(np.float32),
        "pose_error_deg": pose_error_deg,
        "pose_error_m": pose_error_m,
    }


def save_recon(path: str, recon: dict) -> None:
    """
    Caches a reconstruction (float16 points to halve the size).
    Args:
        path (str): .npz file
        recon (dict): of reconstruct
    """
    np.savez_compressed(path, **{k: (v.astype(np.float16) if k == "points_cam0" else v) for k, v in recon.items()})


CHUNK_KEYS = ("points_cam", "conf", "mask", "preprocessed_images", "intrinsics", "pose_error_deg",
              "pose_error_m")  # per-frame arrays of a cached chunk, plus "timestamps"


def save_chunk(directory: str, recon: dict, timestamps: np.ndarray) -> None:
    """
    Caches the reconstruction of a chunk of a session (lamar.session_chunks) as .npy files (memory-mapped by
    SessionCache): the points of every frame in its own camera frame (float16), so that any window of the session
    can be assembled with the ground-truth poses (window_from_cache). timestamps.npy is written last (complete chunk).
    Args:
        directory (str): folder of the chunk
        recon (dict): of reconstruct
        timestamps (np.ndarray): integer tensor with shape [S], microseconds of the frames
    """
    os.makedirs(directory, exist_ok=True)
    ext = recon["extrinsics"].astype(np.float32)  # camera-to-camera-0
    points = (recon["points_cam0"] - ext[:, None, None, :3, 3]) @ ext[:, None, :3, :3]  # R^T (p - t): own camera
    points[~recon["mask"]] = 0.0
    for key in CHUNK_KEYS:
        np.save(os.path.join(directory, f"{key}.npy"), points.astype(np.float16) if key == "points_cam" else recon[key])
    np.save(os.path.join(directory, "timestamps.npy"), np.asarray(timestamps, np.int64))


class SessionCache:
    """The cached chunks (save_chunk) of one session, memory-mapped; frames are taken by timestamp."""

    def __init__(self, directory: str):
        chunk_dirs = sorted(p for p in Path(directory).iterdir() if (p / "timestamps.npy").exists())
        self.chunks = [{k: np.load(p / f"{k}.npy", mmap_mode="r") for k in (*CHUNK_KEYS, "timestamps")} for p in chunk_dirs]
        order = np.argsort([c["timestamps"][0] for c in self.chunks])
        self.chunks = [self.chunks[i] for i in order]
        self.timestamps = np.concatenate([c["timestamps"] for c in self.chunks])
        self.starts = np.cumsum([0] + [len(c["timestamps"]) for c in self.chunks])[:-1]

    def take(self, key: str, timestamps: np.ndarray) -> np.ndarray:
        """
        Per-frame array of the given frames.
        Args:
            key (str): of CHUNK_KEYS
            timestamps (np.ndarray): integer tensor with shape [S], microseconds of cached frames

        Returns:
            np.ndarray: tensor with shape [S, ...]
        """
        pos = np.searchsorted(self.timestamps, timestamps)
        assert (pos < len(self.timestamps)).all() and (self.timestamps[pos] == timestamps).all(), "frames not cached"
        chunk = np.searchsorted(self.starts, pos, side="right") - 1
        first = self.chunks[0][key]
        out = np.empty((len(pos), *first.shape[1:]), dtype=first.dtype)
        for c in np.unique(chunk):
            sel = chunk == c
            out[sel] = self.chunks[c][key][pos[sel] - self.starts[c]]
        return out


def window_from_cache(cache: SessionCache, timestamps: np.ndarray, c2w: np.ndarray, images: bool = False) -> dict:
    """
    Reconstruction of a window assembled from the cached chunks of its session, as reconstruct returns it: the points
    of every frame placed with its ground-truth pose relative to the first frame of the window.
    Args:
        cache (SessionCache): of the session
        timestamps (np.ndarray): integer tensor with shape [S], microseconds of the frames of the window
        c2w (np.ndarray): tensor with shape [S, 4, 4], camera-to-world poses [m]
        images (bool): also "preprocessed_images" (only for visualizations: the samples do not read them)

    Returns:
        dict: the keys of reconstruct ("preprocessed_images" only with images) and "timestamps"
    """
    extrinsics = relative_extrinsics(c2w)
    R, t = extrinsics[:, None, :3, :3].astype(np.float32), extrinsics[:, None, None, :3, 3].astype(np.float32)
    mask = cache.take("mask", timestamps)
    points = cache.take("points_cam", timestamps).astype(np.float32) @ R.transpose(0, 1, 3, 2)
    points += t
    points[~mask] = 0.0
    S, H, W = mask.shape
    recon = {"points_cam0": points, "conf": cache.take("conf", timestamps), "mask": mask,
             "extrinsics": extrinsics.astype(np.float64), "est_gravity_cam": gravity_cam0(c2w[0]),
             "image_sizes": np.tile(np.array([[W, H]]), (S, 1)), "intrinsics": cache.take("intrinsics", timestamps),
             "pose_error_deg": cache.take("pose_error_deg", timestamps),
             "pose_error_m": cache.take("pose_error_m", timestamps), "timestamps": np.asarray(timestamps)}
    if images:
        recon["preprocessed_images"] = cache.take("preprocessed_images", timestamps)
    return recon


def load_recon(path: str) -> dict:
    """
    A cached reconstruction (save_recon).
    Args:
        path (str): .npz file

    Returns:
        dict: of reconstruct, with float32 points
    """
    recon = dict(np.load(path))
    recon["points_cam0"] = recon["points_cam0"].astype(np.float32)
    return recon
