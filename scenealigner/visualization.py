"""2D images of the demo (RGB): density map, its alignment with the floorplan, cameras on the floorplan, and the
training samples logged during training."""
import cv2
import numpy as np
from PIL import Image

ORANGE = (230, 100, 0)  # reliable correspondences
GREEN, BLUE, RED = (0, 160, 0), (0, 80, 255), (220, 0, 0)


def draw_solid_points(img, points, color=ORANGE, radius=10):
    canvas = img.copy()
    for x, y in np.round(points).astype(int):
        cv2.circle(canvas, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def draw_alignment(density_img, plan_img, M, points):
    """Density map warped by M over the floorplan, with points of the density map (e.g., reliable correspondences)."""
    H, W = plan_img.shape[:2]
    canvas = cv2.addWeighted(plan_img, 0.6, cv2.warpAffine(density_img, M, (W, H)), 0.4, 0)
    return draw_solid_points(canvas, cv2.transform(points.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2))


def draw_camera_frustum(img, center, forward_vec, color, length=25, width=15):
    """Camera on the floorplan: filled trapezoid with a black outline."""
    if np.linalg.norm(forward_vec) < 1e-6:  # looking straight up or down
        return img
    forward_vec = forward_vec / np.linalg.norm(forward_vec)
    right_vec = np.array([-forward_vec[1], forward_vec[0]])
    corners = [center + forward_vec * length * a + right_vec * width * b / 2 for a, b in ((0.4, -0.4), (1, -1), (1, 1), (0.4, 0.4))]
    pts = [np.array(corners, dtype=np.int32)]
    cv2.fillPoly(img, pts, color, lineType=cv2.LINE_AA)
    cv2.polylines(img, pts, isClosed=True, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    return img


def draw_scene_cameras(plan_img, cam_to_world, color):
    """Cameras of a floorplan-aligned scene (see get_floorplan_aligned_scene) drawn on the floorplan."""
    for pose in cam_to_world:
        center = (np.array([pose[0, 3], -pose[1, 3]]) + 0.5) * plan_img.shape[0]  # floorplan frame -> pixels
        draw_camera_frustum(plan_img, center, np.array([pose[0, 2], -pose[1, 2]]), color)
    return plan_img


def estimate_camera_on_plan(src_px, dst_px, camera_px, ransac_thr):
    """Camera position on the floorplan from correspondences (density map -> floorplan pixels), or None if the
    similarity transform cannot be estimated."""
    if len(src_px) < 3:
        return None
    M, _ = cv2.estimateAffinePartial2D(src_px.astype(np.float32), dst_px.astype(np.float32), method=cv2.RANSAC,
                                       ransacReprojThreshold=ransac_thr)
    if M is None:
        return None
    return M @ np.array([camera_px[0], camera_px[1], 1.0])


def draw_training_sample(plan_img, pts_plan_gt, pts_plan_pred, valid, camera_plan_gt, camera_plan_pred, rgb_path):
    """Floorplan of a training sample with its correspondences (ground truth in green, predicted in orange) and
    the position of the last camera (ground truth in blue, estimated in red), next to the photo of that camera.

    Points are normalized coordinates in [-1, 1]; a NaN camera position is not drawn. Returns an RGB array.
    """
    canvas = np.ascontiguousarray(plan_img.copy())
    H, W = canvas.shape[:2]

    def to_px(points):
        return (np.asarray(points, dtype=np.float32) + 1) / 2 * np.float32([W, H])

    for x, y in np.round(to_px(pts_plan_gt[valid])).astype(int):
        cv2.circle(canvas, (int(x), int(y)), 2, GREEN, -1, lineType=cv2.LINE_AA)
    for x, y in np.round(to_px(pts_plan_pred[valid])).astype(int):
        cv2.circle(canvas, (int(x), int(y)), 2, ORANGE, -1, lineType=cv2.LINE_AA)
    for camera, color in ((camera_plan_pred, RED), (camera_plan_gt, BLUE)):
        if camera is not None and np.isfinite(camera).all():
            x, y = np.round(to_px(np.asarray(camera)[None])[0]).astype(int)
            cv2.circle(canvas, (int(x), int(y)), max(6, H // 80), color, -1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (int(x), int(y)), max(6, H // 80), (0, 0, 0), 2, lineType=cv2.LINE_AA)

    if rgb_path:
        with Image.open(rgb_path) as img:
            img = img.convert("RGB")
            photo = np.array(img.resize((max(1, round(img.width * H / img.height)), H), Image.Resampling.BILINEAR))
    else:
        photo = np.full((H, H * 3 // 4, 3), 255, dtype=np.uint8)
    return np.hstack([canvas, photo])
