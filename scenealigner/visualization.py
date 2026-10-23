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



RAINBOW = ((230, 0, 0), (255, 140, 0), (240, 220, 0), (0, 170, 0), (0, 90, 255), (75, 0, 160), (180, 0, 220))  # 7 colours


def rainbow(values):
    """Colours (N, 3) of values in [0, 1]: seven bins from red (0) to violet (1)."""
    return np.array(RAINBOW)[np.clip((np.asarray(values) * len(RAINBOW)).astype(int), 0, len(RAINBOW) - 1)]


def draw_colored_points(img, points, colors, radius=3):
    """Points with a black outline, one colour each."""
    canvas = np.ascontiguousarray(img.copy())
    for (x, y), color in zip(np.round(points).astype(int), colors):
        cv2.circle(canvas, (int(x), int(y)), radius, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (int(x), int(y)), radius - 1, tuple(int(c) for c in color), -1, lineType=cv2.LINE_AA)
    return canvas


def draw_trajectory(img, points, color=(90, 90, 90), radius=2):
    """Camera positions (N, 2) joined by a line."""
    pts = np.round(points).astype(np.int32)
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
    for x, y in pts:
        cv2.circle(img, (int(x), int(y)), radius, color, -1, lineType=cv2.LINE_AA)
    return img


def draw_correspondence_rows(rgb, rows, density_img, plan_img, camera_px=None, camera_dir=None, trajectory_density=None,
                             trajectory_plan=None, margin_px=60):
    """Rows of correspondences, each [photo | density map | floorplan cropped around all the points of the rows and
    the trajectory]: a point has the same colour in the three images (the photo shows the points of the rows that
    have rgb_px). The cameras of the window (trajectory, grey) are drawn on the density map and the floorplan.

    rgb: [H, W, 3] uint8; rows: dicts {"rgb_px" ([N, 2] or None), "density_px", "plan_px" ([N, 2] pixels), "colors"
    ([N, 3]), "label" (str)}; density_img, plan_img: [h, w(, 3)] uint8; camera_px, camera_dir: [2] position and
    viewing direction of the camera on the floorplan, or None; trajectory_density, trajectory_plan: [S, 2] or None.
    Returns an RGB array.
    """
    H = rgb.shape[0]
    plan = plan_img if plan_img.ndim == 3 else np.repeat(plan_img[..., None], 3, axis=2)
    anchors = np.vstack([row["plan_px"] for row in rows] + ([np.asarray(camera_px)[None]] if camera_px is not None else [])
                        + ([np.asarray(trajectory_plan)] if trajectory_plan is not None else []))
    x0, y0 = np.maximum(np.floor(anchors.min(axis=0)).astype(int) - margin_px, 0)
    x1, y1 = np.minimum(np.ceil(anchors.max(axis=0)).astype(int) + margin_px, [plan.shape[1], plan.shape[0]])
    plan_zoom = H / (y1 - y0)  # crop scaled to the height of the photo
    plan_crop = cv2.resize(plan[y0:y1, x0:x1], (max(1, round((x1 - x0) * plan_zoom)), H), interpolation=cv2.INTER_AREA)
    density_zoom = H / density_img.shape[0]
    density = cv2.resize(density_img, (round(density_img.shape[1] * density_zoom), H), interpolation=cv2.INTER_AREA)
    if trajectory_density is not None:
        density = draw_trajectory(np.ascontiguousarray(density), np.asarray(trajectory_density) * density_zoom)
    if trajectory_plan is not None:
        plan_crop = draw_trajectory(np.ascontiguousarray(plan_crop), (np.asarray(trajectory_plan) - [x0, y0]) * plan_zoom)

    lines = []
    for row in rows:
        photo = rgb if row["rgb_px"] is None else draw_colored_points(rgb, row["rgb_px"], row["colors"])
        crop = draw_colored_points(plan_crop, (row["plan_px"] - [x0, y0]) * plan_zoom, row["colors"])
        if camera_px is not None:
            center = (np.asarray(camera_px) - [x0, y0]) * plan_zoom
            if camera_dir is not None:
                draw_camera_frustum(crop, center, np.asarray(camera_dir, dtype=np.float64), BLUE, length=40, width=30)
            cv2.circle(crop, tuple(int(v) for v in np.round(center)), 6, BLUE, -1, lineType=cv2.LINE_AA)
        panels = [photo, draw_colored_points(density, np.asarray(row["density_px"]) * density_zoom, row["colors"]), crop]
        line = np.hstack([x for panel in panels for x in (panel, np.full((H, 8, 3), 255, dtype=np.uint8))][:-1])
        cv2.putText(line, row["label"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(line, row["label"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        lines.append(line)
    gap = np.full((8, lines[0].shape[1], 3), 255, dtype=np.uint8)
    return np.vstack([x for line in lines for x in (line, gap)][:-1])
