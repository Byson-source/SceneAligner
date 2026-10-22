"""Structured3D: floorplans drawn from the annotations, cameras and ground-truth geometry.

    python -m scenealigner.structured3d <scene directory> <image>   # draws the floorplan of a scene
"""
import io
import json
import os

import matplotlib
import numpy as np
from PIL import Image, ImageOps
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from .preprocess import resize_plan

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MM_PER_PIXEL = 10.0  # of the drawn floorplan, before it is resized
CANVAS = 80000  # mm: the drawing covers 40 m around the origin of a scene
TEST_SCENES = range(3250, 3500)


def convert_lines_to_vertices(lines):
    """Chains line segments (pairs of junctions) into polygons, as the Structured3D toolbox does. Polygons of
    fewer than three junctions are dropped."""
    polygons, polygon = [], None

    def close(polygon):
        polygon = [v for i, v in enumerate(polygon) if i == 0 or v != polygon[i - 1]]
        if len(set(polygon)) >= 3:
            polygons.append(polygon)

    lines = np.array(lines, dtype=int)
    while len(lines) != 0:
        if polygon is None:
            polygon = lines[0].tolist()
            lines = np.delete(lines, 0, axis=0)
            continue
        line_ids, columns = np.where(lines == polygon[-1])
        if len(line_ids) == 0:
            close(polygon)
            polygon = None
            continue
        vertex = lines[line_ids[0], 1 - columns[0]]
        lines = np.delete(lines, line_ids[0], axis=0)
        if vertex in polygon:
            close(polygon)
            polygon = None
        else:
            polygon.append(vertex)
    if polygon is not None:
        close(polygon)
    return polygons


def load_polygons(annos):
    """Outlines of the rooms, doors and windows (their floor planes) and of the outer wall, as junction indices."""
    def junction_pairs(line_ids):
        pairs = [np.where(np.array(annos["lineJunctionMatrix"][line_id]))[0].tolist() for line_id in line_ids]
        return [pair for pair in pairs if len(pair) == 2]

    def lines_of(plane_id):
        return np.where(np.array(annos["planeLineMatrix"][plane_id]))[0].tolist()

    polygons = []
    for semantic in annos["semantics"]:
        for plane_id in semantic["planeID"]:
            if annos["planes"][plane_id]["type"] == "floor":
                polygon = convert_lines_to_vertices(junction_pairs(lines_of(plane_id)))
                if polygon:
                    polygons.append((polygon[0], semantic["type"]))

    # Outer wall: its lines on the floor, without those of doors and windows
    holes = [line_id for semantic in annos["semantics"] if semantic["type"] in ("window", "door")
             for plane_id in semantic["planeID"] for line_id in lines_of(plane_id)]
    on_floor = np.where(np.isclose(np.array([junction["coordinate"][2] for junction in annos["junctions"]]), 0))[0]
    outer_lines = []
    for semantic in annos["semantics"]:
        if semantic["type"] == "outwall":
            for plane_id in semantic["planeID"]:
                outer_lines += [pair for pair in junction_pairs(np.setdiff1d(lines_of(plane_id), np.unique(holes)))
                                if pair[0] in on_floor and pair[1] in on_floor]
    outer_polygon = convert_lines_to_vertices(outer_lines)
    if outer_polygon:
        polygons.append((outer_polygon[0], "outwall"))
    return polygons


def render_floorplan(annos, dpi=100):
    """Draws the floorplan of a scene at 1 cm per pixel, cropped to its content: walls black, rooms white. Doors
    and windows are not drawn. Returns the image and the position of its top-left corner, in pixels from the
    origin of the scene."""
    junctions = np.array([junction["coordinate"][:2] for junction in annos["junctions"]])
    rooms, outer_wall = [], None
    for indices, polygon_type in load_polygons(annos):
        polygon = Polygon(junctions[np.array(indices, dtype=int)])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue
        if polygon_type == "outwall":
            outer_wall = polygon
        elif polygon_type not in ("door", "window"):
            rooms += list(polygon.geoms) if isinstance(polygon, MultiPolygon) else [polygon]
    rooms = unary_union(rooms)  # no seams between adjacent rooms
    if isinstance(rooms, GeometryCollection):
        rooms = unary_union([g for g in rooms.geoms if isinstance(g, (Polygon, MultiPolygon))])

    def parts(shape):
        if shape is None or shape.is_empty:
            return []
        return [g for g in getattr(shape, "geoms", [shape]) if isinstance(g, Polygon) and not g.is_empty]

    size = CANVAS / MM_PER_PIXEL / dpi
    _, ax = plt.subplots(figsize=(size, size), dpi=dpi)
    ax.set_xlim(-CANVAS / 2, CANVAS / 2)
    ax.set_ylim(-CANVAS / 2, CANVAS / 2)
    ax.set_aspect("equal")
    line_style = {"color": "black", "linewidth": 1, "solid_capstyle": "round", "solid_joinstyle": "round", "antialiased": True}
    for polygon in parts(outer_wall):
        ax.fill(*polygon.exterior.xy, color="black", zorder=1)
    for polygon in parts(rooms):
        ax.fill(*polygon.exterior.xy, color="white", edgecolor="none", zorder=2)
        ax.plot(*polygon.exterior.xy, **line_style, zorder=3)
    for polygon in parts(outer_wall):
        ax.plot(*polygon.exterior.xy, **line_style, zorder=4)
    ax.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buffer = io.BytesIO()
    plt.savefig(buffer, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close()
    img = Image.open(buffer).convert("RGB")
    box = ImageOps.invert(img.convert("L")).getbbox()
    return img.crop(box), (box[0] - img.width / 2, box[1] - img.height / 2)


def load_floorplan(scene_dir, plan_size=512):
    """Floorplan of a scene, resized as the floorplans of C3, and the parameters of world_to_plan."""
    with open(os.path.join(scene_dir, "annotation_3d.json")) as f:
        annos = json.load(f)
    plan_img, origin = render_floorplan(annos)
    plan_img, (scale, off_x, off_y) = resize_plan(plan_img, plan_size)
    return plan_img, (origin, scale, (off_x, off_y))


def world_to_plan(points, plan_params):
    """World coordinates (..., 2 or 3) in mm -> floorplan pixels (..., 2). The rows of the floorplan run along -y."""
    (delta_x, delta_y), scale, (off_x, off_y) = plan_params
    return np.stack([(points[..., 0] / MM_PER_PIXEL - delta_x) * scale + off_x,
                     (-points[..., 1] / MM_PER_PIXEL - delta_y) * scale + off_y], axis=-1)


def list_views(scene_dir):
    """Directories of the furnished perspective views of a scene, in order."""
    rendering_dir = os.path.join(scene_dir, "2D_rendering")
    views = []
    for room_id in sorted(os.listdir(rendering_dir)):
        room_dir = os.path.join(rendering_dir, room_id, "perspective", "full")
        if os.path.isdir(room_dir):
            views += [os.path.join(room_dir, position_id) for position_id in sorted(os.listdir(room_dir))]
    return [view for view in views if all(os.path.exists(os.path.join(view, name)) for name in ("camera_pose.txt", "depth.png", "rgb_rawlight.png"))]


def load_camera(view_dir):
    """Camera of a view: position (mm), viewing direction, up direction and half fields of view (rad)."""
    with open(os.path.join(view_dir, "camera_pose.txt")) as f:
        values = np.array(f.read().split(), dtype=np.float64)
    return values[0:3], values[3:6], values[6:9], values[9], values[10]


if __name__ == "__main__":
    import sys

    with open(os.path.join(sys.argv[1], "annotation_3d.json")) as f:
        render_floorplan(json.load(f))[0].save(sys.argv[2])
