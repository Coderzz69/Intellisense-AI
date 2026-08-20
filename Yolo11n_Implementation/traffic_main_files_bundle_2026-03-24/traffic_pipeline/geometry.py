from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def point_in_polygon(
    point: Tuple[float, float], polygon: Sequence[Tuple[float, float]]
) -> bool:
    x, y = point
    inside = False
    count = len(polygon)
    if count < 3:
        return False

    j = count - 1
    for i in range(count):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x
            < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def denormalize_polygon(
    polygon: Sequence[Tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> List[Tuple[float, float]]:
    return [
        (point_x * frame_width, point_y * frame_height)
        for point_x, point_y in polygon
    ]


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_anchor(
    bbox: Tuple[float, float, float, float],
    anchor: str = "center",
) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2.0
    if anchor == "center":
        return (center_x, (y1 + y2) / 2.0)
    if anchor == "bottom_center":
        return (center_x, y2)
    raise ValueError(
        "Unsupported lane assignment anchor. "
        "Use 'center' or 'bottom_center'."
    )


def assign_lane(
    bbox: Tuple[float, float, float, float],
    frame_shape: Tuple[int, int, int],
    lane_polygons: Dict[str, List[Tuple[float, float]]] | None = None,
    anchor: str = "center",
    require_lane_polygon_match: bool = False,
) -> Optional[str]:
    frame_height, frame_width = frame_shape[:2]
    point = bbox_anchor(bbox, anchor)

    if lane_polygons:
        for lane, polygon in lane_polygons.items():
            pixel_polygon = denormalize_polygon(polygon, frame_width, frame_height)
            if point_in_polygon(point, pixel_polygon):
                return lane
        if require_lane_polygon_match:
            return None

    dx = point[0] - (frame_width / 2.0)
    dy = point[1] - (frame_height / 2.0)
    if abs(dx) > abs(dy):
        return "E" if dx >= 0 else "W"
    return "S" if dy >= 0 else "N"


def lane_queue_span_pixels(
    bboxes: Iterable[Tuple[float, float, float, float]], lane: str
) -> float:
    lane_boxes = list(bboxes)
    if not lane_boxes:
        return 0.0

    if lane in {"N", "S"}:
        start = min(box[1] for box in lane_boxes)
        end = max(box[3] for box in lane_boxes)
    else:
        start = min(box[0] for box in lane_boxes)
        end = max(box[2] for box in lane_boxes)
    return max(0.0, end - start)
