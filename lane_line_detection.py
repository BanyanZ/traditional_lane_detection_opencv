from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from lane_ml import (
    cnn_lane_mask,
    kmeans_lane_seeds,
    ransac_line,
    ransac_polynomial,
)


IMAGE_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class LaneSegment:
    x1: int
    y1: int
    x2: int
    y2: int
    slope: float
    length: float
    x_bottom: float


@dataclass
class LaneCluster:
    segments: list[LaneSegment]
    fitted_line: tuple[int, int, int, int]
    curve_points: list[tuple[int, int]] = field(default_factory=list)
    is_curve: bool = False


@dataclass(frozen=True)
class DetectionProfile:
    key: str
    label: str
    roi_top_ratio: float = 0.43
    clahe_clip: float = 2.0
    white_min: int = 170
    white_percentile: float = 92.0
    white_hsv_min: int = 165
    saturation_max: int = 95
    yellow_s_min: int = 45
    yellow_v_min: int = 90
    channel_diff_min: int = 16
    channel_diff_percentile: float = 88.0
    channel_diff_std_scale: float = 0.35
    canny_low: int = 50
    canny_high: int = 150
    color_dilate: int = 5
    morph_open_size: int = 3
    morph_close_size: int = 5
    glare_block_size: int = 20
    glare_min_brightness: int = 225
    glare_fill_min: float = 0.32
    glare_max_aspect: float = 2.35
    line_merge_kernel: int = 13
    component_min_area: int = 5
    component_min_length: int = 7
    component_max_area_ratio: float = 0.012
    use_mask_components: bool = True
    mask_component_max_area_ratio: float = 0.035
    max_lanes: int = 4
    min_angle: float = 10.0
    left_bottom_limit: float = -0.75
    right_bottom_limit: float = 1.62
    overlay_alpha: float = 0.62
    hough_passes: tuple[tuple[int, int, int], ...] = (
        (18, 16, 8),
        (12, 14, 7),
        (8, 10, 5),
    )
    use_structure_edges: bool = False
    structure_hough_passes: tuple[tuple[int, int, int], ...] = (
        (24, 30, 14),
        (16, 24, 10),
    )
    structure_min_angle: float = 12.0
    structure_max_angle: float = 75.0
    structure_dilate: int = 3
    # Machine-learning extensions ------------------------------------------------
    enable_ransac: bool = True
    enable_kmeans: bool = True
    enable_curve_fit: bool = False
    curve_polyfit_degree: int = 2
    enable_cnn: bool = False
    cnn_blend: float = 0.55


DEFAULT_PROFILE = DetectionProfile(key="normal", label="正常光照")


def imread_unicode(path: Path) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Failed to encode output image: {path}")
    buffer.tofile(str(path))


def list_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def region_of_interest(
    shape: tuple[int, int],
    top_ratio: float = 0.43,
) -> np.ndarray:
    height, width = shape
    top_y = int(height * top_ratio)
    polygon = np.array(
        [
            [
                (int(width * 0.03), height - 1),
                (int(width * 0.24), top_y),
                (int(width * 0.76), top_y),
                (int(width * 0.97), height - 1),
            ]
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, polygon, 255)
    return mask


def keep_lane_like_components(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)
    max_small_area = height * width * 0.0025
    max_long_area = height * width * 0.030

    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if area < 8:
            continue
        elongation = max(w, h) / max(1, min(w, h))
        is_small_mark = area <= max_small_area
        is_long_mark = elongation >= 2.25 and area <= max_long_area
        if is_small_mark or is_long_mark:
            filtered[labels == idx] = 255
    return filtered


def adaptive_threshold_from_roi(
    values: np.ndarray,
    floor: float,
    percentile: float,
    std_scale: float = 0.0,
) -> int:
    if values.size == 0:
        return int(floor)
    threshold = float(np.percentile(values, percentile))
    if std_scale:
        threshold += float(np.std(values)) * std_scale
    return int(np.clip(max(floor, threshold), 0, 255))


def rgb_channel_difference_mask(
    image: np.ndarray,
    roi: np.ndarray,
    gray: np.ndarray,
    profile: DetectionProfile,
) -> tuple[np.ndarray, np.ndarray]:
    b, g, r = cv2.split(image)
    green_over_blue = cv2.subtract(g, b)
    red_over_blue = cv2.subtract(r, b)
    yellow_score = cv2.min(green_over_blue, red_over_blue)
    diff_score = cv2.max(green_over_blue, yellow_score)

    roi_values = diff_score[roi > 0]
    threshold = adaptive_threshold_from_roi(
        roi_values,
        profile.channel_diff_min,
        profile.channel_diff_percentile,
        profile.channel_diff_std_scale,
    )
    if roi_values.size == 0 or int(roi_values.max()) < threshold:
        return np.zeros_like(gray), diff_score

    diff_mask = cv2.inRange(diff_score, threshold, 255)
    gray_values = gray[roi > 0]
    brightness_floor = adaptive_threshold_from_roi(gray_values, 70, 45.0)
    diff_mask = cv2.bitwise_and(diff_mask, cv2.inRange(gray, brightness_floor, 255))
    diff_mask = cv2.bitwise_and(diff_mask, roi)
    return diff_mask, diff_score


def remove_glare_components(
    mask: np.ndarray,
    gray: np.ndarray,
    roi: np.ndarray,
    profile: DetectionProfile,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return mask

    filtered = mask.copy()
    roi_gray = gray[roi > 0]
    glare_threshold = adaptive_threshold_from_roi(
        roi_gray,
        profile.glare_min_brightness,
        98.0,
    )
    block = max(8, int(profile.glare_block_size))

    for idx in range(1, count):
        x, y, w, h, area = stats[idx]
        if w < block or h < block:
            continue

        aspect = max(w, h) / max(1, min(w, h))
        fill_ratio = area / max(1, w * h)
        if aspect > profile.glare_max_aspect or fill_ratio < profile.glare_fill_min:
            continue

        component_pixels = labels == idx
        bright_ratio = float(np.mean(gray[component_pixels] >= glare_threshold))
        mean_brightness = float(np.mean(gray[component_pixels]))
        if bright_ratio >= 0.42 or mean_brightness >= glare_threshold - 6:
            filtered[component_pixels] = 0

    return filtered


def merge_lane_fragments(
    mask: np.ndarray,
    roi: np.ndarray,
    profile: DetectionProfile,
) -> np.ndarray:
    kernel_size = int(profile.line_merge_kernel)
    if kernel_size < 3:
        return cv2.bitwise_and(mask, roi)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernels: list[np.ndarray] = []
    for start, end in (((0, kernel_size - 1), (kernel_size - 1, 0)), ((0, 0), (kernel_size - 1, kernel_size - 1))):
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)
        cv2.line(kernel, start, end, 1, 1)
        kernels.append(kernel)

    merged = mask.copy()
    for kernel in kernels:
        merged = cv2.bitwise_or(merged, cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel))

    small_close = max(3, min(7, kernel_size // 2))
    merged = cv2.morphologyEx(
        merged,
        cv2.MORPH_CLOSE,
        np.ones((small_close, small_close), np.uint8),
    )
    return cv2.bitwise_and(merged, roi)


def make_lane_mask(
    image: np.ndarray,
    profile: DetectionProfile = DEFAULT_PROFILE,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=profile.clahe_clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    roi = region_of_interest((height, width), profile.roi_top_ratio)
    road_pixels = enhanced[roi > 0]
    adaptive_white = (
        int(max(profile.white_min, np.percentile(road_pixels, profile.white_percentile)))
        if road_pixels.size
        else profile.white_min
    )

    white_gray = cv2.inRange(enhanced, adaptive_white, 255)
    white_hsv = cv2.inRange(v, max(profile.white_hsv_min, adaptive_white - 10), 255) & cv2.inRange(
        s, 0, profile.saturation_max
    )
    yellow_hsv = (
        cv2.inRange(h, 12, 40)
        & cv2.inRange(s, profile.yellow_s_min, 255)
        & cv2.inRange(v, profile.yellow_v_min, 255)
    )
    diff_mask, diff_score = rgb_channel_difference_mask(image, roi, gray, profile)

    color_mask = cv2.bitwise_or(
        cv2.bitwise_or(white_gray, white_hsv),
        cv2.bitwise_or(yellow_hsv, diff_mask),
    )
    color_mask = cv2.bitwise_and(color_mask, roi)
    open_size = max(1, int(profile.morph_open_size))
    close_size = max(1, int(profile.morph_close_size))
    if open_size > 1:
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8)
        )
    if close_size > 1:
        color_mask = cv2.morphologyEx(
            color_mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8)
        )
    color_mask = remove_glare_components(color_mask, gray, roi, profile)
    color_mask = merge_lane_fragments(color_mask, roi, profile)
    if float(np.mean(gray)) < 85:
        color_mask = keep_lane_like_components(color_mask)

    edges = cv2.Canny(blur, profile.canny_low, profile.canny_high)
    expanded_color = cv2.dilate(
        color_mask, np.ones((profile.color_dilate, profile.color_dilate), np.uint8), iterations=1
    )
    lane_edges = cv2.bitwise_and(edges, expanded_color)
    if np.any(diff_mask):
        diff_blur = cv2.GaussianBlur(diff_score, (5, 5), 0)
        diff_edges = cv2.Canny(
            diff_blur,
            max(12, profile.canny_low // 2),
            max(30, profile.canny_high // 2),
        )
        lane_edges = cv2.bitwise_or(lane_edges, cv2.bitwise_and(diff_edges, expanded_color))
    lane_edges = cv2.bitwise_and(lane_edges, roi)

    if profile.use_structure_edges:
        support_edges = cv2.bitwise_and(edges, roi)
        support_mask = np.zeros_like(lane_edges)
        for threshold, min_length, max_gap in profile.structure_hough_passes:
            lines = cv2.HoughLinesP(
                support_edges,
                rho=1,
                theta=np.pi / 180,
                threshold=threshold,
                minLineLength=min_length,
                maxLineGap=max_gap,
            )
            if lines is None:
                continue
            for item in lines[:, 0, :]:
                x1, y1, x2, y2 = map(int, item)
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0:
                    slope = 999.0
                    angle = 90.0
                else:
                    slope = dy / dx
                    angle = abs(math.degrees(math.atan(slope)))
                if angle < profile.structure_min_angle or angle > profile.structure_max_angle:
                    continue

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                length = math.hypot(dx, dy)
                if float(np.mean(gray)) < 90:
                    if center_x < width * 0.38 and center_y < height * 0.78:
                        continue
                    if length > width * 0.30 and center_y < height * 0.76:
                        continue

                if center_x < width * 0.45 and slope > 0:
                    continue
                if center_x > width * 0.55 and slope < 0:
                    continue

                xb = x_at_y(LaneSegment(x1, y1, x2, y2, slope, length, 0.0), height - 1)
                if xb < width * profile.left_bottom_limit or xb > width * profile.right_bottom_limit:
                    continue
                if width * 0.18 < xb < width * 0.82:
                    continue

                cv2.line(support_mask, (x1, y1), (x2, y2), 255, profile.structure_dilate)
        lane_edges = cv2.bitwise_or(lane_edges, support_mask)

    if profile.enable_cnn:
        cnn_mask = cnn_lane_mask(image)
        if cnn_mask is not None:
            cnn_mask = cv2.bitwise_and(cnn_mask, roi)
            lane_edges = cv2.bitwise_or(lane_edges, cnn_mask)
            color_mask = cv2.bitwise_or(color_mask, cnn_mask)

    return lane_edges, color_mask


def x_at_y(seg: LaneSegment, y: int) -> float:
    if abs(seg.slope) < 1e-6:
        return (seg.x1 + seg.x2) / 2
    intercept = seg.y1 - seg.slope * seg.x1
    return (y - intercept) / seg.slope


def segment_angle(seg: LaneSegment) -> float:
    if abs(seg.slope) >= 998:
        return 90.0
    return abs(math.degrees(math.atan(seg.slope)))


def is_shallow_mark(seg: LaneSegment, image_shape: tuple[int, int]) -> bool:
    height, width = image_shape
    angle = segment_angle(seg)
    center_x = (seg.x1 + seg.x2) / 2
    center_y = (seg.y1 + seg.y2) / 2
    return (
        4.0 <= angle < 12.0
        and seg.length >= max(12, width * 0.035)
        and seg.length <= width * 0.22
        and height * 0.40 <= center_y <= height * 0.70
        and width * 0.16 <= center_x <= width * 0.84
    )


def detect_segments(
    edge_mask: np.ndarray,
    profile: DetectionProfile = DEFAULT_PROFILE,
    component_mask: np.ndarray | None = None,
) -> list[LaneSegment]:
    height, width = edge_mask.shape[:2]
    segments: list[LaneSegment] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add_segment(x1: int, y1: int, x2: int, y2: int, min_length: int) -> None:
        key = tuple(sorted(((x1, y1), (x2, y2))))
        flat_key = (key[0][0], key[0][1], key[1][0], key[1][1])
        if flat_key in seen:
            return

        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            slope = 999.0
            angle = 90.0
        else:
            slope = dy / dx
            angle = abs(math.degrees(math.atan(slope)))

        length = math.hypot(dx, dy)
        if length < min_length or angle > 86:
            return

        raw = LaneSegment(x1, y1, x2, y2, slope, length, 0.0)
        shallow_mark = is_shallow_mark(raw, (height, width))
        if angle < profile.min_angle and not shallow_mark:
            return
        center_x = (x1 + x2) / 2
        if not shallow_mark and angle >= 12:
            if center_x < width * 0.45 and slope > 0:
                return
            if center_x > width * 0.55 and slope < 0:
                return

        if shallow_mark:
            xb = center_x
        else:
            xb = x_at_y(raw, height - 1)
            if xb < width * profile.left_bottom_limit or xb > width * profile.right_bottom_limit:
                return

        seen.add(flat_key)
        segments.append(LaneSegment(x1, y1, x2, y2, slope, length, xb))

    def add_lines(lines: np.ndarray | None, min_length: int) -> None:
        if lines is None:
            return

        for item in lines[:, 0, :]:
            x1, y1, x2, y2 = map(int, item)
            add_segment(x1, y1, x2, y2, min_length)

    def add_component_segments(mask: np.ndarray, max_area_ratio: float) -> None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        max_area = max(profile.component_min_area, int(height * width * max_area_ratio))
        min_length = max(4, int(profile.component_min_length))

        for idx in range(1, count):
            x, y, w, h, area = stats[idx]
            elongation = max(w, h) / max(1, min(w, h))
            if area < profile.component_min_area or area > max_area:
                continue
            if max(w, h) < min_length or w > width * 0.55 or h > height * 0.55:
                continue
            if elongation < 1.35 and area > height * width * 0.0012:
                continue

            pts_yx = np.column_stack(np.where(labels == idx))
            if pts_yx.shape[0] < profile.component_min_area:
                continue

            pts = pts_yx[:, ::-1].astype(np.float32)
            vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
            direction = np.array([vx, vy], dtype=np.float32)
            origin = np.array([x0, y0], dtype=np.float32)
            projection = (pts - origin) @ direction
            p1 = origin + direction * float(np.min(projection))
            p2 = origin + direction * float(np.max(projection))
            add_segment(
                int(round(p1[0])),
                int(round(p1[1])),
                int(round(p2[0])),
                int(round(p2[1])),
                min_length,
            )

    for threshold, min_length, max_gap in profile.hough_passes:
        lines = cv2.HoughLinesP(
            edge_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_length,
            maxLineGap=max_gap,
        )
        add_lines(lines, min_length)

    add_component_segments(edge_mask, profile.component_max_area_ratio)
    if profile.use_mask_components and component_mask is not None:
        add_component_segments(component_mask, profile.mask_component_max_area_ratio)
    return segments


def cluster_segments(segments: list[LaneSegment], image_shape: tuple[int, int]) -> list[list[LaneSegment]]:
    if not segments:
        return []

    height, width = image_shape
    max_gap = max(34, int(width * 0.11))
    normal_segments = [seg for seg in segments if not is_shallow_mark(seg, (height, width))]
    shallow_segments = [seg for seg in segments if is_shallow_mark(seg, (height, width))]
    ordered = sorted(normal_segments, key=lambda seg: (seg.slope > 0, seg.x_bottom))
    clusters: list[list[LaneSegment]] = []

    def cluster_x_at_y(cluster: list[LaneSegment], y: float) -> float | None:
        xs = []
        for item in cluster:
            if abs(item.slope) < 1e-6:
                continue
            x = x_at_y(item, int(y))
            if -width <= x <= width * 2:
                xs.append(x)
        if not xs:
            return None
        return float(np.mean(xs))

    for seg in ordered:
        placed = False
        for cluster in clusters:
            avg_x = float(np.mean([s.x_bottom for s in cluster]))
            avg_slope = float(np.mean([s.slope for s in cluster]))
            avg_angle = float(np.mean([segment_angle(s) for s in cluster]))
            same_side = (avg_slope >= 0) == (seg.slope >= 0)
            center_x = (seg.x1 + seg.x2) / 2
            center_y = (seg.y1 + seg.y2) / 2
            predicted_x = cluster_x_at_y(cluster, center_y)
            near_fitted_line = (
                predicted_x is not None
                and abs(center_x - predicted_x) <= max(18, width * 0.06)
                and abs(segment_angle(seg) - avg_angle) <= 18
            )
            if same_side and (abs(seg.x_bottom - avg_x) <= max_gap or near_fitted_line):
                cluster.append(seg)
                placed = True
                break
        if not placed:
            clusters.append([seg])

    for seg in shallow_segments:
        center_x = (seg.x1 + seg.x2) / 2
        center_y = (seg.y1 + seg.y2) / 2
        best_index: int | None = None
        best_distance = float("inf")
        for idx, cluster in enumerate(clusters):
            if not cluster or float(np.mean([segment_angle(s) for s in cluster])) < 12:
                continue
            predicted_x = cluster_x_at_y(cluster, center_y)
            if predicted_x is None:
                continue
            distance = abs(center_x - predicted_x)
            if distance < best_distance:
                best_distance = distance
                best_index = idx

        if best_index is not None and best_distance <= max(24, width * 0.08):
            clusters[best_index].append(seg)

    return [c for c in clusters if sum(s.length for s in c) >= 24]


def kmeans_post_split(
    cluster: list[LaneSegment], image_shape: tuple[int, int]
) -> list[list[LaneSegment]]:
    """Use cv2.kmeans on segment bottom-x to split lanes that were merged by chance."""
    if len(cluster) < 6:
        return [cluster]
    height, width = image_shape
    avg_slope = float(np.mean([s.slope for s in cluster]))
    if abs(avg_slope) < 0.05:
        return [cluster]

    xs = np.array([[seg.x_bottom] for seg in cluster], dtype=np.float32)
    spread = float(np.std(xs))
    if spread < max(20, width * 0.06):
        return [cluster]

    seeds_pts = np.array([[seg.x_bottom, (seg.y1 + seg.y2) / 2] for seg in cluster], dtype=np.float32)
    seeds = kmeans_lane_seeds(seeds_pts, max_clusters=3, image_width=width)
    if len(seeds) < 2:
        return [cluster]

    output: list[list[LaneSegment]] = [[] for _ in seeds]
    for seg in cluster:
        best_idx = 0
        best_dist = float("inf")
        for idx, group in enumerate(seeds):
            if group.size == 0:
                continue
            target = float(np.mean(group[:, 0]))
            dist = abs(seg.x_bottom - target)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        output[best_idx].append(seg)

    output = [group for group in output if len(group) >= 2]
    return output if len(output) > 1 else [cluster]


def cluster_x_at_y(segments: list[LaneSegment], y: float, image_width: int) -> float | None:
    xs = []
    for item in segments:
        if abs(item.slope) < 1e-6:
            continue
        x = x_at_y(item, int(y))
        if -image_width <= x <= image_width * 2:
            xs.append(x)
    if not xs:
        return None
    return float(np.mean(xs))


def clusters_are_close(
    left: list[LaneSegment],
    right: list[LaneSegment],
    image_shape: tuple[int, int, int],
) -> bool:
    height, width = image_shape[:2]
    left_slope = float(np.mean([s.slope for s in left]))
    right_slope = float(np.mean([s.slope for s in right]))
    if (left_slope >= 0) != (right_slope >= 0):
        return False

    left_angle = float(np.mean([segment_angle(s) for s in left]))
    right_angle = float(np.mean([segment_angle(s) for s in right]))
    if abs(left_angle - right_angle) > 18:
        return False

    probe_y_values = (height * 0.55, height * 0.70, height * 0.84)
    distances = []
    for y in probe_y_values:
        lx = cluster_x_at_y(left, y, width)
        rx = cluster_x_at_y(right, y, width)
        if lx is not None and rx is not None:
            distances.append(abs(lx - rx))
    if not distances:
        return False
    return min(distances) <= max(18, width * 0.075)


def cluster_y_range(segments: list[LaneSegment]) -> tuple[int, int]:
    y_values = [s.y1 for s in segments] + [s.y2 for s in segments]
    return min(y_values), max(y_values)


def clusters_are_path_continuation(
    left: list[LaneSegment],
    right: list[LaneSegment],
    image_shape: tuple[int, int, int],
) -> bool:
    height, width = image_shape[:2]
    left_angle = float(np.mean([segment_angle(s) for s in left]))
    right_angle = float(np.mean([segment_angle(s) for s in right]))
    left_slope = float(np.mean([s.slope for s in left]))
    right_slope = float(np.mean([s.slope for s in right]))

    if (left_slope >= 0) != (right_slope >= 0) and min(left_angle, right_angle) > 18:
        return False
    if abs(left_angle - right_angle) > 32 and min(left_angle, right_angle) > 16:
        return False

    left_min_y, left_max_y = cluster_y_range(left)
    right_min_y, right_max_y = cluster_y_range(right)
    overlap_start = max(left_min_y, right_min_y)
    overlap_end = min(left_max_y, right_max_y)
    probe_values: list[float] = []

    if overlap_end - overlap_start >= height * 0.045:
        probe_values.extend(
            [
                overlap_start + (overlap_end - overlap_start) * 0.35,
                overlap_start + (overlap_end - overlap_start) * 0.65,
            ]
        )
    else:
        gap = max(0, max(left_min_y, right_min_y) - min(left_max_y, right_max_y))
        if gap > height * 0.22:
            return False
        probe_values.append((left_min_y + left_max_y + right_min_y + right_max_y) / 4)

    distances = []
    for y in probe_values:
        lx = cluster_x_at_y(left, y, width)
        rx = cluster_x_at_y(right, y, width)
        if lx is not None and rx is not None:
            distances.append(abs(lx - rx))

    if not distances:
        return False

    return min(distances) <= max(24, width * 0.105)


def lane_quality_score(lane: LaneCluster, image_shape: tuple[int, int, int]) -> float:
    height, width = image_shape[:2]
    total_len = sum(s.length for s in lane.segments)
    y_values = [s.y1 for s in lane.segments] + [s.y2 for s in lane.segments]
    span = max(y_values) - min(y_values)
    max_y = max(y_values)
    avg_x_bottom = float(np.mean([s.x_bottom for s in lane.segments]))
    outside = max(0.0, -avg_x_bottom, avg_x_bottom - width)
    bottom_bonus = (max_y / max(1, height)) * 80
    curve_bonus = 15 if lane.is_curve else 0
    return total_len + span * 1.8 + bottom_bonus + curve_bonus - outside * 0.75


def lane_is_probable_noise(lane: LaneCluster, image_shape: tuple[int, int, int]) -> bool:
    height, width = image_shape[:2]
    total_len = sum(s.length for s in lane.segments)
    y_values = [s.y1 for s in lane.segments] + [s.y2 for s in lane.segments]
    min_y = min(y_values)
    max_y = max(y_values)
    span = max_y - min_y
    avg_angle = float(np.mean([segment_angle(s) for s in lane.segments]))
    avg_x_bottom = float(np.mean([s.x_bottom for s in lane.segments]))
    shallow_count = sum(1 for s in lane.segments if is_shallow_mark(s, (height, width)))
    non_shallow_len = sum(
        s.length for s in lane.segments if not is_shallow_mark(s, (height, width))
    )
    lower_len = sum(s.length for s in lane.segments if max(s.y1, s.y2) >= height * 0.68)
    center_bottom = width * 0.12 <= avg_x_bottom <= width * 0.88

    if lane.is_curve and total_len > height * 0.28:
        return False
    if shallow_count == len(lane.segments):
        return True
    if shallow_count and non_shallow_len < max(26, height * 0.13):
        return True
    if avg_angle < 13 and span < height * 0.24:
        return True
    if max_y < height * 0.70 and total_len < height * 0.42:
        return True
    if avg_angle < 18 and center_bottom and total_len < height * 0.45 and lower_len < height * 0.12:
        return True
    if len(lane.segments) <= 2 and max_y < height * 0.70 and total_len < height * 0.34:
        return True
    if max_y < height * 0.58 and total_len < height * 0.50:
        return True
    if span < height * 0.12 and total_len < height * 0.55:
        return True

    return False


def lanes_are_duplicate(
    current: LaneCluster,
    selected: LaneCluster,
    image_shape: tuple[int, int, int],
) -> bool:
    height, width = image_shape[:2]
    current_slope = float(np.mean([s.slope for s in current.segments]))
    selected_slope = float(np.mean([s.slope for s in selected.segments]))
    current_angle = float(np.mean([segment_angle(s) for s in current.segments]))
    selected_angle = float(np.mean([segment_angle(s) for s in selected.segments]))

    if (current_slope >= 0) != (selected_slope >= 0) and min(current_angle, selected_angle) > 18:
        return False
    if abs(current_angle - selected_angle) > 30 and min(current_angle, selected_angle) > 16:
        return False

    distances = []
    for y in (height * 0.56, height * 0.70, height * 0.84):
        cx = cluster_x_at_y(current.segments, y, width)
        sx = cluster_x_at_y(selected.segments, y, width)
        if cx is not None and sx is not None:
            distances.append(abs(cx - sx))
    if not distances:
        return False

    return min(distances) <= max(22, width * 0.085)


def filter_lanes(
    lanes: list[LaneCluster],
    image_shape: tuple[int, int, int],
    profile: DetectionProfile = DEFAULT_PROFILE,
) -> list[LaneCluster]:
    height, width = image_shape[:2]
    filtered: list[LaneCluster] = []

    for lane in lanes:
        if not lane.segments:
            continue
        total_len = sum(s.length for s in lane.segments)
        avg_x_bottom = float(np.mean([s.x_bottom for s in lane.segments]))
        y_values = [s.y1 for s in lane.segments] + [s.y2 for s in lane.segments]
        span = max(y_values) - min(y_values)
        max_y = max(y_values)
        shallow_count = sum(1 for s in lane.segments if is_shallow_mark(s, (height, width)))

        if avg_x_bottom < -width * 0.45 or avg_x_bottom > width * 1.45:
            continue
        if total_len < max(34, height * 0.14):
            continue
        if shallow_count == len(lane.segments) and total_len < width * 0.25:
            continue
        if max_y < height * 0.52 and total_len < height * 0.45:
            continue
        if lane_is_probable_noise(lane, image_shape):
            continue

        filtered.append(lane)

    filtered.sort(key=lambda item: lane_quality_score(item, image_shape), reverse=True)
    selected: list[LaneCluster] = []
    for lane in filtered:
        if any(lanes_are_duplicate(lane, kept, image_shape) for kept in selected):
            continue
        selected.append(lane)
        if len(selected) >= max(1, int(profile.max_lanes)):
            break
    return selected


def fit_cluster_line(
    segments: list[LaneSegment],
    image_shape: tuple[int, int, int],
    profile: DetectionProfile = DEFAULT_PROFILE,
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    points = []
    for seg in segments:
        points.append([seg.x1, seg.y1])
        points.append([seg.x2, seg.y2])
    pts = np.array(points, dtype=np.float32)

    if profile.enable_ransac and pts.shape[0] >= 6:
        line = ransac_line(pts, threshold=2.5)
        if line is not None and line.inliers.sum() >= 4:
            y_values = [s.y1 for s in segments] + [s.y2 for s in segments]
            y1 = int(np.clip(min(y_values), 0, height - 1))
            y2 = int(np.clip(max(y_values), 0, height - 1))
            x1 = line.x_at_y(y1)
            x2 = line.x_at_y(y2)
            if not (math.isnan(x1) or math.isnan(x2)):
                clipped = cv2.clipLine(
                    (0, 0, width, height), (int(x1), int(y1)), (int(x2), int(y2))
                )
                if clipped[0]:
                    (x1c, y1c), (x2c, y2c) = clipped[1], clipped[2]
                    return int(x1c), int(y1c), int(x2c), int(y2c)

    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()

    y_values = [s.y1 for s in segments] + [s.y2 for s in segments]
    y1 = int(np.clip(min(y_values), 0, height - 1))
    y2 = int(np.clip(max(y_values), 0, height - 1))
    if abs(vy) < 1e-6:
        x1 = x2 = int(x0)
    else:
        x1 = int(x0 + (y1 - y0) * vx / vy)
        x2 = int(x0 + (y2 - y0) * vx / vy)

    clipped = cv2.clipLine((0, 0, width, height), (x1, y1), (x2, y2))
    if clipped[0]:
        (x1, y1), (x2, y2) = clipped[1], clipped[2]
        return int(x1), int(y1), int(x2), int(y2)

    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    return x1, y1, x2, y2


def fit_cluster_curve(
    segments: list[LaneSegment],
    image_shape: tuple[int, int, int],
    profile: DetectionProfile,
) -> tuple[list[tuple[int, int]], bool]:
    """Return polyline points and ``True`` if a curve fit looks better than a line."""
    height, width = image_shape[:2]
    pts: list[list[float]] = []
    for seg in segments:
        pts.append([seg.x1, seg.y1])
        pts.append([seg.x2, seg.y2])
    arr = np.array(pts, dtype=np.float32)

    if arr.shape[0] < 15:
        return [], False

    poly = ransac_polynomial(arr, threshold=4.0)
    if poly is None:
        return [], False

    inlier_count = int(poly.inliers.sum())
    line = ransac_line(arr, threshold=2.5)
    line_inliers = int(line.inliers.sum()) if line is not None else 0

    # A polynomial always fits at least as well as a line (it has an extra
    # degree of freedom), so a curve is only declared when *all* hold:
    #   1. the quadratic term is genuinely large (a real bend), and
    #   2. the poly is not worse than the straight line on inliers, and
    #   3. the parabola vertex falls outside the sampled span (else it loops).
    curvature = abs(poly.coeffs[2])
    if curvature < 3e-3:
        return [], False
    if inlier_count + 1 < line_inliers:
        return [], False

    ys = np.array([s.y1 for s in segments] + [s.y2 for s in segments], dtype=np.float32)
    y_min = int(np.clip(ys.min(), 0, height - 1))
    y_max = int(np.clip(ys.max(), 0, height - 1))
    if y_max - y_min < height * 0.16:
        return [], False

    c0, c1, c2 = poly.coeffs
    # Reject parabolas whose vertex (dx/dy = 0) lies inside the sampled range:
    # those switch back on themselves and render as loops, never real lanes.
    if abs(c2) > 1e-9:
        vertex_y = -c1 / (2.0 * c2)
        if y_min - 5 <= vertex_y <= y_max + 5:
            return [], False

    sample_ys = np.linspace(y_min, y_max, 16)
    sample_xs = c0 + c1 * sample_ys + c2 * sample_ys * sample_ys

    # Reject implausible horizontal swing (a lane should not wander more than
    # ~40% of the frame width across its vertical extent).
    if float(np.nanmax(sample_xs) - np.nanmin(sample_xs)) > width * 0.45:
        return [], False

    polyline: list[tuple[int, int]] = []
    for x, y in zip(sample_xs, sample_ys):
        if math.isnan(x):
            continue
        polyline.append((int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))))
    if len(polyline) < 4:
        return [], False
    return polyline, True


def build_clusters(
    segments: list[LaneSegment],
    image_shape: tuple[int, int, int],
    profile: DetectionProfile = DEFAULT_PROFILE,
) -> list[LaneCluster]:
    height, width = image_shape[:2]
    raw_clusters = cluster_segments(segments, (height, width))

    if profile.enable_kmeans:
        expanded: list[list[LaneSegment]] = []
        for cluster in raw_clusters:
            expanded.extend(kmeans_post_split(cluster, (height, width)))
        raw_clusters = expanded

    lanes: list[LaneCluster] = []
    for cluster in raw_clusters:
        if len(cluster) == 1:
            seg = cluster[0]
            if seg.length < height * 0.12 and not is_shallow_mark(seg, (height, width)):
                continue
        fitted = fit_cluster_line(cluster, image_shape, profile)
        curve_points: list[tuple[int, int]] = []
        is_curve = False
        if profile.enable_curve_fit:
            curve_points, is_curve = fit_cluster_curve(cluster, image_shape, profile)
        lanes.append(
            LaneCluster(
                segments=cluster,
                fitted_line=fitted,
                curve_points=curve_points,
                is_curve=is_curve,
            )
        )

    merged_clusters: list[list[LaneSegment]] = []
    merge_gap = max(28, int(width * 0.09))
    for lane in sorted(lanes, key=lambda item: sum(s.length for s in item.segments), reverse=True):
        avg_x = float(np.mean([s.x_bottom for s in lane.segments]))
        avg_slope = float(np.mean([s.slope for s in lane.segments]))
        placed = False
        for cluster in merged_clusters:
            cluster_x = float(np.mean([s.x_bottom for s in cluster]))
            cluster_slope = float(np.mean([s.slope for s in cluster]))
            same_side = (avg_slope >= 0) == (cluster_slope >= 0)
            if (
                same_side
                and (
                    abs(avg_x - cluster_x) <= merge_gap
                    or clusters_are_close(lane.segments, cluster, image_shape)
                )
            ) or clusters_are_path_continuation(lane.segments, cluster, image_shape):
                cluster.extend(lane.segments)
                placed = True
                break
        if not placed:
            merged_clusters.append(list(lane.segments))

    merged_lanes: list[LaneCluster] = []
    for cluster in merged_clusters:
        fitted = fit_cluster_line(cluster, image_shape, profile)
        curve_points: list[tuple[int, int]] = []
        is_curve = False
        if profile.enable_curve_fit:
            curve_points, is_curve = fit_cluster_curve(cluster, image_shape, profile)
        merged_lanes.append(
            LaneCluster(
                segments=cluster,
                fitted_line=fitted,
                curve_points=curve_points,
                is_curve=is_curve,
            )
        )
    return filter_lanes(merged_lanes, image_shape, profile)


def draw_lanes(
    image: np.ndarray,
    lanes: list[LaneCluster],
    edge_mask: np.ndarray,
    profile: DetectionProfile = DEFAULT_PROFILE,
) -> np.ndarray:
    output = image.copy()
    overlay = image.copy()
    height, width = image.shape[:2]

    for lane in lanes:
        color = (0, 220, 0)

        if lane.is_curve and len(lane.curve_points) >= 2:
            pts = np.array(lane.curve_points, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], False, (255, 180, 60), 5, cv2.LINE_AA)
            continue

        x1, y1, x2, y2 = lane.fitted_line
        if y2 == y1:
            continue
        if y2 < y1:
            x1, y1, x2, y2 = x2, y2, x1, y1
        start_y = int(np.clip(y1, 0, height - 1))
        end_y = int(np.clip(y2, 0, height - 1))
        if end_y <= start_y:
            continue
        start_x = int(x1 + (start_y - y1) * (x2 - x1) / (y2 - y1))
        end_x = int(x1 + (end_y - y1) * (x2 - x1) / (y2 - y1))
        start_x = int(np.clip(start_x, 0, width - 1))
        end_x = int(np.clip(end_x, 0, width - 1))
        cv2.line(overlay, (start_x, start_y), (end_x, end_y), color, 5, cv2.LINE_AA)

    output = cv2.addWeighted(overlay, profile.overlay_alpha, output, 1.0 - profile.overlay_alpha, 0)
    return output


def detect_lanes(
    image: np.ndarray,
    profile: DetectionProfile = DEFAULT_PROFILE,
) -> tuple[np.ndarray, list[LaneCluster], np.ndarray]:
    edge_mask, color_mask = make_lane_mask(image, profile)
    segments = detect_segments(edge_mask, profile, color_mask)
    lanes = build_clusters(segments, image.shape, profile)
    annotated = draw_lanes(image, lanes, edge_mask, profile)
    return annotated, lanes, edge_mask


def process_images(input_path: Path, output_dir: Path, use_auto_profile: bool = True) -> None:
    images = list_images(input_path)
    if not images:
        raise FileNotFoundError(f"No images found in: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    auto_detector = None
    if use_auto_profile:
        try:
            from lane_profile_router import detect_auto

            auto_detector = detect_auto
        except Exception:
            auto_detector = None

    for image_path in images:
        image = imread_unicode(image_path)
        if image is None:
            print(f"[skip] cannot read: {image_path}")
            continue

        profile_key = DEFAULT_PROFILE.key
        profile_label = DEFAULT_PROFILE.label
        if auto_detector is not None:
            annotated, lanes, _, decision = auto_detector(image)
            profile_key = decision.key
            profile_label = decision.label
        else:
            annotated, lanes, _ = detect_lanes(image)
        relative = image_path.name if input_path.is_file() else image_path.relative_to(input_path)
        save_path = (output_dir / relative).with_suffix(".png")
        imwrite_unicode(save_path, annotated)

        rows.append(
            {
                "image": str(relative),
                "total_count": len(lanes),
                "profile_key": profile_key,
                "profile_label": profile_label,
                "output": str(save_path),
            }
        )
        print(
            f"[ok] {relative}: lanes={len(lanes)}, "
            f"profile={profile_key}, output={save_path}"
        )

    report_path = output_dir / "lane_detection_report.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "total_count", "profile_key", "profile_label", "output"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] report: {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect lane lines in road images."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="1-车道线检测作业-照片素材-20240517",
        help="Image file or folder. Default: the provided homework image folder.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="lane_detection_output",
        help="Folder for annotated images and CSV report.",
    )
    parser.add_argument(
        "--no-auto-profile",
        action="store_true",
        help="Use the default profile instead of the automatic scene router.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_images(Path(args.input), Path(args.output), not args.no_auto_profile)


if __name__ == "__main__":
    main()
