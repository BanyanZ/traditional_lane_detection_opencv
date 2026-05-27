from __future__ import annotations

import numpy as np

from lane_line_detection import DetectionProfile, LaneCluster, detect_lanes


PROFILE = DetectionProfile(
    key="low_light",
    label="低照度/特殊工况",
    roi_top_ratio=0.53,
    clahe_clip=2.4,
    white_min=150,
    white_percentile=88.0,
    white_hsv_min=145,
    saturation_max=105,
    yellow_s_min=35,
    yellow_v_min=70,
    canny_low=45,
    canny_high=135,
    color_dilate=2,
    morph_open_size=2,
    morph_close_size=3,
    component_min_area=4,
    component_min_length=5,
    component_max_area_ratio=0.008,
    max_lanes=3,
    min_angle=11.0,
    left_bottom_limit=-0.80,
    right_bottom_limit=1.65,
    hough_passes=((18, 14, 5), (12, 10, 4), (7, 6, 3)),
    use_structure_edges=True,
    structure_hough_passes=((24, 22, 8), (18, 16, 6)),
    structure_min_angle=10.0,
    structure_max_angle=75.0,
    structure_dilate=2,
)


def detect(image: np.ndarray) -> tuple[np.ndarray, list[LaneCluster], np.ndarray]:
    return detect_lanes(image, PROFILE)
