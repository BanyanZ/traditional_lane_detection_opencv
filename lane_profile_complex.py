from __future__ import annotations

import numpy as np

from lane_line_detection import DetectionProfile, LaneCluster, detect_lanes


PROFILE = DetectionProfile(
    key="complex",
    label="多车道/弯道/遮挡",
    roi_top_ratio=0.46,
    clahe_clip=2.1,
    white_min=175,
    white_percentile=93.0,
    white_hsv_min=168,
    saturation_max=90,
    canny_low=60,
    canny_high=170,
    color_dilate=3,
    morph_close_size=3,
    component_min_area=5,
    component_min_length=6,
    use_mask_components=False,
    min_angle=10.0,
    hough_passes=((22, 18, 6), (15, 14, 5), (9, 10, 4)),
    enable_ransac=True,
    enable_kmeans=True,
    enable_curve_fit=True,
    curve_polyfit_degree=2,
    enable_cnn=False,
)


def detect(image: np.ndarray) -> tuple[np.ndarray, list[LaneCluster], np.ndarray]:
    return detect_lanes(image, PROFILE)
