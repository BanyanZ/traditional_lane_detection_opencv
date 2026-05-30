from __future__ import annotations

import numpy as np

from lane_line_detection import DetectionProfile, LaneCluster, detect_lanes


PROFILE = DetectionProfile(
    key="bright",
    label="强光/高反差",
    roi_top_ratio=0.45,
    clahe_clip=1.6,
    white_min=188,
    white_percentile=96.0,
    white_hsv_min=180,
    saturation_max=80,
    canny_low=70,
    canny_high=190,
    color_dilate=3,
    morph_close_size=3,
    component_min_area=5,
    component_min_length=6,
    use_mask_components=False,
    min_angle=12.0,
    hough_passes=((24, 20, 6), (16, 16, 5), (10, 12, 4)),
    enable_ransac=True,
    enable_kmeans=True,
    enable_curve_fit=True,
    enable_cnn=False,
)


def detect(image: np.ndarray) -> tuple[np.ndarray, list[LaneCluster], np.ndarray]:
    return detect_lanes(image, PROFILE)
