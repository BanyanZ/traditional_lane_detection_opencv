from __future__ import annotations

import numpy as np

from lane_line_detection import DetectionProfile, LaneCluster, detect_lanes


PROFILE = DetectionProfile(
    key="normal",
    label="正常光照",
    roi_top_ratio=0.43,
    white_min=170,
    white_percentile=92.0,
    white_hsv_min=165,
    canny_low=50,
    canny_high=150,
    color_dilate=3,
    morph_close_size=3,
    component_min_area=5,
    component_min_length=6,
    use_mask_components=False,
    hough_passes=((18, 16, 8), (12, 14, 7), (8, 10, 5)),
    enable_ransac=True,
    enable_kmeans=True,
    enable_curve_fit=True,
    enable_cnn=False,
)


def detect(image: np.ndarray) -> tuple[np.ndarray, list[LaneCluster], np.ndarray]:
    return detect_lanes(image, PROFILE)
