from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import lane_profile_bright
import lane_profile_complex
import lane_profile_low_light
import lane_profile_normal
from lane_line_detection import LaneCluster, region_of_interest


@dataclass(frozen=True)
class ProfileDecision:
    key: str
    label: str
    script: str
    confidence: float
    reason: str


DETECTORS = {
    lane_profile_normal.PROFILE.key: (lane_profile_normal.PROFILE, lane_profile_normal.detect, "lane_profile_normal.py"),
    lane_profile_bright.PROFILE.key: (lane_profile_bright.PROFILE, lane_profile_bright.detect, "lane_profile_bright.py"),
    lane_profile_low_light.PROFILE.key: (
        lane_profile_low_light.PROFILE,
        lane_profile_low_light.detect,
        "lane_profile_low_light.py",
    ),
    lane_profile_complex.PROFILE.key: (lane_profile_complex.PROFILE, lane_profile_complex.detect, "lane_profile_complex.py"),
}


def classify_environment(image: np.ndarray) -> ProfileDecision:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    roi = region_of_interest((height, width), top_ratio=0.42)
    road_pixels = gray[roi > 0]

    if road_pixels.size == 0:
        profile, _, script = DETECTORS["normal"]
        return ProfileDecision(profile.key, profile.label, script, 0.5, "未取得道路区域，使用默认亮度参数")

    mean = float(np.mean(road_pixels))
    median = float(np.median(road_pixels))
    p90 = float(np.percentile(road_pixels, 90))
    p95 = float(np.percentile(road_pixels, 95))
    bright_ratio = float(np.mean(road_pixels >= 230))
    dark_ratio = float(np.mean(road_pixels <= 70))

    reason = (
        f"道路亮度: mean={mean:.1f}, median={median:.1f}, "
        f"p90={p90:.1f}, p95={p95:.1f}"
    )

    if mean < 105 or median < 98 or p90 < 150 or dark_ratio > 0.22:
        profile, _, script = DETECTORS["low_light"]
        confidence = min(
            0.95,
            0.55 + max(0.0, 110 - mean) / 120 + dark_ratio * 0.8 + max(0.0, 155 - p90) / 180,
        )
        return ProfileDecision(profile.key, profile.label, script, confidence, f"{reason}，判为偏暗")

    if mean > 185 or median > 180 or bright_ratio > 0.14 or (mean > 170 and p95 > 245):
        profile, _, script = DETECTORS["bright"]
        confidence = min(
            0.96,
            0.55 + max(0.0, mean - 175) / 120 + bright_ratio * 1.4 + max(0.0, p95 - 225) / 120,
        )
        return ProfileDecision(profile.key, profile.label, script, confidence, f"{reason}，判为过亮/强反光")

    profile, _, script = DETECTORS["normal"]
    confidence = min(0.9, 0.62 + (1.0 - abs(mean - 145) / 145) * 0.22)
    return ProfileDecision(profile.key, profile.label, script, confidence, f"{reason}，判为正常亮度")


def detect_auto(
    image: np.ndarray,
) -> tuple[np.ndarray, list[LaneCluster], np.ndarray, ProfileDecision]:
    decision = classify_environment(image)
    _, detector, _ = DETECTORS[decision.key]
    annotated, lanes, edge_mask = detector(image)
    return annotated, lanes, edge_mask, decision
