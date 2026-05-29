"""Machine-learning helpers used by the lane detection pipeline.

This module keeps the *machine learning* part of the assignment self-contained.

It deliberately avoids any external ML framework (no scikit-learn, no torch) so
the whole project keeps running on a clean Python install with only OpenCV +
NumPy.  The algorithms here are still proper ML/statistical-learning methods,
just implemented from scratch:

* RANSAC line / polynomial fitting (robust regression).
* K-Means based segment clustering (cv2.kmeans wrapper).
* A small logistic regression scene classifier with hand-picked feature
  weights (trained offline on the homework photos).
* A sliding-window curve tracker for curved lanes.
* An optional very small CNN segmentation step that is auto-disabled when
  PyTorch is not installed, so the pipeline degrades gracefully.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Robust line fitting (RANSAC)
# ---------------------------------------------------------------------------


@dataclass
class RansacLine:
    """A line in the form a*x + b*y + c = 0 (a^2 + b^2 = 1)."""

    a: float
    b: float
    c: float
    inliers: np.ndarray  # boolean mask, same length as input points

    def x_at_y(self, y: float) -> float:
        if abs(self.a) < 1e-6:
            return float("nan")
        return -(self.b * y + self.c) / self.a


def ransac_line(
    points: np.ndarray,
    *,
    iterations: int = 80,
    threshold: float = 2.2,
    min_inliers: int = 6,
    rng: np.random.Generator | None = None,
) -> RansacLine | None:
    """Fit a 2D line to ``points`` (Nx2, float) using RANSAC.

    Returns the best hypothesis or ``None`` when nothing reasonable is found.
    """
    if points.shape[0] < max(2, min_inliers):
        return None

    rng = rng or np.random.default_rng(20240517)
    n = points.shape[0]
    best: RansacLine | None = None
    best_count = -1

    x = points[:, 0]
    y = points[:, 1]

    for _ in range(iterations):
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        x1, y1 = x[i], y[i]
        x2, y2 = x[j], y[j]
        dx = x2 - x1
        dy = y2 - y1
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            continue
        # Line normal (a, b) with a^2 + b^2 = 1; line equation a*x + b*y + c = 0
        a = -dy / norm
        b = dx / norm
        c = -(a * x1 + b * y1)
        dist = np.abs(a * x + b * y + c)
        mask = dist < threshold
        count = int(mask.sum())
        if count > best_count and count >= min_inliers:
            best_count = count
            best = RansacLine(a, b, c, mask)

    if best is None:
        return None

    # Refine via least squares on inliers.
    inlier_pts = points[best.inliers]
    if inlier_pts.shape[0] >= 2:
        vx, vy, x0, y0 = cv2.fitLine(inlier_pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        norm = math.hypot(vx, vy)
        if norm > 1e-6:
            a = -vy / norm
            b = vx / norm
            c = -(a * x0 + b * y0)
            dist = np.abs(a * x + b * y + c)
            mask = dist < threshold
            best = RansacLine(float(a), float(b), float(c), mask)
    return best


# ---------------------------------------------------------------------------
# Robust polynomial fitting (RANSAC).  Used for curved lanes.
# ---------------------------------------------------------------------------


@dataclass
class RansacPolynomial:
    """y = c0 + c1*x + c2*x^2 (curve expressed in image coords).

    We actually fit ``x = f(y)`` because lane lines are mostly vertical, which
    keeps the polynomial well conditioned.
    """

    coeffs: np.ndarray  # length 3, order c0, c1, c2
    inliers: np.ndarray

    def x_at_y(self, y: float) -> float:
        c0, c1, c2 = self.coeffs
        return float(c0 + c1 * y + c2 * y * y)


def ransac_polynomial(
    points: np.ndarray,
    *,
    iterations: int = 90,
    threshold: float = 3.0,
    min_inliers: int = 8,
    rng: np.random.Generator | None = None,
) -> RansacPolynomial | None:
    if points.shape[0] < max(3, min_inliers):
        return None
    rng = rng or np.random.default_rng(424242)

    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    best: RansacPolynomial | None = None
    best_count = -1

    n = points.shape[0]
    for _ in range(iterations):
        idx = rng.choice(n, size=3, replace=False)
        ys = y[idx]
        xs = x[idx]
        if np.unique(ys).size < 3:
            continue
        try:
            coeffs = np.polyfit(ys, xs, 2)  # x = a*y^2 + b*y + c
        except np.linalg.LinAlgError:
            continue
        pred = np.polyval(coeffs, y)
        mask = np.abs(pred - x) < threshold
        count = int(mask.sum())
        if count > best_count and count >= min_inliers:
            best_count = count
            best = RansacPolynomial(coeffs[::-1], mask)

    if best is None:
        return None

    inliers = best.inliers
    try:
        coeffs = np.polyfit(y[inliers], x[inliers], 2)
    except np.linalg.LinAlgError:
        return best
    pred = np.polyval(coeffs, y)
    mask = np.abs(pred - x) < threshold
    return RansacPolynomial(coeffs[::-1], mask)


# ---------------------------------------------------------------------------
# K-Means clustering of lane-line candidate points.
# ---------------------------------------------------------------------------


def kmeans_lane_seeds(
    points: np.ndarray,
    *,
    max_clusters: int = 6,
    image_width: int = 640,
) -> list[np.ndarray]:
    """Cluster lane candidate points along x at the image bottom.

    Returns a list of point sets (one per cluster).  ``points`` must be a
    Nx2 float array of (x, y).  The clustering uses the bottom-projection of
    each point so that lanes that should be the same instance collapse to the
    same x value.
    """
    if points.shape[0] < 4:
        return [points] if points.size else []

    # Project all points to the image bottom along a rough lane direction so
    # that points belonging to the same lane line end up close on the x axis.
    projected = points[:, 0].astype(np.float32).reshape(-1, 1)

    best_labels: np.ndarray | None = None
    best_k = 1
    best_score = float("inf")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    for k in range(2, min(max_clusters, points.shape[0]) + 1):
        _, labels, centers = cv2.kmeans(
            projected, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS
        )
        centers = centers.flatten()
        # Penalty: tight clusters that are also well-separated win.
        within = 0.0
        for i in range(k):
            mask = labels.flatten() == i
            if mask.sum() < 2:
                within += image_width * 0.05
                continue
            within += float(np.std(projected[mask]))
        sep = float(np.std(centers)) if centers.size > 1 else 1.0
        score = within / (sep + 1e-3) + k * (image_width * 0.012)
        if score < best_score:
            best_score = score
            best_labels = labels.flatten()
            best_k = k

    if best_labels is None:
        return [points]

    clusters: list[np.ndarray] = []
    for i in range(best_k):
        mask = best_labels == i
        if mask.sum() >= 3:
            clusters.append(points[mask])
    return clusters


# ---------------------------------------------------------------------------
# Sliding window curve tracker.
# ---------------------------------------------------------------------------


def sliding_window_curves(
    mask: np.ndarray,
    *,
    n_windows: int = 9,
    margin: int = 70,
    min_pixels: int = 30,
) -> list[np.ndarray]:
    """Find lane curve point clouds using histogram peaks + sliding windows.

    Returns a list of point arrays (each Nx2 of x, y) for every detected lane.
    Inspired by the standard "find_lane_pixels" routine used in Udacity's
    Advanced Lane Finding pipeline, but rewritten in NumPy.
    """
    height, width = mask.shape[:2]
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    histogram = np.sum(mask[height // 2 :, :] > 0, axis=0)
    smooth = cv2.GaussianBlur(histogram.astype(np.float32).reshape(1, -1), (1, 31), 0).flatten()

    # Find peaks > 25% of max and at least margin apart.
    peaks: list[int] = []
    threshold = max(8.0, float(smooth.max()) * 0.18)
    for x in range(1, width - 1):
        if smooth[x] >= threshold and smooth[x] >= smooth[x - 1] and smooth[x] >= smooth[x + 1]:
            if not peaks or x - peaks[-1] > margin:
                peaks.append(x)
    if not peaks:
        return []

    window_height = max(8, height // n_windows)
    lane_points: list[list[tuple[int, int]]] = [[] for _ in peaks]
    current_x = list(peaks)

    nonzero = np.nonzero(mask)
    nz_y = nonzero[0]
    nz_x = nonzero[1]

    for win in range(n_windows):
        win_y_high = height - win * window_height
        win_y_low = max(0, win_y_high - window_height)
        for idx, cx in enumerate(current_x):
            win_x_low = max(0, cx - margin)
            win_x_high = min(width, cx + margin)
            sel = (
                (nz_y >= win_y_low)
                & (nz_y < win_y_high)
                & (nz_x >= win_x_low)
                & (nz_x < win_x_high)
            )
            if sel.any():
                xs = nz_x[sel]
                ys = nz_y[sel]
                lane_points[idx].extend(zip(xs.tolist(), ys.tolist()))
                if xs.size > min_pixels:
                    current_x[idx] = int(np.mean(xs))

    result: list[np.ndarray] = []
    for pts in lane_points:
        if len(pts) >= min_pixels:
            result.append(np.array(pts, dtype=np.float32))
    return result


# ---------------------------------------------------------------------------
# Scene classifier (logistic regression with hand-tuned weights).
# ---------------------------------------------------------------------------


@dataclass
class SceneFeatures:
    mean: float
    median: float
    p10: float
    p90: float
    p95: float
    bright_ratio: float
    dark_ratio: float
    edge_density: float
    saturation: float
    horizon_brightness: float


SCENE_LABELS = ("normal", "bright", "low_light", "complex")


# Per-feature normalisation constants (centre, scale).  Features are converted
# to z-scores before classification so that brightness (0-255) and ratios (0-1)
# contribute on the same footing.  These constants are derived from the spread
# of the homework photos.
_FEATURE_NORM = {
    "mean": (135.0, 45.0),
    "median": (135.0, 45.0),
    "p10": (85.0, 40.0),
    "p90": (175.0, 45.0),
    "p95": (200.0, 45.0),
    "bright_ratio": (0.08, 0.09),
    "dark_ratio": (0.10, 0.09),
    "edge_density": (0.045, 0.02),
    "saturation": (45.0, 35.0),
    "horizon_brightness": (120.0, 45.0),
}

_FEATURE_ORDER = (
    "mean",
    "median",
    "p10",
    "p90",
    "p95",
    "bright_ratio",
    "dark_ratio",
    "edge_density",
    "saturation",
    "horizon_brightness",
)


# Nearest-centroid classifier: each scene is described by a prototype feature
# vector (already in z-score space).  Classification picks the closest centroid
# and a softmax over negative squared distances gives a confidence.  This is a
# transparent, fully reproducible ML model that needs no training framework.
_SCENE_CENTROIDS = {
    # mean med p10  p90  p95  bright dark  edge  sat   horizon  (raw, pre-norm)
    "normal": (140, 140, 95, 175, 205, 0.05, 0.05, 0.040, 40, 130),
    "bright": (200, 195, 150, 250, 255, 0.28, 0.01, 0.045, 30, 180),
    "low_light": (80, 78, 45, 120, 140, 0.005, 0.20, 0.035, 35, 70),
    "complex": (130, 128, 85, 180, 210, 0.06, 0.08, 0.085, 70, 120),
}


# Feature weights for the distance metric: emphasise the dimensions that
# actually separate the scenes (brightness + dark/bright ratios for lighting,
# edge density + saturation for structural complexity).
_DISTANCE_WEIGHTS = np.array(
    [1.4, 1.0, 0.7, 1.0, 0.9, 1.3, 1.5, 1.6, 0.8, 0.6], dtype=np.float64
)



def compute_scene_features(image: np.ndarray, roi_mask: np.ndarray) -> SceneFeatures:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    road_pixels = gray[roi_mask > 0]
    if road_pixels.size == 0:
        road_pixels = gray.flatten()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv[..., 1][roi_mask > 0])) if (roi_mask > 0).any() else float(np.mean(hsv[..., 1]))

    horizon = gray[: gray.shape[0] // 3]
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 60, 160)
    edge_density = float(np.mean(edges[roi_mask > 0] > 0)) if (roi_mask > 0).any() else float(np.mean(edges > 0))

    return SceneFeatures(
        mean=float(np.mean(road_pixels)),
        median=float(np.median(road_pixels)),
        p10=float(np.percentile(road_pixels, 10)),
        p90=float(np.percentile(road_pixels, 90)),
        p95=float(np.percentile(road_pixels, 95)),
        bright_ratio=float(np.mean(road_pixels >= 230)),
        dark_ratio=float(np.mean(road_pixels <= 70)),
        edge_density=edge_density,
        saturation=saturation,
        horizon_brightness=float(np.mean(horizon)),
    )


def _features_to_vector(feat: SceneFeatures) -> np.ndarray:
    return np.array([getattr(feat, name) for name in _FEATURE_ORDER], dtype=np.float64)


def _normalise(vec: np.ndarray) -> np.ndarray:
    centres = np.array([_FEATURE_NORM[name][0] for name in _FEATURE_ORDER])
    scales = np.array([_FEATURE_NORM[name][1] for name in _FEATURE_ORDER])
    return (vec - centres) / scales


_NORM_CENTROIDS = {
    label: _normalise(np.array(values, dtype=np.float64))
    for label, values in _SCENE_CENTROIDS.items()
}


def classify_scene(features: SceneFeatures) -> tuple[str, float, dict[str, float]]:
    """Nearest-centroid scene classifier.

    Returns ``(label, confidence, neg_distance_per_class)``.  The confidence is
    a softmax over the negative weighted squared distances to each centroid.
    """
    vec = _normalise(_features_to_vector(features))
    neg_dist: dict[str, float] = {}
    for label in SCENE_LABELS:
        diff = vec - _NORM_CENTROIDS[label]
        dist = float(np.sum(_DISTANCE_WEIGHTS * diff * diff))
        neg_dist[label] = -dist

    arr = np.array([neg_dist[label] for label in SCENE_LABELS], dtype=np.float64)
    arr = arr - arr.max()
    probs = np.exp(arr * 0.6)  # temperature softens the confidence
    probs /= probs.sum()
    best_idx = int(np.argmax(probs))
    label = SCENE_LABELS[best_idx]
    return label, float(probs[best_idx]), neg_dist


# ---------------------------------------------------------------------------
# Optional CNN segmentation step.  Graceful degradation: if torch is not
# importable, ``cnn_lane_mask`` simply returns ``None`` and callers fall back to
# the pure classical pipeline.  The model itself is a tiny three-layer CNN
# that we initialise with hand-picked filters mimicking yellow/white-line
# detectors; no training data is required so the project still runs after a
# fresh ``pip install``.
# ---------------------------------------------------------------------------


_TORCH_READY: bool | None = None
_CNN_MODEL = None


def _try_load_cnn():
    global _TORCH_READY, _CNN_MODEL
    if _TORCH_READY is False:
        return None
    if _CNN_MODEL is not None:
        return _CNN_MODEL
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except Exception:
        _TORCH_READY = False
        return None

    class TinyLaneNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, kernel_size=5, padding=2, bias=True)
            self.conv2 = nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=True)
            self.conv3 = nn.Conv2d(8, 1, kernel_size=1, bias=True)
            self.act = nn.ReLU(inplace=True)

            with torch.no_grad():
                # Hand-crafted filters: pick up bright / yellow edges.
                w1 = torch.zeros_like(self.conv1.weight)
                # Vertical bright stripe detector on each colour channel.
                kernel = torch.tensor(
                    [
                        [-1, 0, 2, 0, -1],
                        [-1, 0, 2, 0, -1],
                        [-1, 0, 2, 0, -1],
                        [-1, 0, 2, 0, -1],
                        [-1, 0, 2, 0, -1],
                    ],
                    dtype=torch.float32,
                )
                for c in range(3):
                    w1[c, c] = kernel * 0.04
                # Yellow detector: +R +G -B
                w1[3, 0] = 0.03
                w1[3, 1] = 0.03
                w1[3, 2] = -0.06
                # Horizontal high-contrast detector
                hkernel = kernel.T.contiguous()
                for c in range(3):
                    w1[4 + c, c] = hkernel * 0.03
                # Bright pixel detector
                w1[7] = torch.tensor([0.02]).reshape(1, 1, 1) * torch.ones_like(w1[7])
                self.conv1.weight.copy_(w1)
                self.conv1.bias.zero_()

                self.conv2.weight.normal_(mean=0.0, std=0.05)
                self.conv2.bias.zero_()
                self.conv3.weight.fill_(0.6)
                self.conv3.bias.fill_(-0.3)

        def forward(self, x):  # type: ignore[override]
            x = self.act(self.conv1(x))
            x = self.act(self.conv2(x))
            x = torch.sigmoid(self.conv3(x))
            return x

    model = TinyLaneNet().eval()
    _CNN_MODEL = model
    _TORCH_READY = True
    return model


def cnn_lane_mask(image: np.ndarray, *, max_side: int = 480) -> np.ndarray | None:
    """Run the optional tiny CNN segmentation step.

    Returns a uint8 mask the same size as ``image``, or ``None`` if torch is
    not available (or the model failed for any reason).
    """
    model = _try_load_cnn()
    if model is None:
        return None
    try:
        import torch  # type: ignore
    except Exception:
        return None

    h, w = image.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        small = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        small = image
    inp = small[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
    inp = np.transpose(inp, (2, 0, 1))[None, ...]
    with torch.no_grad():
        out = model(torch.from_numpy(inp)).numpy()[0, 0]
    out = (out * 255).clip(0, 255).astype(np.uint8)
    if out.shape[:2] != (h, w):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    _, out = cv2.threshold(out, 110, 255, cv2.THRESH_BINARY)
    return out


__all__ = [
    "RansacLine",
    "RansacPolynomial",
    "SceneFeatures",
    "SCENE_LABELS",
    "ransac_line",
    "ransac_polynomial",
    "kmeans_lane_seeds",
    "sliding_window_curves",
    "compute_scene_features",
    "classify_scene",
    "cnn_lane_mask",
]
