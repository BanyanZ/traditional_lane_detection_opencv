from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lane_profile_router import detect_auto  # noqa: E402


ALLOWED_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

app = Flask(__name__)


@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "message": "lane detection backend is running"})


@app.post("/api/detect")
def detect():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "请先选择一张图片"}), 400

    file = request.files["file"]
    source_name = Path(file.filename or "").name
    suffix = Path(source_name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        return jsonify({"ok": False, "error": "仅支持 bmp、jpg、jpeg、png、tif 图片"}), 400

    image_bytes = np.frombuffer(file.read(), dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"ok": False, "error": "图片读取失败，请换一张图片重试"}), 400

    annotated, lanes, _, decision = detect_auto(image)

    ok, buffer = cv2.imencode(".png", annotated)
    if not ok:
        return jsonify({"ok": False, "error": "检测结果编码失败"}), 500

    solid_count = sum(1 for lane in lanes if lane.lane_type == "solid")
    dashed_count = sum(1 for lane in lanes if lane.lane_type == "dashed")
    image_base64 = base64.b64encode(buffer).decode("ascii")

    return jsonify(
        {
            "ok": True,
            "fileName": source_name,
            "solidCount": solid_count,
            "dashedCount": dashed_count,
            "totalCount": len(lanes),
            "profileKey": decision.key,
            "profileLabel": decision.label,
            "profileScript": decision.script,
            "profileConfidence": round(decision.confidence, 2),
            "profileReason": decision.reason,
            "resultImage": f"data:image/png;base64,{image_base64}",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
