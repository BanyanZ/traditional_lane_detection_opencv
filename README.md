# 车道线检测项目说明

本项目是一个车道线检测系统，包含前端页面、Flask 后端接口和车道线检测算法脚本。检测流程采用**经典图像处理算法 + 机器学习方法**的混合管线：经典 CV 负责预处理与候选提取，机器学习方法（RANSAC 鲁棒拟合、K-Means 聚类、最近质心场景分类器、多项式弯道回归）负责拟合、聚类与决策，并保留一个**可选的轻量 CNN 分割增强**（未安装 PyTorch 时自动降级为纯经典+ML 管线）。

## 项目结构

```text
.
├── backend/
│   └── app.py                 # Flask 后端接口，接收图片并返回检测结果（含耗时/弯道计数）
├── frontend/
│   ├── index.html             # 前端页面（霓虹大楼背景 + 车道小车 + 多标签座舱）
│   ├── app.js                 # 上传、调用接口、展示结果、历史/批量/快捷键
│   └── style.css              # 页面样式和动效
├── lane_line_detection.py     # 核心车道线检测算法（经典 CV + ML 拟合/聚类/弯道）
├── lane_ml.py                 # 机器学习模块：RANSAC / K-Means / 场景分类器 / 多项式 / 可选 CNN
├── lane_profile_router.py     # 用 ML 场景分类器自动选择检测参数（规则兜底）
├── lane_profile_normal.py     # 正常光照参数
├── lane_profile_bright.py     # 强光/高反差参数
├── lane_profile_low_light.py  # 低照度参数
├── lane_profile_complex.py    # 复杂道路参数
└── requirements.txt           # Python 依赖
```

## 一、经典图像处理算法部分

主要检测逻辑集中在 `lane_line_detection.py`，属于传统计算机视觉算法管线。

### 1. 图像预处理

- `cv2.cvtColor`：将图像转换为灰度图和 HSV 颜色空间。
- `cv2.createCLAHE`：增强局部对比度，提升暗光或低对比场景下的车道线可见性。
- `cv2.GaussianBlur`：对灰度图进行平滑，降低噪声对边缘检测的影响。

对应位置：`make_lane_mask()`。

### 2. 感兴趣区域 ROI

`region_of_interest()` 使用多边形掩膜只保留道路区域，减少天空、建筑、车辆等区域对检测结果的干扰。

### 3. 颜色阈值分割

在 `make_lane_mask()` 中，通过灰度亮度阈值和 HSV 阈值提取白色、黄色车道线，阈值会根据 ROI 内道路像素的亮度百分位数进行自适应调整。

### 4. 形态学处理

使用 `cv2.morphologyEx` 的开运算（去噪点）和闭运算（连接断裂）清理掩膜。

### 5. 边缘检测

使用 `cv2.Canny` 提取边缘，再和颜色掩膜结合，保留更像车道线的边缘区域。低照度场景额外启用 `use_structure_edges` 结构边缘辅助。

### 6. 霍夫直线检测

`detect_segments()` 使用多阈值 `cv2.HoughLinesP` 检测线段，并通过角度、长度、斜率、底部交点等规则过滤不合理线段，再用 `cv2.connectedComponentsWithStats` + `cv2.fitLine` 做连通域几何重建作为补充。

## 二、机器学习部分（`lane_ml.py`）

为提高检测率，本项目在经典管线之上加入了多个机器学习 / 统计学习方法。这些方法全部基于 NumPy + OpenCV 自实现，**不依赖 scikit-learn / TensorFlow**，因此干净环境下 `pip install -r requirements.txt` 即可运行。

| 方法 | 函数 | 作用 |
| --- | --- | --- |
| RANSAC 直线拟合 | `ransac_line()` | 对线段端点做鲁棒回归，剔除离群点，直线更稳 |
| RANSAC 多项式拟合 | `ransac_polynomial()` | 拟合 `x = a·y² + b·y + c`，用于弯道 |
| K-Means 车道分裂 | `kmeans_lane_seeds()` / `kmeans_post_split()` | 用 `cv2.kmeans` 把被误并的多车道重新分开 |
| 最近质心场景分类器 | `classify_scene()` | 用归一化亮度/边缘/饱和度特征判定光照场景，softmax 给置信度 |
| 滑动窗口 | `sliding_window_curves()` | 直方图峰值 + 窗口跟踪车道点云（备用） |
| 可选 CNN 分割 | `cnn_lane_mask()` | 极轻量 3 层卷积，提取车道掩膜增强；无 PyTorch 时返回 `None` 自动降级 |

### 弯道支持

`fit_cluster_curve()` 对每个车道簇同时尝试 RANSAC 直线与二次多项式，仅当**曲率足够大、多项式内点不劣于直线、且抛物线顶点不落在采样区间内（防止画出回环）**时才判为弯道，用 `cv2.polylines` 以曲线绘制（蓝色系），否则按直线绘制。

### 场景路由（ML 分类器 + 规则兜底）

`lane_profile_router.py` 的 `classify_environment()`：
1. 先用 `compute_scene_features()` 提取 10 维特征（均值/中位数/百分位/亮暗占比/边缘密度/饱和度/地平线亮度）。
2. 用 `classify_scene()` 做最近质心分类，得到场景标签与置信度。
3. 当 ML 置信度过低或与显式亮度规则冲突时，回退到规则判定（`_rule_decision`），保证鲁棒性。

## 三、可选 Transformer / CNN 扩展

当前仓库的深度学习部分是**可选的轻量 CNN**（`lane_ml.cnn_lane_mask`），默认在未安装 PyTorch 时自动关闭，不影响经典+ML 管线运行。若需要更强的端到端 Transformer 方法，可参考 LSTR：

- GitHub：<https://github.com/liuruijin17/LSTR>
- 论文：End-to-end Lane Shape Prediction with Transformers

| 分类 | 是否实际使用 | 对应文件 |
| --- | --- | --- |
| 经典图像处理算法 | 是 | `lane_line_detection.py`、`lane_profile_*.py` |
| 机器学习方法 | 是 | `lane_ml.py`（RANSAC / K-Means / 场景分类器 / 多项式回归） |
| 轻量 CNN（可选） | 可选 | `lane_ml.cnn_lane_mask`，需 `torch`，缺失自动降级 |
| 端到端 Transformer | 否 | 仅作扩展方向，参考 LSTR |

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

> 可选：安装 `torch` 后会自动启用 `lane_ml.cnn_lane_mask` 的轻量 CNN 增强；不安装也能正常运行。

启动后端：

```bash
python backend/app.py
```

前端页面位于：

```text
frontend/index.html
```

前端通过 `http://127.0.0.1:5000` 调用后端，直接用浏览器打开 `frontend/index.html` 即可（如遇浏览器本地文件限制，可在 `frontend/` 目录下运行 `python -m http.server 8080` 后访问 `http://127.0.0.1:8080`）。

打开页面后上传图片，前端会调用后端 `/api/detect` 接口，返回标注后的车道线图片、车道线总数、ML 场景策略、置信度与推理耗时。绿色为检测到的车道线，蓝色曲线为弯道。

### 命令行批量检测

也可以脱离前后端，直接批量处理素材文件夹并输出标注图 + CSV 报告：

```bash
python lane_line_detection.py "1-车道线检测作业-照片素材-20240517" -o lane_detection_output
```

## 算法整体流程

```text
输入图像
  └─ 经典预处理 (CLAHE/HSV/Canny/形态学) ── make_lane_mask
       └─ 候选线段 (多阈值 HoughLinesP + 连通域 fitLine) ── detect_segments
            └─ 聚类 (规则聚类 + K-Means 分裂) ── build_clusters
                 ├─ RANSAC 直线 / 二次多项式拟合 (弯道) ── lane_ml
                 └─ 质量过滤 + 去重 ── filter_lanes
  场景路由：ML 最近质心分类器 + 规则兜底 ── lane_profile_router
```
