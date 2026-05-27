# 车道线检测项目说明

本项目是一个基于 OpenCV 的车道线检测系统，包含前端页面、Flask 后端接口和车道线检测算法脚本。当前仓库中的实际检测流程主要使用传统图像处理算法完成，没有集成可训练或可推理的 Transformer 深度学习模型。

## 项目结构

```text
.
├── backend/
│   └── app.py                 # Flask 后端接口，接收图片并返回检测结果
├── frontend/
│   ├── index.html             # 前端页面
│   ├── app.js                 # 上传、调用接口、展示结果
│   └── style.css              # 页面样式和动效
├── lane_line_detection.py     # 核心车道线检测算法
├── lane_profile_router.py     # 根据图像亮度自动选择检测参数
├── lane_profile_normal.py     # 正常光照参数
├── lane_profile_bright.py     # 强光/高反差参数
├── lane_profile_low_light.py  # 低照度参数
├── lane_profile_complex.py    # 复杂道路参数
└── requirements.txt           # Python 依赖
```

## 传统算法部分

当前项目的主要检测逻辑集中在 `lane_line_detection.py`，属于传统计算机视觉算法管线。

### 1. 图像预处理

使用 OpenCV 对输入图片进行基础处理：

- `cv2.cvtColor`：将图像转换为灰度图和 HSV 颜色空间。
- `cv2.createCLAHE`：增强局部对比度，提升暗光或低对比场景下的车道线可见性。
- `cv2.GaussianBlur`：对灰度图进行平滑，降低噪声对边缘检测的影响。

对应位置：`make_lane_mask()`。

### 2. 感兴趣区域 ROI

`region_of_interest()` 使用多边形掩膜只保留道路区域，减少天空、建筑、车辆等区域对检测结果的干扰。

### 3. 颜色阈值分割

在 `make_lane_mask()` 中，通过灰度亮度阈值和 HSV 阈值提取白色、黄色车道线：

- 白色车道线：结合灰度增强图和 HSV 的亮度、饱和度范围。
- 黄色车道线：使用 HSV 中的色相、饱和度和亮度范围。
- 阈值会根据 ROI 内道路像素的亮度百分位数进行自适应调整。

### 4. 形态学处理

使用 `cv2.morphologyEx` 的开运算和闭运算清理掩膜：

- 开运算：去除孤立噪声点。
- 闭运算：连接断裂的车道线区域。

### 5. 边缘检测

使用 `cv2.Canny` 提取边缘，再和颜色掩膜结合，保留更像车道线的边缘区域。

### 6. 霍夫直线检测

`detect_segments()` 使用 `cv2.HoughLinesP` 检测线段，并通过角度、长度、斜率、底部交点等规则过滤不合理线段。

低照度场景还会启用 `use_structure_edges`，在 `make_lane_mask()` 中额外使用结构边缘辅助检测。

### 7. 连通域与拟合

项目还使用了传统几何方法补充霍夫线检测：

- `cv2.connectedComponentsWithStats`：提取连通区域。
- `cv2.fitLine`：对连通区域或聚类后的线段进行直线拟合。
- `cv2.clipLine`：把拟合直线裁剪到图像范围内。

### 8. 聚类和实线/虚线分类

`cluster_segments()` 根据线段的斜率、位置和底部交点将线段聚成车道线。

`classify_lane()` 根据以下特征判断实线或虚线：

- 线段在 y 方向上的覆盖范围。
- 线段之间的间隔比例。
- 线段平均长度。
- 总长度和位置。

最终 `draw_lanes()` 将检测到的车道线绘制到原图上，绿色表示实线，红色表示虚线。

## 多场景参数配置

项目没有为所有场景使用一套固定参数，而是拆分了多个 profile：

- `lane_profile_normal.py`：正常光照。
- `lane_profile_bright.py`：强光、高反差、曝光较强场景。
- `lane_profile_low_light.py`：低照度、暗光场景。
- `lane_profile_complex.py`：复杂道路、多车道、弯道或遮挡场景。

`lane_profile_router.py` 会先分析 ROI 内道路亮度，包括均值、中位数、90/95 分位数、过亮比例和过暗比例，然后自动选择对应的 profile。

这部分仍然属于传统规则算法，不是深度学习分类器。

## Transformer 部分

当前仓库没有真正使用 Transformer 模型。

具体表现为：

- `requirements.txt` 中没有 `torch`、`tensorflow`、`onnxruntime` 等深度学习推理依赖。
- 代码中没有模型权重加载逻辑。
- 没有 Transformer encoder/decoder、attention、query 或神经网络前向推理代码。
- `frontend/style.css` 中出现的 `transform` 是 CSS 视觉变换属性，只用于页面动画和布局效果，不是机器学习中的 Transformer。

因此，如果按“传统算法 / Transformer”分类，本项目可以写成：

| 分类 | 是否实际使用 | 对应文件 | 说明 |
| --- | --- | --- | --- |
| 传统图像处理算法 | 是 | `lane_line_detection.py`、`lane_profile_*.py`、`lane_profile_router.py` | 完成车道线检测、聚类、拟合和实线/虚线分类 |
| Transformer 深度学习模型 | 否 | 无 | 当前仅作为可扩展方向或论文/项目参考 |
| CSS transform | 是 | `frontend/style.css` | 只负责前端动画、缩放、旋转、位移效果 |

## Transformer 参考项目

如果需要在论文、报告或后续扩展中引用 Transformer 车道线检测方法，可以参考 LSTR：

- GitHub 项目：<https://github.com/liuruijin17/LSTR>
- 论文名称：End-to-end Lane Shape Prediction with Transformers
- 方法特点：LSTR 使用 Transformer 结构进行端到端车道线形状预测，直接预测车道线形状参数，而不是像本项目这样先做颜色阈值、边缘检测、霍夫变换和几何规则后处理。

可以在报告中这样表述：

> 本项目当前实现采用 OpenCV 传统图像处理流程完成车道线检测，包括 ROI 掩膜、颜色阈值分割、Canny 边缘检测、霍夫直线检测、连通域分析、线段聚类和规则分类。Transformer 方法未在当前代码中实际部署，仅作为对比和后续扩展方向，可参考 LSTR: End-to-end Lane Shape Prediction with Transformers。

## 运行方式

安装依赖：

```bash
pip install -r requirements.txt
```

启动后端：

```bash
python backend/app.py
```

前端页面位于：

```text
frontend/index.html
```

打开页面后上传图片，前端会调用后端 `/api/detect` 接口，返回标注后的车道线图片以及实线、虚线数量。

