const API_BASE = "http://127.0.0.1:5000";
const THEMES = ["cyber", "green", "blue", "amber", "violet"];

const $ = (sel) => document.querySelector(sel);

const fileInput = $("#fileInput");
const openButton = $("#openButton");
const detectButton = $("#detectButton");
const saveButton = $("#saveButton");
const sampleButton = $("#sampleButton");
const themeButton = $("#themeButton");
const actionToggle = $("#actionToggle");
const commandDeck = $("#commandDeck");
const detectorWindow = $("#detectorWindow");
const fileName = $("#fileName");
const sourcePreview = $("#sourcePreview");
const resultPreview = $("#resultPreview");
const sourceBadge = $("#sourceBadge");
const resultBadge = $("#resultBadge");
const serverStatus = $("#serverStatus");
const scanCount = $("#scanCount");
const message = $("#message");
const toast = $("#toast");
const laneCount = $("#laneCount");
const profileLabel = $("#profileLabel");
const profileScript = $("#profileScript");
const confidenceLabel = $("#confidenceLabel");
const confidenceBar = $("#confidenceBar");
const timeLabel = $("#timeLabel");
const reasonText = $("#reasonText");
const imageModal = $("#imageModal");
const modalImage = $("#modalImage");
const modalClose = $("#modalClose");
const tabs = $("#tabs");
const historyList = $("#historyList");
const batchGrid = $("#batchGrid");

let selectedFile = null;
let previewUrl = null;
let resultImageData = null;
let resultFileName = "lane-detection-result.png";
let scanTotal = 0;
const history = [];

let themeIndex = Math.max(0, THEMES.indexOf(localStorage.getItem("lane-theme") || "cyber"));
document.body.dataset.theme = THEMES[themeIndex];

const SAMPLE_GROUPS = [
  { name: "正常光照", desc: "标准白天直道", count: 6 },
  { name: "光照强烈", desc: "强反光 / 过曝", count: 6 },
  { name: "弯道图片", desc: "多项式弯道拟合", count: 3 },
  { name: "多车道", desc: "K-Means 车道分裂", count: 7 },
  { name: "特殊工况", desc: "低照度 / 遮挡", count: 6 },
];

/* ---------------- helpers ---------------- */

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
  message.hidden = !text;
}

let toastTimer = null;
function showToast(text, isError = false) {
  toast.textContent = text;
  toast.classList.toggle("error", isError);
  toast.hidden = false;
  requestAnimationFrame(() => toast.classList.add("show"));
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toast.classList.remove("show");
    window.setTimeout(() => (toast.hidden = true), 260);
  }, 2600);
}

function setImage(container, src, alt) {
  container.innerHTML = "";
  const image = document.createElement("img");
  image.src = src;
  image.alt = alt;
  image.decoding = "async";
  container.appendChild(image);
}

function setPlaceholder(container, text, scan = false) {
  container.innerHTML = "";
  const wrap = document.createElement("span");
  wrap.className = "slot-placeholder";
  const ring = document.createElement("span");
  ring.className = scan ? "slot-ring scanring" : "slot-ring";
  wrap.appendChild(ring);
  wrap.appendChild(document.createTextNode(text));
  container.appendChild(wrap);
}

function resultNameFrom(file) {
  const name = file && file.name ? file.name : "image";
  const dotIndex = name.lastIndexOf(".");
  const baseName = dotIndex > 0 ? name.slice(0, dotIndex) : name;
  return `${baseName}_lane_result.png`;
}

function resetResult() {
  resultImageData = null;
  resultFileName = resultNameFrom(selectedFile);
  saveButton.disabled = true;
  laneCount.textContent = "0";
  profileLabel.textContent = "-";
  profileScript.textContent = "—";
  confidenceLabel.textContent = "-";
  confidenceBar.style.width = "0%";
  timeLabel.textContent = "-";
  resultBadge.textContent = "等待";
}

function setDeckOpen(isOpen) {
  commandDeck.classList.toggle("open", isOpen);
  actionToggle.setAttribute("aria-expanded", String(isOpen));
}

function popButton(button) {
  button.classList.add("pop-click");
  window.setTimeout(() => button.classList.remove("pop-click"), 170);
}

function addRipple(button, event) {
  const rect = button.getBoundingClientRect();
  const ripple = document.createElement("span");
  ripple.className = "ripple";
  ripple.style.left = `${(event?.clientX ?? rect.left + rect.width / 2) - rect.left}px`;
  ripple.style.top = `${(event?.clientY ?? rect.top + rect.height / 2) - rect.top}px`;
  button.appendChild(ripple);
  ripple.addEventListener("animationend", () => ripple.remove(), { once: true });
}

function animateButton(button, event) {
  if (button.disabled) return;
  popButton(button);
  addRipple(button, event);
}

function cycleTheme() {
  themeIndex = (themeIndex + 1) % THEMES.length;
  const nextTheme = THEMES[themeIndex];
  document.body.dataset.theme = nextTheme;
  localStorage.setItem("lane-theme", nextTheme);
  document.body.classList.remove("theme-flash");
  void document.body.offsetWidth;
  document.body.classList.add("theme-flash");
  showToast(`配色：${nextTheme}`);
}

function loadFile(file) {
  if (!file) return;
  if (!file.type.startsWith("image/") && !/\.(bmp|tif|tiff)$/i.test(file.name)) {
    showToast("请选择图片文件", true);
    return;
  }

  selectedFile = file;
  detectButton.disabled = false;
  fileName.textContent = file.name;
  sourceBadge.textContent = "已载入";
  setPlaceholder(resultPreview, "等待检测", true);
  resetResult();
  setMessage("图片已载入，可以开始检测。");

  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  setImage(sourcePreview, previewUrl, "原始图像预览");
  setDeckOpen(true);
}

function confidenceText(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return `${Math.round(value * 100)}%`;
}

function pulseMetric() {
  const card = laneCount.closest(".lane-count-card");
  card.classList.remove("metric-pop");
  void card.offsetWidth;
  card.classList.add("metric-pop");
}

function openImageModal(src) {
  if (!src) return;
  modalImage.src = src;
  imageModal.hidden = false;
}

function closeImageModal() {
  imageModal.hidden = true;
  modalImage.removeAttribute("src");
}

/* ---------------- history ---------------- */

function renderHistory() {
  $("#histTotal").textContent = String(history.length);
  if (history.length) {
    const avg = (key) => history.reduce((s, h) => s + (h[key] || 0), 0) / history.length;
    $("#histLaneAvg").textContent = avg("total").toFixed(1);
    $("#histTime").textContent = `${Math.round(avg("ms"))} ms`;
  } else {
    $("#histLaneAvg").textContent = "0.0";
    $("#histTime").textContent = "0 ms";
  }

  historyList.innerHTML = "";
  if (!history.length) {
    const li = document.createElement("li");
    li.className = "empty-row";
    li.textContent = "暂无检测记录";
    historyList.appendChild(li);
    return;
  }
  [...history].reverse().forEach((h) => {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "item-name";
    name.textContent = h.name;
    const meta = document.createElement("span");
    meta.className = "item-meta";
    meta.textContent = `${h.total} 条 · ${h.label} · ${h.ms} ms`;
    li.append(name, meta);
    historyList.appendChild(li);
  });
}

/* ---------------- batch panel ---------------- */

function renderBatch() {
  batchGrid.innerHTML = "";
  SAMPLE_GROUPS.forEach((group) => {
    const card = document.createElement("div");
    card.className = "batch-card";
    card.innerHTML = `<h3>${group.name}</h3><p>${group.desc}</p><div class="count">${group.count} 张</div>`;
    card.addEventListener("click", () => {
      showToast(`「${group.name}」素材位于 1-车道线检测作业-照片素材-20240517/${group.name}`);
    });
    batchGrid.appendChild(card);
  });
}

/* ---------------- backend ---------------- */

async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    if (!response.ok) throw new Error("backend unavailable");
    serverStatus.textContent = "已连接";
    serverStatus.className = "status online";
    return true;
  } catch {
    serverStatus.textContent = "未连接";
    serverStatus.className = "status offline";
    setMessage("后端未启动，请先运行 backend/app.py。", true);
    return false;
  }
}

async function runDetect() {
  if (!selectedFile) {
    setMessage("请先打开一张图像。", true);
    return;
  }

  detectButton.disabled = true;
  openButton.disabled = true;
  saveButton.disabled = true;
  profileLabel.textContent = "检测中…";
  resultBadge.textContent = "推理中";
  confidenceLabel.textContent = "-";
  setPlaceholder(resultPreview, "检测中…", true);
  setMessage("正在调用经典+ML 管线检测车道线…");

  const formData = new FormData();
  formData.append("file", selectedFile);
  const started = performance.now();

  try {
    const response = await fetch(`${API_BASE}/api/detect`, { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error(data.error || "检测失败");

    const clientMs = Math.round(performance.now() - started);
    const serverMs = typeof data.elapsedMs === "number" ? data.elapsedMs : clientMs;

    resultImageData = data.resultImage;
    resultFileName = resultNameFrom(selectedFile);
    setImage(resultPreview, resultImageData, "车道线检测结果");
    resultBadge.textContent = "完成";

    const total = data.laneCount ?? data.totalCount ?? 0;
    laneCount.textContent = String(total);
    profileLabel.textContent = data.profileLabel || "智能检测";
    profileScript.textContent = data.profileScript || "—";
    confidenceLabel.textContent = confidenceText(data.profileConfidence);
    confidenceBar.style.width = `${Math.round((data.profileConfidence || 0) * 100)}%`;
    timeLabel.textContent = `${serverMs} ms`;
    reasonText.textContent = data.profileReason || "—";

    scanTotal += 1;
    scanCount.textContent = String(scanTotal);
    history.push({
      name: data.fileName || selectedFile.name,
      total,
      label: data.profileLabel || "-",
      ms: serverMs,
    });
    renderHistory();

    pulseMetric();
    saveButton.disabled = false;
    setMessage("");
    showToast(`检测完成：${total} 条车道线`);
  } catch (error) {
    profileLabel.textContent = "-";
    resultBadge.textContent = "失败";
    setPlaceholder(resultPreview, "检测失败", false);
    setMessage(error.message || "检测失败，请检查后端服务。", true);
    showToast(error.message || "检测失败", true);
    await checkBackend();
  } finally {
    detectButton.disabled = false;
    openButton.disabled = false;
  }
}

/* ---------------- events ---------------- */

actionToggle.addEventListener("click", (event) => {
  animateButton(actionToggle, event);
  setDeckOpen(!commandDeck.classList.contains("open"));
});

[openButton, detectButton, saveButton, sampleButton, themeButton].forEach((button) => {
  button.addEventListener("click", (event) => animateButton(button, event));
});

themeButton.addEventListener("click", cycleTheme);
openButton.addEventListener("click", () => fileInput.click());

sampleButton.addEventListener("click", () => {
  tabs.querySelector('[data-tab="batch"]').click();
  showToast("切换到批量场景演示");
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  loadFile(file);
});

detectButton.addEventListener("click", runDetect);

sourcePreview.addEventListener("click", (event) => {
  const image = event.target instanceof Element ? event.target.closest("img") : null;
  if (image) {
    openImageModal(image.src);
  } else {
    fileInput.click();
  }
});

resultPreview.addEventListener("click", (event) => {
  const image = event.target instanceof Element ? event.target.closest("img") : null;
  if (image) openImageModal(image.src);
});

/* tabs */
tabs.addEventListener("click", (event) => {
  const tab = event.target.closest(".tab");
  if (!tab) return;
  tabs.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
  const target = tab.dataset.tab;
  document.querySelectorAll(".panel").forEach((p) => {
    p.classList.toggle("active", p.dataset.panel === target);
  });
});

/* cursor glow */
detectorWindow.addEventListener("pointermove", (event) => {
  const rect = detectorWindow.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * 100;
  const y = ((event.clientY - rect.top) / rect.height) * 100;
  detectorWindow.style.setProperty("--cursor-x", `${x.toFixed(2)}%`);
  detectorWindow.style.setProperty("--cursor-y", `${y.toFixed(2)}%`);
});

/* drag & drop */
["dragenter", "dragover"].forEach((type) => {
  detectorWindow.addEventListener(type, (event) => {
    event.preventDefault();
    detectorWindow.classList.add("dragging");
    sourcePreview.classList.add("active");
  });
});

["dragleave", "drop"].forEach((type) => {
  detectorWindow.addEventListener(type, (event) => {
    event.preventDefault();
    detectorWindow.classList.remove("dragging");
    sourcePreview.classList.remove("active");
  });
});

detectorWindow.addEventListener("drop", (event) => {
  const file = event.dataTransfer?.files?.[0];
  loadFile(file);
});

modalClose.addEventListener("click", closeImageModal);
imageModal.addEventListener("click", (event) => {
  if (event.target === imageModal) closeImageModal();
});

saveButton.addEventListener("click", () => {
  if (!resultImageData) return;
  const link = document.createElement("a");
  link.href = resultImageData;
  link.download = resultFileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  showToast("结果已保存");
});

/* keyboard shortcuts */
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !imageModal.hidden) {
    closeImageModal();
    return;
  }
  if (event.target instanceof HTMLInputElement) return;
  const key = event.key.toLowerCase();
  if (key === "o") {
    fileInput.click();
  } else if (event.key === "Enter" && !detectButton.disabled) {
    runDetect();
  } else if (key === "s" && !saveButton.disabled) {
    saveButton.click();
  } else if (key === "t") {
    cycleTheme();
  }
});

/* init */
renderBatch();
renderHistory();
checkBackend();
