const API_BASE = "http://127.0.0.1:5000";
const THEMES = ["green", "blue", "amber", "violet"];

const fileInput = document.querySelector("#fileInput");
const openButton = document.querySelector("#openButton");
const detectButton = document.querySelector("#detectButton");
const saveButton = document.querySelector("#saveButton");
const themeButton = document.querySelector("#themeButton");
const actionToggle = document.querySelector("#actionToggle");
const commandDeck = document.querySelector("#commandDeck");
const fileName = document.querySelector("#fileName");
const sourcePreview = document.querySelector("#sourcePreview");
const resultPreview = document.querySelector("#resultPreview");
const serverStatus = document.querySelector("#serverStatus");
const message = document.querySelector("#message");
const solidCount = document.querySelector("#solidCount");
const dashedCount = document.querySelector("#dashedCount");
const totalCount = document.querySelector("#totalCount");
const profileLabel = document.querySelector("#profileLabel");

let selectedFile = null;
let previewUrl = null;
let resultImageData = null;
let resultFileName = "lane-detection-result.png";
let themeIndex = Math.max(0, THEMES.indexOf(localStorage.getItem("lane-theme") || "green"));

document.body.dataset.theme = THEMES[themeIndex];

function setMessage(text, isError = false) {
  message.textContent = text;
  message.classList.toggle("error", isError);
  message.hidden = !text;
}

function setImage(container, src, alt) {
  container.innerHTML = "";
  const image = document.createElement("img");
  image.src = src;
  image.alt = alt;
  container.appendChild(image);
}

function setPlaceholder(container, text) {
  container.innerHTML = "";
  const span = document.createElement("span");
  span.textContent = text;
  container.appendChild(span);
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
  solidCount.textContent = "0";
  dashedCount.textContent = "0";
  totalCount.textContent = "0";
  profileLabel.textContent = "-";
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
  if (button.disabled) {
    return;
  }
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
}

async function checkBackend() {
  try {
    const response = await fetch(`${API_BASE}/api/health`);
    if (!response.ok) {
      throw new Error("backend unavailable");
    }
    serverStatus.textContent = "已连接";
    serverStatus.className = "status online";
  } catch {
    serverStatus.textContent = "未连接";
    serverStatus.className = "status offline";
    setMessage("后端未启动，请先运行 backend/app.py。", true);
  }
}

actionToggle.addEventListener("click", (event) => {
  animateButton(actionToggle, event);
  setDeckOpen(!commandDeck.classList.contains("open"));
});

[openButton, detectButton, saveButton, themeButton].forEach((button) => {
  button.addEventListener("click", (event) => animateButton(button, event));
});

themeButton.addEventListener("click", cycleTheme);

openButton.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    return;
  }

  selectedFile = file;
  detectButton.disabled = false;
  fileName.textContent = file.name;
  setPlaceholder(resultPreview, "等待检测");
  resetResult();
  setMessage("图片已载入，可以开始检测。");

  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  setImage(sourcePreview, previewUrl, "原始图像预览");
});

detectButton.addEventListener("click", async () => {
  if (!selectedFile) {
    setMessage("请先打开一张图像。", true);
    return;
  }

  detectButton.disabled = true;
  openButton.disabled = true;
  saveButton.disabled = true;
  profileLabel.textContent = "检测中";
  setMessage("正在检测车道线...");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(`${API_BASE}/api/detect`, {
      method: "POST",
      body: formData,
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || "检测失败");
    }

    resultImageData = data.resultImage;
    resultFileName = resultNameFrom(selectedFile);
    setImage(resultPreview, resultImageData, "车道线检测结果");
    solidCount.textContent = String(data.solidCount);
    dashedCount.textContent = String(data.dashedCount);
    totalCount.textContent = String(data.totalCount);
    profileLabel.textContent = "智能检测";
    saveButton.disabled = false;
    setMessage("");
    await checkBackend();
  } catch (error) {
    profileLabel.textContent = "-";
    setMessage(error.message || "检测失败，请检查后端服务。", true);
    await checkBackend();
  } finally {
    detectButton.disabled = false;
    openButton.disabled = false;
  }
});

saveButton.addEventListener("click", () => {
  if (!resultImageData) {
    return;
  }
  const link = document.createElement("a");
  link.href = resultImageData;
  link.download = resultFileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
});

checkBackend();
