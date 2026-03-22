import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT_NAME = "layer13.multi_image_input";
const NODE_TYPE = "Layer13MultiImageInput";
const THUMBS_PER_ROW = 3;
const CARD_MIN_WIDTH = 108;
const GRID_GAP = 8;
const THUMB_HEIGHT = 112;
const HEADER_HEIGHT = 76;
const NODE_MIN_WIDTH = 360;
const NODE_MIN_HEIGHT = 240;
const NODE_WIDTH_PADDING = 28;
const NODE_HEIGHT_PADDING = 96;

function getCurrentItems(node) {
  return parseItems(node?.__layer13MultiImage?.jsonWidget?.value);
}

function estimateLayout(count) {
  const safeCount = Math.max(count, 1);
  const columns = Math.max(1, Math.min(THUMBS_PER_ROW, safeCount));
  const rows = Math.max(1, Math.ceil(safeCount / THUMBS_PER_ROW));
  const width = Math.max(NODE_MIN_WIDTH, columns * CARD_MIN_WIDTH + (columns - 1) * GRID_GAP + 44);
  const height = HEADER_HEIGHT + rows * THUMB_HEIGHT + 22;
  return { columns, rows, width, height };
}

function moveItem(items, fromIndex, toIndex) {
  const next = [...items];
  const [moved] = next.splice(fromIndex, 1);
  next.splice(toIndex, 0, moved);
  return next;
}

function removeItem(node, index) {
  const items = getCurrentItems(node);
  if (index < 0 || index >= items.length) return;
  items.splice(index, 1);
  syncWidgetValue(node, items);
  renderItems(node, items);
}

function setDropZoneActive(state, active) {
  if (!state?.dropZone) return;
  state.dropZone.style.borderColor = active ? "#58a6ff" : "rgba(255,255,255,0.18)";
  state.dropZone.style.background = active ? "rgba(88,166,255,0.12)" : "rgba(255,255,255,0.03)";
  state.dropZone.style.color = active ? "#dbeafe" : "#bdbdbd";
}

function hideWidget(widget) {
  if (!widget || widget.__layer13Hidden) return;
  widget.__layer13Hidden = true;
  widget.hidden = true;
  widget.origComputeSize = widget.computeSize;
  widget.computeSize = () => [0, -4];
}

function parseItems(rawValue) {
  if (!rawValue) return [];
  try {
    const parsed = typeof rawValue === "string" ? JSON.parse(rawValue) : rawValue;
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function buildViewUrl(item) {
  const params = new URLSearchParams();
  params.set("filename", item.name || "");
  params.set("type", item.type || "input");
  if (item.subfolder) {
    params.set("subfolder", item.subfolder);
  }
  params.set("size", "192");
  return api.apiURL(`/layer13/thumb?${params.toString()}`);
}

function estimateHeight(count) {
  const rows = Math.max(1, Math.ceil(Math.max(count, 1) / THUMBS_PER_ROW));
  return HEADER_HEIGHT + rows * THUMB_HEIGHT;
}

function scheduleNodeResize(node, layout = null) {
  const state = node.__layer13MultiImage;
  if (!state || state.resizeScheduled) return;
  state.resizeScheduled = true;

  requestAnimationFrame(() => {
    state.resizeScheduled = false;
    const fallback = layout || estimateLayout(getCurrentItems(node).length);
    const contentWidth = Math.ceil(Math.max(fallback.width, NODE_MIN_WIDTH));
    const contentHeight = Math.ceil(Math.max(
      fallback.height,
      NODE_MIN_HEIGHT,
      state.container?.scrollHeight || 0,
    ));

    state.previewWidget.computeSize = () => [contentWidth, contentHeight];

    if (typeof node.setSize === "function") {
      const targetWidth = Math.max(contentWidth + NODE_WIDTH_PADDING, NODE_MIN_WIDTH);
      const targetHeight = Math.max(contentHeight + NODE_HEIGHT_PADDING, NODE_MIN_HEIGHT);
      const lastWidth = state.lastAppliedWidth || 0;
      const lastHeight = state.lastAppliedHeight || 0;

      if (Math.abs(targetWidth - lastWidth) > 1 || Math.abs(targetHeight - lastHeight) > 1) {
        state.lastAppliedWidth = targetWidth;
        state.lastAppliedHeight = targetHeight;
        node.setSize([targetWidth, targetHeight]);
      }
    }
    node.setDirtyCanvas?.(true, true);
  });
}

function syncWidgetValue(node, items) {
  const jsonWidget = node.__layer13MultiImage?.jsonWidget;
  if (!jsonWidget) return;
  jsonWidget.value = JSON.stringify(items, null, 2);
  if (typeof jsonWidget.callback === "function") {
    jsonWidget.callback(jsonWidget.value);
  }
}

function renderItems(node, items) {
  const state = node.__layer13MultiImage;
  if (!state) return;
  const layout = estimateLayout(items.length);
  const { columns } = layout;

  state.grid.innerHTML = "";
  state.grid.style.gridTemplateColumns = `repeat(${columns}, minmax(0, 1fr))`;
  state.summary.textContent = items.length ? `已选择 ${items.length} 张图` : "未选择图片";
  state.clearButton.disabled = items.length === 0;
  state.dropZoneText.textContent = items.length
    ? "可继续拖动图片到这里追加，或点击重新选择"
    : "拖动多张图片到这里，或点击选择";

  for (const [index, item] of items.entries()) {
    const card = document.createElement("div");
    card.style.display = "flex";
    card.style.flexDirection = "column";
    card.style.gap = "4px";
    card.style.padding = "6px";
    card.style.background = "rgba(255,255,255,0.04)";
    card.style.border = "1px solid rgba(255,255,255,0.08)";
    card.style.borderRadius = "8px";
    card.style.cursor = "grab";
    card.draggable = true;

    const thumbWrap = document.createElement("div");
    thumbWrap.style.position = "relative";
    thumbWrap.style.width = "100%";
    thumbWrap.style.aspectRatio = "1 / 1";
    thumbWrap.style.overflow = "hidden";
    thumbWrap.style.borderRadius = "6px";
    thumbWrap.style.background = "#222";

    const badge = document.createElement("div");
    badge.textContent = String(index + 1);
    badge.style.position = "absolute";
    badge.style.left = "6px";
    badge.style.top = "6px";
    badge.style.minWidth = "20px";
    badge.style.height = "20px";
    badge.style.padding = "0 6px";
    badge.style.display = "flex";
    badge.style.alignItems = "center";
    badge.style.justifyContent = "center";
    badge.style.background = "rgba(0,0,0,0.75)";
    badge.style.color = "#fff";
    badge.style.fontSize = "12px";
    badge.style.fontWeight = "700";
    badge.style.borderRadius = "999px";
    badge.style.zIndex = "2";

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.textContent = "×";
    removeButton.title = "删除这张图";
    removeButton.style.position = "absolute";
    removeButton.style.right = "6px";
    removeButton.style.top = "6px";
    removeButton.style.width = "22px";
    removeButton.style.height = "22px";
    removeButton.style.display = "flex";
    removeButton.style.alignItems = "center";
    removeButton.style.justifyContent = "center";
    removeButton.style.border = "none";
    removeButton.style.borderRadius = "999px";
    removeButton.style.background = "rgba(0,0,0,0.72)";
    removeButton.style.color = "#fff";
    removeButton.style.fontSize = "16px";
    removeButton.style.lineHeight = "1";
    removeButton.style.cursor = "pointer";
    removeButton.style.zIndex = "2";
    removeButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      removeItem(node, index);
    });

    const img = document.createElement("img");
    img.src = buildViewUrl(item);
    img.alt = item.name || `image-${index + 1}`;
    img.draggable = false;
    img.loading = "lazy";
    img.decoding = "async";
    img.fetchPriority = "low";
    img.style.width = "100%";
    img.style.height = "100%";
    img.style.objectFit = "cover";
    img.style.display = "block";
    img.addEventListener("load", () => scheduleNodeResize(node, layout), { once: true });

    const filename = document.createElement("div");
    filename.textContent = item.name || "未命名";
    filename.title = item.name || "";
    filename.style.fontSize = "11px";
    filename.style.lineHeight = "1.3";
    filename.style.color = "#ccc";
    filename.style.wordBreak = "break-all";

    thumbWrap.appendChild(img);
    thumbWrap.appendChild(badge);
    thumbWrap.appendChild(removeButton);
    card.appendChild(thumbWrap);
    card.appendChild(filename);

    card.addEventListener("dragstart", (event) => {
      state.dragIndex = index;
      card.style.opacity = "0.45";
      if (event.dataTransfer) {
        event.dataTransfer.effectAllowed = "move";
        event.dataTransfer.setData("text/plain", String(index));
      }
    });

    card.addEventListener("dragend", () => {
      state.dragIndex = null;
      card.style.opacity = "1";
      state.grid.querySelectorAll("[data-layer13-card]").forEach((element) => {
        element.style.borderColor = "rgba(255,255,255,0.08)";
      });
    });

    card.dataset.layer13Card = "1";
    card.addEventListener("dragover", (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (state.dragIndex == null || state.dragIndex === index) return;
      card.style.borderColor = "#58a6ff";
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "move";
      }
    });

    card.addEventListener("dragleave", () => {
      card.style.borderColor = "rgba(255,255,255,0.08)";
    });

    card.addEventListener("drop", (event) => {
      event.preventDefault();
      event.stopPropagation();
      card.style.borderColor = "rgba(255,255,255,0.08)";
      if (state.dragIndex == null || state.dragIndex === index) return;
      const reordered = moveItem(getCurrentItems(node), state.dragIndex, index);
      syncWidgetValue(node, reordered);
      renderItems(node, reordered);
    });

    state.grid.appendChild(card);
  }

  scheduleNodeResize(node, layout);
}

async function uploadFiles(node, files) {
  const state = node.__layer13MultiImage;
  if (!state) return;

  const imageFiles = Array.from(files || []).filter((file) => file?.type?.startsWith?.("image/"));
  if (!imageFiles.length) {
    state.summary.textContent = "没有可上传的图片文件";
    return;
  }

  state.uploadButton.disabled = true;
  state.clearButton.disabled = true;
  setDropZoneActive(state, false);
  state.summary.textContent = `正在上传 0/${imageFiles.length}`;

  const uploaded = [];
  try {
    for (const [index, file] of imageFiles.entries()) {
      const body = new FormData();
      body.append("image", file);
      const resp = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
      });
      if (!(resp.status === 200 || resp.status === 201)) {
        throw new Error(`上传失败: ${resp.status} ${resp.statusText}`);
      }
      const data = await resp.json();
      uploaded.push({
        name: data.name,
        subfolder: data.subfolder || "",
        type: data.type || "input",
      });
      state.summary.textContent = `正在上传 ${index + 1}/${imageFiles.length}`;
    }

    const merged = [...getCurrentItems(node), ...uploaded];
    syncWidgetValue(node, merged);
    renderItems(node, merged);
    state.summary.textContent = `已选择 ${merged.length} 张图`;
  } catch (error) {
    console.error("Layer13MultiImageInput upload error:", error);
    state.summary.textContent = `上传失败: ${error.message}`;
  } finally {
    state.uploadButton.disabled = false;
    state.clearButton.disabled = parseItems(state.jsonWidget.value).length === 0;
  }
}

function ensureNodeUI(node) {
  if (node.__layer13MultiImage) {
    renderItems(node, parseItems(node.__layer13MultiImage.jsonWidget.value));
    return;
  }

  const jsonWidget = node.widgets?.find((widget) => widget.name === "文件列表JSON");
  if (!jsonWidget) {
    console.error("Layer13MultiImageInput: missing 文件列表JSON widget", node.id);
    return;
  }
  hideWidget(jsonWidget);

  const container = document.createElement("div");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.gap = "8px";
  container.style.width = "100%";
  container.style.boxSizing = "border-box";

  const toolbar = document.createElement("div");
  toolbar.style.display = "flex";
  toolbar.style.gap = "8px";

  const uploadButton = document.createElement("button");
  uploadButton.textContent = "选择多张图片";
  uploadButton.style.flex = "1";

  const clearButton = document.createElement("button");
  clearButton.textContent = "清空";
  clearButton.style.width = "68px";

  const summary = document.createElement("div");
  summary.style.fontSize = "12px";
  summary.style.color = "#bbb";
  summary.textContent = "未选择图片";

  const dropZone = document.createElement("button");
  dropZone.type = "button";
  dropZone.style.display = "flex";
  dropZone.style.alignItems = "center";
  dropZone.style.justifyContent = "center";
  dropZone.style.minHeight = "74px";
  dropZone.style.padding = "12px";
  dropZone.style.background = "rgba(255,255,255,0.03)";
  dropZone.style.border = "1px dashed rgba(255,255,255,0.18)";
  dropZone.style.borderRadius = "10px";
  dropZone.style.color = "#bdbdbd";
  dropZone.style.fontSize = "12px";
  dropZone.style.lineHeight = "1.5";
  dropZone.style.textAlign = "center";
  dropZone.style.cursor = "pointer";

  const dropZoneText = document.createElement("div");
  dropZoneText.textContent = "拖动多张图片到这里，或点击选择";
  dropZone.appendChild(dropZoneText);

  const grid = document.createElement("div");
  grid.style.display = "grid";
  grid.style.gridTemplateColumns = "repeat(3, minmax(0, 1fr))";
  grid.style.gap = "8px";
  grid.style.alignItems = "start";

  toolbar.appendChild(uploadButton);
  toolbar.appendChild(clearButton);
  container.appendChild(toolbar);
  container.appendChild(summary);
  container.appendChild(dropZone);
  container.appendChild(grid);

  const previewWidget = node.addDOMWidget("layer13_multi_image_preview", "preview", container, {
    serialize: false,
    hideOnZoom: false,
    getValue() {
      return null;
    },
    setValue() {},
  });
  previewWidget.computeSize = (width) => [Math.max(NODE_MIN_WIDTH, width), estimateHeight(0)];

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = "image/*";
  fileInput.multiple = true;
  fileInput.style.display = "none";
  fileInput.addEventListener("change", async () => {
    if (fileInput.files?.length) {
      await uploadFiles(node, fileInput.files);
    }
    fileInput.value = "";
  });
  document.body.appendChild(fileInput);

  const originalRemoved = node.onRemoved;
  node.onRemoved = function () {
    fileInput.remove();
    return originalRemoved?.apply(this, arguments);
  };

  uploadButton.addEventListener("click", () => {
    app.canvas.node_widget = null;
    fileInput.click();
  });

  dropZone.addEventListener("click", () => {
    app.canvas.node_widget = null;
    fileInput.click();
  });

  for (const eventName of ["dragenter", "dragover"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      if (event.dataTransfer) {
        event.dataTransfer.dropEffect = "copy";
      }
      setDropZoneActive(node.__layer13MultiImage, true);
    });
  }

  for (const eventName of ["dragleave", "dragend"]) {
    dropZone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
      setDropZoneActive(node.__layer13MultiImage, false);
    });
  }

  dropZone.addEventListener("drop", async (event) => {
    event.preventDefault();
    event.stopPropagation();
    setDropZoneActive(node.__layer13MultiImage, false);
    const files = Array.from(event.dataTransfer?.files || []).filter((file) => file?.type?.startsWith?.("image/"));
    if (files.length) {
      await uploadFiles(node, files);
    }
  });

  clearButton.addEventListener("click", () => {
    syncWidgetValue(node, []);
    renderItems(node, []);
  });

  node.onDragOver = (event) => {
    const allowed = !!event?.dataTransfer?.types?.includes?.("Files");
    setDropZoneActive(node.__layer13MultiImage, allowed);
    return allowed;
  };
  node.onDragDrop = async (event) => {
    setDropZoneActive(node.__layer13MultiImage, false);
    const files = Array.from(event?.dataTransfer?.files || []).filter((file) => file?.type?.startsWith?.("image/"));
    if (!files.length) {
      return false;
    }
    await uploadFiles(node, files);
    return true;
  };

  node.__layer13MultiImage = {
    jsonWidget,
    previewWidget,
    uploadButton,
    clearButton,
    summary,
    dropZone,
    dropZoneText,
    dragIndex: null,
    grid,
    container,
    resizeScheduled: false,
    lastAppliedWidth: 0,
    lastAppliedHeight: 0,
  };

  renderItems(node, parseItems(jsonWidget.value));
}

app.registerExtension({
  name: EXT_NAME,
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      ensureNodeUI(this);
      return result;
    };

    const originalOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalOnConfigure?.apply(this, arguments);
      requestAnimationFrame(() => ensureNodeUI(this));
      return result;
    };
  },
});
