import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const EXT_NAME = "layer13.persistent_counter";
const NODE_TYPE = "Layer13PersistentCounter";

function findWidget(node, name) {
  return node?.widgets?.find((widget) => widget?.name === name) || null;
}

function getCounterName(node) {
  const widget = findWidget(node, "计数器名称");
  const raw = widget?.value;
  const value = typeof raw === "string" ? raw.trim() : String(raw ?? "").trim();
  return value || "default";
}

function setStatus(node, text, tone = "muted") {
  const state = node.__layer13PersistentCounter;
  if (!state?.status) return;

  state.status.textContent = text;
  if (tone === "success") {
    state.status.style.color = "#7ee787";
  } else if (tone === "error") {
    state.status.style.color = "#ff7b72";
  } else {
    state.status.style.color = "#bdbdbd";
  }
}

async function resetCounter(node) {
  const state = node.__layer13PersistentCounter;
  if (!state) return;

  const counterName = getCounterName(node);
  state.button.disabled = true;
  setStatus(node, `正在重置: ${counterName}`, "muted");

  try {
    const response = await api.fetchApi(
      `/layer13/persistent_counter/reset?name=${encodeURIComponent(counterName)}&_ts=${Date.now()}`,
      {
        method: "GET",
        cache: "no-store",
      },
    );

    if (response.status !== 200) {
      throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    const message = data?.existed
      ? `已重置: ${data.name}`
      : `已清空(原本无状态): ${data.name}`;
    setStatus(node, message, "success");
  } catch (error) {
    console.error("Layer13PersistentCounter reset error:", error);
    setStatus(node, `重置失败: ${error.message}`, "error");
  } finally {
    state.button.disabled = false;
    node.setDirtyCanvas?.(true, true);
  }
}

function ensureNodeUI(node) {
  if (node.__layer13PersistentCounter) {
    return;
  }

  const container = document.createElement("div");
  container.style.display = "flex";
  container.style.flexDirection = "column";
  container.style.gap = "6px";
  container.style.width = "100%";
  container.style.boxSizing = "border-box";
  container.style.padding = "2px 0 4px";

  const button = document.createElement("button");
  button.type = "button";
  button.textContent = "手动重置";
  button.style.width = "100%";
  button.style.minHeight = "30px";
  button.style.cursor = "pointer";

  const status = document.createElement("div");
  status.style.fontSize = "12px";
  status.style.lineHeight = "1.4";
  status.style.color = "#bdbdbd";
  status.textContent = "点击后立即清空当前计数器状态";

  button.addEventListener("click", async () => {
    app.canvas.node_widget = null;
    await resetCounter(node);
  });

  container.appendChild(button);
  container.appendChild(status);

  const widget = node.addDOMWidget("layer13_persistent_counter_reset", "reset", container, {
    serialize: false,
    hideOnZoom: false,
    getValue() {
      return null;
    },
    setValue() {},
  });
  widget.computeSize = (width) => [Math.max(220, width), 52];

  node.__layer13PersistentCounter = {
    button,
    status,
    widget,
  };
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
      ensureNodeUI(this);
      return result;
    };
  },
});
