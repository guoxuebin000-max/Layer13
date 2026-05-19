import { app } from "../../scripts/app.js";

const EXT_NAME = "layer13.random_loader.fix";
const NODE_TYPE = "Layer13RandomImageLoaderV2";

function sanitizeStartIndexWidget(node) {
  if (!node || !node.widgets) return;
  const w = node.widgets.find((x) => x.name === "起始编号");
  if (!w) return;
  if (w.value === "" || w.value === null || typeof w.value === "undefined") {
    w.value = 1;
  } else if (typeof w.value === "string" && w.value.trim() === "") {
    w.value = 1;
  } else if (typeof w.value === "number" && !Number.isFinite(w.value)) {
    w.value = 1;
  } else if (Number.isNaN(Number(w.value))) {
    w.value = 1;
  }
}

function sanitizePromptObject(res) {
  try {
    const output = res?.output ?? res;
    const prompt = output?.prompt ?? output;
    if (!prompt || typeof prompt !== "object") return;
    for (const key of Object.keys(prompt)) {
      const node = prompt[key];
      if (!node || node.class_type !== NODE_TYPE) continue;
      if (!node.inputs) continue;
      const v = node.inputs["起始编号"];
      if (Array.isArray(v) || (v && typeof v === "object")) {
        continue;
      }
      if (v === "" || v === null || typeof v === "undefined") {
        node.inputs["起始编号"] = 1;
      } else if (typeof v === "string" && v.trim() === "") {
        node.inputs["起始编号"] = 1;
      } else if (Number.isNaN(Number(v))) {
        node.inputs["起始编号"] = 1;
      }
    }
  } catch (e) {
    // ignore
  }
}

app.registerExtension({
  name: EXT_NAME,
  async setup() {
    if (app._layer13_random_loader_patched) return;
    app._layer13_random_loader_patched = true;
    const origGraphToPrompt = app.graphToPrompt;
    app.graphToPrompt = function () {
      const res = origGraphToPrompt.apply(this, arguments);
      const nodes = app.graph?._nodes || [];
      for (const node of nodes) {
        if (node?.type === NODE_TYPE) sanitizeStartIndexWidget(node);
      }
      sanitizePromptObject(res);
      return res;
    };

    const origQueuePrompt = app.queuePrompt;
    if (origQueuePrompt) {
      app.queuePrompt = async function () {
        const nodes = app.graph?._nodes || [];
        for (const node of nodes) {
          if (node?.type === NODE_TYPE) sanitizeStartIndexWidget(node);
        }
        return origQueuePrompt.apply(this, arguments);
      };
    }
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;

    const origOnAdded = nodeType.prototype.onAdded;
    nodeType.prototype.onAdded = function () {
      origOnAdded?.apply(this, arguments);
      sanitizeStartIndexWidget(this);
    };

    const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function (info) {
      origOnConfigure?.apply(this, arguments);
      sanitizeStartIndexWidget(this);
    };

    const origOnSerialize = nodeType.prototype.onSerialize;
    nodeType.prototype.onSerialize = function (o) {
      sanitizeStartIndexWidget(this);
      return origOnSerialize ? origOnSerialize.call(this, o) : o;
    };
  },
});
