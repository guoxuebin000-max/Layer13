import { app } from "../../scripts/app.js";

const NODE_CONFIGS = {
  Layer13AnyIndexSwitch: {
    fixedInputs: new Set(["选择"]),
    usesDynamicInputs: true,
    syncOutputType: true,
    trailingInputType: null,
    dynamicOutputs: false,
  },
  Layer13InputCountControl: {
    fixedInputs: new Set(["值", "最小值", "最大值"]),
    usesDynamicInputs: false,
    syncOutputType: false,
    trailingInputType: null,
    dynamicOutputs: false,
    syncIntegerRange: true,
  },
};

const INPUT_NAME_RE = /^input(\d+)$/;

function getConfig(node) {
  return NODE_CONFIGS[node.comfyClass] || null;
}

function parseInputIndex(name) {
  const match = INPUT_NAME_RE.exec(name || "");
  return match ? Number.parseInt(match[1], 10) : null;
}

function getDynamicInputs(node, config) {
  return (node.inputs || []).filter((input) => {
    if (config.fixedInputs.has(input.name)) {
      return false;
    }
    return parseInputIndex(input.name) != null;
  });
}

function getDynamicInputsSorted(node, config) {
  return [...getDynamicInputs(node, config)].sort((a, b) => parseInputIndex(a.name) - parseInputIndex(b.name));
}

function getSelectWidget(node) {
  return (node.widgets || []).find((widget) => widget.name === "选择");
}

function getWidget(node, name) {
  return (node.widgets || []).find((widget) => widget.name === name);
}

function getFirstExistingWidget(node, names) {
  for (const name of names) {
    const widget = getWidget(node, name);
    if (widget) {
      return widget;
    }
  }
  return null;
}

function getConnectedIndices(node, config) {
  return getDynamicInputsSorted(node, config)
    .filter((input) => input.link != null)
    .map((input) => parseInputIndex(input.name));
}

function snapToNearestIndex(value, validIndices) {
  if (!validIndices.length) {
    return Math.max(1, Math.trunc(Number(value) || 1));
  }

  const target = Math.trunc(Number(value) || validIndices[0]);
  if (validIndices.includes(target)) {
    return target;
  }

  return validIndices.reduce((best, current) => {
    const bestDelta = Math.abs(best - target);
    const currentDelta = Math.abs(current - target);
    if (currentDelta < bestDelta) {
      return current;
    }
    if (currentDelta === bestDelta && current < best) {
      return current;
    }
    return best;
  }, validIndices[0]);
}

function updateSelectBounds(node, config) {
  const selectWidget = getSelectWidget(node);
  if (!selectWidget) {
    return;
  }

  const connectedIndices = getConnectedIndices(node, config);
  const dynamicInputs = getDynamicInputsSorted(node, config);
  const highestExisting = dynamicInputs.length
    ? parseInputIndex(dynamicInputs[dynamicInputs.length - 1].name)
    : 1;

  const minValue = connectedIndices.length ? connectedIndices[0] : 1;
  const maxValue = connectedIndices.length ? connectedIndices[connectedIndices.length - 1] : Math.max(1, highestExisting);

  selectWidget.options = selectWidget.options || {};
  selectWidget.options.min = minValue;
  selectWidget.options.max = maxValue;

  if (typeof selectWidget.value !== "number") {
    selectWidget.value = minValue;
  }

  if (connectedIndices.length) {
    selectWidget.value = snapToNearestIndex(selectWidget.value, connectedIndices);
  } else {
    selectWidget.value = Math.min(Math.max(minValue, Math.trunc(Number(selectWidget.value) || minValue)), maxValue);
  }
}

function syncTypes(node, config, forcedType = null) {
  if (!config.syncOutputType) {
    return;
  }

  const dynamicInputs = getDynamicInputsSorted(node, config);
  const connectedType = forcedType
    || dynamicInputs.find((input) => input.link != null && input.type && input.type !== "*")?.type
    || "*";

  for (const input of dynamicInputs) {
    input.type = connectedType;
  }

  if (node.outputs?.[0]) {
    node.outputs[0].type = connectedType;
    node.outputs[0].name = connectedType;
    node.outputs[0].label = connectedType;
  }
}

function syncDynamicOutputs(node, config) {
  if (!config.dynamicOutputs) {
    return;
  }

  const baseOutput = node.outputs?.[0];
  if (!baseOutput) {
    return;
  }

  baseOutput.name = "selected_index";
  baseOutput.label = "selected_index";
  baseOutput.type = "INT";

  const connectedInputs = getDynamicInputsSorted(node, config).filter((input) => input.link != null);
  const desiredDynamicCount = connectedInputs.length;
  const desiredTotal = 1 + desiredDynamicCount;

  while ((node.outputs || []).length > desiredTotal) {
    node.removeOutput((node.outputs || []).length - 1);
  }

  while ((node.outputs || []).length < desiredTotal) {
    const nextIndex = (node.outputs || []).length;
    node.addOutput(`output${nextIndex}`, "*");
  }

  for (let i = 0; i < desiredDynamicCount; i += 1) {
    const output = node.outputs?.[i + 1];
    if (!output) {
      continue;
    }

    output.name = `output${i + 1}`;
    output.label = `output${i + 1}`;
    output.type = connectedInputs[i].type && connectedInputs[i].type !== "*" ? connectedInputs[i].type : "*";
  }
}

function ensureTrailingInput(node, config) {
  const dynamicInputs = getDynamicInputsSorted(node, config);
  const inputType = config.trailingInputType ?? node.outputs?.[0]?.type ?? "*";
  if (!dynamicInputs.length) {
    node.addInput("input1", inputType);
    return;
  }

  const lastInput = dynamicInputs[dynamicInputs.length - 1];
  const lastIndex = parseInputIndex(lastInput.name);
  if (lastInput.link != null) {
    node.addInput(`input${lastIndex + 1}`, inputType);
  }
}

function pruneExtraEmptyInputs(node, config) {
  const dynamicInputs = getDynamicInputsSorted(node, config);
  const emptyInputs = dynamicInputs.filter((input) => input.link == null);
  if (emptyInputs.length <= 1) {
    return;
  }

  const keepInput = emptyInputs.reduce((best, current) => {
    return parseInputIndex(current.name) > parseInputIndex(best.name) ? current : best;
  });

  const removeIndexes = emptyInputs
    .filter((input) => input !== keepInput)
    .map((input) => (node.inputs || []).indexOf(input))
    .filter((index) => index >= 0)
    .sort((a, b) => b - a);

  for (const index of removeIndexes) {
    node.removeInput(index);
  }
}

function installSelectSnapper(node, config) {
  const selectWidget = getSelectWidget(node);
  if (!selectWidget) {
    return;
  }

  if (!selectWidget._layer13OriginalCallback) {
    selectWidget._layer13OriginalCallback = selectWidget.callback;
  }

  selectWidget.callback = function (value, ...rest) {
    const connectedIndices = getConnectedIndices(node, config);
    let nextValue;

    if (connectedIndices.length) {
      nextValue = snapToNearestIndex(value, connectedIndices);
    } else {
      const minValue = Number(selectWidget.options?.min ?? 1);
      const maxValue = Number(selectWidget.options?.max ?? minValue);
      nextValue = Math.min(Math.max(minValue, Math.trunc(Number(value) || minValue)), maxValue);
    }

    selectWidget.value = nextValue;

    if (typeof selectWidget._layer13OriginalCallback === "function") {
      return selectWidget._layer13OriginalCallback.call(this, nextValue, ...rest);
    }

    return nextValue;
  };
}

function installIntegerRangeSync(node, config) {
  if (!config.syncIntegerRange) {
    return;
  }

  const selectWidget = getFirstExistingWidget(node, ["值", "选择"]);
  const minWidget = getWidget(node, "最小值");
  const maxWidget = getWidget(node, "最大值");
  if (!selectWidget || !minWidget || !maxWidget) {
    return;
  }

  const syncRange = () => {
    let lower = Math.trunc(Number(minWidget.value) || 0);
    let upper = Math.trunc(Number(maxWidget.value) || 0);
    if (lower > upper) {
      [lower, upper] = [upper, lower];
    }

    selectWidget.options = selectWidget.options || {};
    selectWidget.options.min = lower;
    selectWidget.options.max = upper;

    let current = Math.trunc(Number(selectWidget.value) || lower);
    current = Math.min(Math.max(current, lower), upper);
    selectWidget.value = current;
    return current;
  };

  const wrapWidget = (widget) => {
    if (!widget || widget._layer13RangeWrapped) {
      return;
    }
    widget._layer13RangeWrapped = true;
    const originalCallback = widget.callback;
    widget.callback = function (value, ...rest) {
      const result = typeof originalCallback === "function"
        ? originalCallback.call(this, value, ...rest)
        : value;
      syncRange();
      return result;
    };
  };

  wrapWidget(selectWidget);
  wrapWidget(minWidget);
  wrapWidget(maxWidget);
  syncRange();
}

function installCollapsedIntegerWidgetOnly(node, config) {
  if (!config.syncIntegerRange || node._layer13CollapsedWidgetOnlyInstalled) {
    return;
  }

  node._layer13CollapsedWidgetOnlyInstalled = true;

  const originalComputeSize = node.computeSize;
  if (typeof originalComputeSize === "function") {
    node.computeSize = function (...args) {
      if (!this.flags?.collapsed || !Array.isArray(this.widgets)) {
        return originalComputeSize.apply(this, args);
      }

      const originalWidgets = this.widgets;
      const valueWidget = getFirstExistingWidget(this, ["值", "选择"]);
      if (!valueWidget) {
        return originalComputeSize.apply(this, args);
      }

      this.widgets = [valueWidget];
      try {
        return originalComputeSize.apply(this, args);
      } finally {
        this.widgets = originalWidgets;
      }
    };
  }
}

function refreshNode(node, config, forcedType = null) {
  if (config.usesDynamicInputs) {
    pruneExtraEmptyInputs(node, config);
    ensureTrailingInput(node, config);
    syncTypes(node, config, forcedType);
    syncDynamicOutputs(node, config);
    updateSelectBounds(node, config);
    installSelectSnapper(node, config);
  }

  installIntegerRangeSync(node, config);
  installCollapsedIntegerWidgetOnly(node, config);
}

app.registerExtension({
  name: "Layer13.AnyIndexSwitch",
  setup() {
    if (
      typeof LGraphCanvas === "undefined"
      || !LGraphCanvas?.prototype
      || LGraphCanvas.prototype._layer13IntegerDrawWrapped
    ) {
      return;
    }

    const originalDrawNodeWidgets = LGraphCanvas.prototype.drawNodeWidgets;
    if (typeof originalDrawNodeWidgets !== "function") {
      return;
    }

    LGraphCanvas.prototype._layer13IntegerDrawWrapped = true;
    LGraphCanvas.prototype.drawNodeWidgets = function (node, ...args) {
      if (
        node?.comfyClass !== "Layer13InputCountControl"
        || !node.flags?.collapsed
        || !Array.isArray(node.widgets)
      ) {
        return originalDrawNodeWidgets.call(this, node, ...args);
      }

      const originalWidgets = node.widgets;
      const valueWidget = getFirstExistingWidget(node, ["值", "选择"]);
      if (!valueWidget) {
        return originalDrawNodeWidgets.call(this, node, ...args);
      }

      node.widgets = [valueWidget];
      try {
        return originalDrawNodeWidgets.call(this, node, ...args);
      } finally {
        node.widgets = originalWidgets;
      }
    };
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const config = NODE_CONFIGS[nodeData.name];
    if (!config) {
      return;
    }

    const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      refreshNode(this, config);
      return result;
    };

    const originalOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo) {
      originalOnConnectionsChange?.apply(this, arguments);

      const stackTrace = new Error().stack || "";
      if (stackTrace.includes("convertToSubgraph") || stackTrace.includes("Subgraph.configure")) {
        return;
      }

      if (stackTrace.includes("loadGraphData") || stackTrace.includes("pasteFromClipboard")) {
        refreshNode(this, config);
        return;
      }

      if (type === 2) {
        return;
      }

      const input = this.inputs?.[index];
      if (!input || config.fixedInputs.has(input.name) || parseInputIndex(input.name) == null) {
        return;
      }

      let originType = null;
      if (connected && linkInfo) {
        const originNode = app.graph.getNodeById(linkInfo.origin_id);
        originType = originNode?.outputs?.[linkInfo.origin_slot]?.type || null;
        if (originType === "*") {
          originType = null;
        }
      }

      refreshNode(this, config, originType);
    };
  },
  nodeCreated(node) {
    const config = getConfig(node);
    if (!config) {
      return;
    }
    refreshNode(node, config);
  },
});
