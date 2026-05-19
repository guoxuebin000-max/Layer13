import { app } from "../../scripts/app.js";

const L13_NODE_TYPES = new Set([
    "Layer13GuidedTiledKSampler8K",
    "Layer13GuidedTiledKSamplerAdvanced8K",
    "Layer13RedrawSettings",
    "Layer13ImageColorMatch",
    "Layer13ContextMaskedRedraw8K",
    "Layer13ContextMaskedRedrawAdvanced8K",
]);

const HEADER_COLOR = "#111111";
const BODY_COLOR = "#000000";

app.registerExtension({
    name: "Layer13.DefaultBlackNodes",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!L13_NODE_TYPES.has(nodeData.name)) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            this.color = HEADER_COLOR;
            this.bgcolor = BODY_COLOR;
            return result;
        };
    },
});
