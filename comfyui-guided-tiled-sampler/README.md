# Guided Tiled KSampler 8K

ComfyUI custom node for high-resolution tiled sampling from a low-resolution composition latent.

This is not an image upscaler. The node creates the final target-size latent first, then performs every denoise step by tiling the latent, blending overlapping denoise predictions, and updating the same full latent over the sampler timeline.

## Install

Copy this folder into:

```text
ComfyUI/custom_nodes/comfyui-guided-tiled-sampler
```

Then restart ComfyUI.

## Nodes

`sampling/guided_tiled -> Guided Tiled KSampler 8K (Layer13)`

`sampling/guided_tiled -> Guided Tiled KSampler Advanced 8K (Layer13)`

`sampling/l13_redraw -> L13 参考重绘放大`

`sampling/l13_redraw -> L13 参考重绘放大（高级）`

`sampling/l13_redraw -> L13 参考重绘放大参数`

## L13 Reference Redraw Workflow

This is the safer path for人物主体图. It does not directly upscale low-resolution latent. It uses the first-pass image as an anchor:

1. Generate a complete low-resolution reference image.
2. Connect that image to `参考图像` and connect the matching `VAE`.
3. Choose `目标规格 = 4K` or `8K`.
4. The node scales the image to target size, VAE-encodes it into a high-resolution latent canvas, then redraws center tiles with context using `noise_mask`.
5. Decode the output latent with tiled VAE decode.

Recommended人物 8K defaults:

- `目标规格`: `4K` by default; switch to `8K` after the 4K settings are stable.
- `总步数`: `10`
- `重绘强度`: `0.12 - 0.22`
- `CFG引导`: `1.0` by default for reference-anchored redraw; raise only if the prompt is too weak.
- `采样器` / `调度器`: `euler` / `ddim_uniform`
- `分块宽度` / `分块高度`: `1024 - 1536`
- `重叠像素`: `192 - 256`
- `上下文像素`: `384 - 512`
- `采样缓冲像素`: `96 - 160`
- `重绘轮数`: `1`
- `预览频率`: `每个分块`
- `细节扰动`: `0.00 - 0.04` for人物, `0.03 - 0.08` for背景/材质
- `递进放大模式`: `关闭` by default; use `快速2倍`, `平衡1024阶梯`, or `稳定1.5倍` only when progressive upscale is needed.
- `递进强度衰减`: `0.85`
- `色彩稳定强度`: `0`
- `参考保留强度`: `0.04 - 0.12` for人物, lower it if the redraw becomes too conservative.
- `参数预设`: `人物稳定` for人物, `背景增强` for scenery/materials.
- `人物安全模式`: `启用`
- `主体重绘上限`: `0.12 - 0.16` with a subject mask.
- `接缝修复`: `禁用` by default; use `启用` with `接缝修复强度 = 0.04 - 0.08` if seams remain.

The progress bar advances by sampler step across all progressive stages, passes, and tiles. With `预览频率 = 每个分块`, ComfyUI receives the same KSampler-style latent preview from each sampler callback step for the current context tile.

The main `L13 参考重绘放大` nodes keep only the workflow-level controls visible and mark size/tile/detail controls as advanced. If you want a cleaner graph or want to share the same advanced settings between normal and advanced nodes, add `L13 参考重绘放大参数` and connect its `高级参数` output to the main node. Connected `高级参数` overrides the folded advanced controls on the main node, including custom width/height, redraw denoise, tile size, overlap, context, detail perturbation, sample halo, blend mode, preview frequency, reference retention, subject denoise cap, background multiplier, and seam repair settings.

`细节扰动` adds a tiny high-frequency latent perturbation only inside the center write mask. It uses one global noise field cropped per tile, and subject protection masks also reduce this perturbation.

`采样缓冲像素` separates the sampled area from the written area. Each tile now samples `write core + sample halo` inside the larger context crop, but only writes the original center tile back to the full latent canvas. This reduces hard mask edges while keeping context read-only.

`主体保护遮罩` now does two jobs. It still softens the noise mask inside protected areas, and it also enables adaptive tile denoise: subject-heavy tiles are capped by `主体重绘上限`, while low-subject tiles may use `背景重绘倍率` to add more background texture. `主体重绘上限` is not a second global denoise value; it only applies when a subject mask is connected and the current tile contains enough subject pixels. `人物安全模式` applies extra runtime caps for人物 presets or when a subject mask is connected.

`接缝修复` runs only on the final stage. It creates a seam mask along tile boundaries and performs one low-denoise masked redraw pass there, then writes only the seam areas back with the normal feather accumulation. Keep it off unless visible seams remain.

`递进放大模式` creates intermediate canvases before the final size. For example, a 1024px source targeting 4K with `平衡1024阶梯` runs approximately `2048 -> 3072 -> 4096`. Between stages, the node decodes the current latent to pixels, scales to the next stage, VAE-encodes again, then performs the same context-masked tile redraw. This avoids relying on one large latent interpolation jump.

`色彩稳定强度` is now a compatibility-only control for old workflows. It no longer matches latent mean/contrast because that can wash out some models. `参考保留强度` mixes a small amount of the reference latent back into the sampled tile and is the safer way to preserve pre-upscale texture.

## L13 Advanced Context Redraw

`L13 参考重绘放大（高级）` uses the same reference image, context crop, center `noise_mask`, progressive stages, and feather writeback path as the normal node, but exposes KSampler Advanced controls:

- `加噪`
- `噪声种子`
- `起始步`
- `结束步`
- `保留剩余噪声`

For a segmented run:

1. First segment: `加噪 = 启用`, `起始步 = 0`, `结束步 = 3`, `保留剩余噪声 = 启用`.
2. Next segment: same `总步数` / `采样器` / `调度器` / prompt, `加噪 = 禁用`, `起始步 = 3`, `结束步 = 总步数`, `保留剩余噪声 = 禁用`.

For most人物图, the non-advanced local redraw node is still the simpler default. Use the advanced version only when you really need KSampler Advanced step splitting.

## Typical Workflow

1. Generate a low-resolution composition latent with a normal KSampler, for example `1024x1024` or `1536x1536`.
2. Connect that latent into `构图潜空间`.
3. Set `目标规格` to `4K` or `8K`, or leave it as `自定义`.
4. If using `自定义`, set `目标宽度` and `目标高度`. When both are the same, for example `8192 / 8192`, the value is treated as the long edge and the original latent aspect ratio is preserved.
5. Set `重绘强度` around `0.55 - 0.75`.
6. Decode the output latent with your VAE.

## Practical Defaults

- `分块宽度` / `分块高度`: `768 - 1536` pixels
- `重叠像素`: `128 - 256` pixels
- `融合方式`: `余弦`
- `重绘强度`: `0.65`

Smaller tiles reduce VRAM use but increase runtime and can reduce global consistency. Larger overlap reduces seams but costs more time.

## Chinese Inputs

All visible input names are Chinese, and every input includes a Chinese tooltip in ComfyUI:

- Basic inputs: `模型`, `正向条件`, `负向条件`, `构图潜空间`, `随机种子`, `总步数`, `CFG引导`, `采样器`, `调度器`
- Target inputs: `目标规格`, `目标宽度`, `目标高度`, `重绘强度`
- Tiling inputs: `分块宽度`, `分块高度`, `重叠像素`, `融合方式`, `构图缩放算法`, `最大分块数`
- Noise inputs: `加噪`, `噪声种子`, `保留剩余噪声`

The dropdown values `加噪`, `保留剩余噪声`, and `融合方式` are also shown in Chinese.

## Target Size Logic

- `目标规格 = 4K`: the input latent's long edge becomes `4096`, and the short edge is calculated from the original latent ratio.
- `目标规格 = 8K`: the input latent's long edge becomes `8192`, and the short edge is calculated from the original latent ratio.
- `目标规格 = 自定义`: `目标宽度` / `目标高度` are used directly, except when they are equal. If they are equal, that value is treated as the long edge and the original latent ratio is preserved.

Examples:

- Input latent ratio `16:9`, `目标规格 = 8K` -> `8192 x 4608`
- Input latent ratio `9:16`, `目标规格 = 8K` -> `4608 x 8192`
- Input latent ratio `16:9`, `目标宽度 = 8192`, `目标高度 = 8192` -> `8192 x 4608`
- Input latent ratio `1:1`, `目标宽度 = 8192`, `目标高度 = 8192` -> `8192 x 8192`

## Advanced Step-Split Workflow

Use this when you want the first pass to run only the composition portion, then continue the same sampler timeline at 8K.

1. Low-resolution `KSampler Advanced`:
   - `steps`: final total steps, for example `30`
   - `start_at_step`: `0`
   - `end_at_step`: `3`
   - `add_noise`: `enable`
   - `return_with_leftover_noise`: `enable`
2. Connect its output latent to `Guided Tiled KSampler Advanced 8K (Layer13)` `输入潜空间`.
3. Guided tiled advanced node:
   - `总步数`: same total steps, for example `30`
   - `起始步`: `3`
   - `结束步`: `30`
   - `加噪`: `禁用`
   - `保留剩余噪声`: `禁用`
   - `重绘强度`: `1.0`
   - `目标规格`: `8K`, or use `目标宽度 = 8192` and `目标高度 = 8192` to keep the original latent ratio

This is closer to native `KSampler Advanced` segmented sampling than using a completed low-resolution image as img2img guidance.

## Notes

- The node supports prompt conditioning and latent composition guidance.
- Area and mask conditioning are shifted into each tile.
- ControlNet may work in simple cases, but tiled ControlNet hints can be spatially imperfect. Prefer using the low-resolution composition latent as the main global guide.
