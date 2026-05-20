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

`sampling/l13_redraw -> L13 参考重绘放大参数（高级）`

`sampling/l13_redraw -> L13 区域提示词`

`sampling/l13_redraw -> L13 视觉区域规划提示词`

`sampling/l13_redraw -> L13 视觉区域JSON转提示词`

`image/l13 -> L13 图像颜色匹配`

## L13 Reference Redraw Workflow

This is the safer path for人物主体图. It does not directly upscale low-resolution latent. It uses the first-pass image as an anchor:

1. Generate a complete low-resolution reference image.
2. Connect that image to `参考图像` and connect the matching `VAE`.
3. Choose `目标规格 = 4K` or `8K`.
4. The node scales the image to target size, VAE-encodes it into a high-resolution latent canvas, then redraws center tiles with context using `noise_mask`.
5. The node outputs an internally decoded `图像`. Internal VAE encode/decode uses 1024px tiles with 128px overlap.

Recommended人物 8K defaults:

- `目标规格`: `4K` by default; switch to `8K` after the 4K settings are stable.
- `步数`: `10`
- `降噪`: `0.22` by default; raise toward `0.35 - 0.45` only when you need stronger redraw/detail.
- `CFG引导`: `1.0` by default for reference-anchored redraw; raise only if the prompt is too weak.
- `采样器` / `调度器`: `euler` / `ddim_uniform`
- `分块宽度` / `分块高度`: `1024` by default; raise to `1280 - 1536` for stronger consistency.
- `重叠像素`: `128` by default; raise to `192 - 256` if seams appear.
- `上下文像素`: `256` by default; raise to `384 - 512` if local redraw loses context.
- `采样缓冲像素`: `64` by default; raise to `96 - 160` for smoother masked edges.
- `重绘轮数`: `1`
- `细节扰动`: `0.008` by default for the normal node, `0.006` for the advanced node. Lower it to `0` for maximum smoothness, or raise carefully toward `0.015 - 0.03` for stronger material texture.
- `细节噪声模式`: `多尺度` is the default because it preserves material variation better than plain high-frequency grain; `参考纹理` boosts high-frequency texture already present in the reference latent; `像素颗粒` is harsher and should use very low strength.
- `细节噪声位置`: `采样前` lets the sampler absorb the texture naturally; `写回前` directly adds pixel-like grain before tile blending; `两者` is stronger and should be used cautiously.
- `参考噪声强度`: `0.08 - 0.25` mixes the reference latent's normalized structure into the global sampler noise, similar to KleinTiled's `latent_blend` noise guidance. Use it to test stronger structure/light retention without raising denoise.
- `递进放大模式`: `开启` by default. It advances by about `2048px` on the short edge per stage while preserving aspect ratio; set it to `关闭` to jump directly to the target size.
- `递进强度衰减`: `1.0` by default, so progressive stages do not automatically lose denoise/detail.
- `色彩稳定强度`: `0`
- `参考保留强度`: `0.03` by default. It now only pulls low-frequency reference structure back into the sampled tile, so newly sampled high-frequency texture is not blended away.
- `主体重绘上限`: `0.12 - 0.16` with a subject mask.
- `接缝修复`: `禁用` by default; use `启用` with `接缝修复强度 = 0.04 - 0.08` if seams remain.

Preview is handled by ComfyUI's built-in sampler preview system. There is no node-level preview mode; use the normal ComfyUI preview/progress settings to enable or disable sampler previews.

The normal `L13 参考重绘放大` node keeps `降噪` visible because it is the same denoise concept as KSampler img2img. Size/tile/detail controls are removed from the main node UI and use built-in defaults unless you connect `L13 参考重绘放大参数` to the `高级参数` input. Connected `高级参数` overrides custom width/height, tile size, overlap, context, detail perturbation, sample halo, blend mode, reference retention, subject denoise cap, background multiplier, seam repair settings, detail residual scale, and output mask size.

The `图像` output is decoded inside the node with tiled VAE decode and then color-matched against the reference image with low-frequency color transfer at strength 1.0.

The normal and advanced redraw nodes now also expose internal self-use debug outputs after the first `图像` output:

- `最终潜空间`: the final high-resolution latent canvas after tiled redraw.
- `参考潜空间`: the final-stage base latent encoded from the scaled reference image.
- `细节差异潜空间`: a high-frequency residual layer computed from `最终潜空间` and `参考潜空间`, useful for inspecting or later controlling AI-added detail.
- `接缝遮罩`: the final-stage tile seam mask, generated even when seam repair is disabled; it is black when there is no seam.
- `主体保护遮罩`: the final-stage subject protection mask after scaling; it is black when no subject mask is connected.

These latent outputs are for L13 follow-up processing and debugging. They are not recommended for direct full-image KSampler use, because 4K/8K latent canvases can be too large for a normal untiled sampler.

`细节差异尺度` controls the lowpass kernel used to extract `细节差异潜空间`. `输出遮罩尺寸` controls whether `接缝遮罩` and `主体保护遮罩` are returned at latent resolution or final image resolution; `latent尺寸` is the default and uses less memory.

`并行分块数` batches multiple same-size tiles into one sampler call so the GPU can be used more fully. Keep it at `1` for the safest behavior. Try `2` or `4` when VRAM is underused. It increases VRAM usage, and the node automatically falls back to single-tile sampling when regional prompts, spatial masks/areas, or ControlNet-style conditioning are present.

`细节扰动` adds controlled latent texture only inside the center write mask. It uses one global noise field cropped per tile, so adjacent tiles share the same noise distribution. `细节噪声模式` controls the texture shape, and `细节噪声位置` controls whether the noise is added before sampling, directly before writeback, or both. Subject protection masks also reduce this perturbation. Progressive mode also applies a very light pixel-space sharpening between stages to counter VAE decode/encode low-pass loss.

This is deliberately simpler than LG Noise Injection's model-level CFG feature injection. LG-style injection changes the model's CFG output over part of the sampler timeline; the L13 detail noise is local to each masked tile, easier to predict, and less likely to change the subject. For “more pixels / less plastic texture”, start with `细节扰动 = 0.008`, `细节噪声模式 = 多尺度`, `细节噪声位置 = 采样前`. If the image is still too smooth, try `写回前` at `0.004 - 0.01`.

`参考噪声强度` is a separate sampler-noise guide. It blends the global Gaussian noise with a normalized version of the high-resolution reference latent before each tile is sampled, then restores the original noise variance. This is closer to the KleinTiled idea of mixing `latent_blend` into the noise. Start around `0.12 - 0.18`; high values can become too conservative or imprint low-resolution structure.

`采样缓冲像素` separates the sampled area from the written area. Each tile now samples `write core + sample halo` inside the larger context crop, but only writes the original center tile back to the full latent canvas. This reduces hard mask edges while keeping context read-only.

`主体保护遮罩` now does two jobs. It still softens the noise mask inside protected areas, and it also enables adaptive tile denoise: subject-heavy tiles are capped by `主体重绘上限`, while low-subject tiles may use `背景重绘倍率` to add more background texture. `主体重绘上限` is not a second global denoise value; it only applies when a subject mask is connected and the current tile contains enough subject pixels.

`L13 区域提示词` adds mask-based regional prompts without requiring the user to know tile boundaries. Connect `CLIP`, a full-image `遮罩`, and regional positive/negative text. Chain multiple region nodes through `已有区域提示词`, then connect the final `区域提示词` output to the normal or advanced redraw node. During progressive upscale, the redraw node scales each region mask to the current stage latent size and crops it to the current context tile before sampling. `遮罩羽化` is converted from pixels to latent units per stage. `区域范围 = 遮罩范围` keeps the conditioning area tight around the local mask; `整块上下文` keeps the whole context active and only uses the mask for blending strength.

For automatic regional prompts, use `L13 视觉区域规划提示词` with any vision-language node that can read an image and return text. Feed its `视觉模型提示词` to the VLM prompt input and the first-pass image to the VLM image input. Connect the VLM text output to `L13 视觉区域JSON转提示词 -> 视觉模型JSON`, connect the same first-pass image to `参考图像`, then connect `区域提示词` to the redraw node. The expected VLM output is JSON with normalized `bbox` values in `x0,y0,x1,y1` order plus `positive`, `negative`, `strength`, and `feather` fields. The converter also returns a combined `区域遮罩` and `解析摘要` for checking what it found.

`接缝修复` runs only on the final stage. It creates a seam mask along tile boundaries and performs one low-denoise masked redraw pass there, then writes only the seam areas back with the normal feather accumulation. Keep it off unless visible seams remain.

`递进放大模式` creates intermediate canvases before the final size. It now only has `开启` and `关闭`. When enabled, it steps by about `2048px` on the short edge while keeping the reference aspect ratio. Between stages, the node decodes the current latent to pixels, scales to the next stage, VAE-encodes again, then performs the same context-masked tile redraw. This avoids relying on one large latent interpolation jump.

`色彩稳定强度` is now a compatibility-only control for old workflows. It no longer matches latent mean/contrast because that can wash out some models. `参考保留强度` now blends only low-frequency reference structure back into the sampled tile; this keeps composition anchoring without erasing newly generated texture.

`L13 图像颜色匹配` remains available as a standalone utility, but the normal and advanced redraw nodes now include the same low-frequency color transfer on their `图像` output. Use the standalone node only when you want to compare or override the internal matched image.

## L13 Advanced Context Redraw

`L13 参考重绘放大（高级）` uses the same context crop, center `noise_mask`, and feather writeback path as the normal node, but follows KSampler Advanced style controls instead of img2img denoise controls:

- `加噪`
- `噪声种子`
- `起始步`
- `结束步`
- `保留剩余噪声`

Unlike the normal node, the advanced node has no `降噪` input. It keeps the sampler denoise timeline at `1.0`, matching KSampler Advanced behavior. `起始步` / `结束步` define the actual sigma segment, so the segment length is the redraw strength.

Reference input:

- The advanced node uses `参考图像` again and no longer requires `输入潜空间`.
- The reference image is resized in pixel space and VAE-encoded into the high-resolution latent canvas.
- The advanced redraw node no longer directly upscales latent tensors. This avoids the colored-noise artifacts caused by resizing noisy latent data.
- During tile writeback, advanced mode applies `结构锁定` against that high-resolution internal reference latent. It locks low-frequency structure and now also mixes back a small amount of the reference latent based on the same strength, which helps suppress duplicated bodies when the sampler tries to reinterpret a tile.
- Advanced mode also applies internal latent color matching at fixed strength `1.0` against the same high-resolution internal reference latent before each tile is written back. This matches the color/contrast statistics of the reference result instead of letting each tile drift.

The advanced node now exposes `递进放大模式`. With `L13 参考重绘放大参数（高级） -> 递进步数模式 = 起始步递进`, every size stage keeps the same `结束步`, while later stages move `起始步` forward. For example, `起始步=4`, `结束步=12`: two size stages run `4-12`, `8-12`; three size stages run `4-12`, `7-12`, `10-12`. `随尺寸递进` keeps the old split behavior, dividing `起始步 -> 结束步` into continuous stage windows such as `0-3`, `3-6`, `6-9`, `9-12`. `固定起止步` keeps every size stage on the same KSampler Advanced segment.

`L13 参考重绘放大参数（高级）` is a separate settings node for the advanced version. It contains target size, tile size, overlap, context, sample halo, blend mode, image scaling, tile order, max tile count, detail noise controls, structure lock controls, `参考保留强度`, detail residual scale, and output mask size. It intentionally does not contain `降噪`, `重绘强度`, adaptive subject denoise, or seam repair controls.

For full-step advanced redraws that still duplicate the subject, raise `结构锁定强度` from the default `0.55` toward `0.70`. If texture becomes too conservative, lower it or raise `结构锁定尺度` from `64` to `96-128` so only larger composition shapes are locked.

For a segmented run:

Connect the completed first-pass image to `参考图像`. The node encodes that image as the high-resolution reference canvas, then performs the selected KSampler Advanced segment with context-masked tile redraw.

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

- Basic inputs: `模型`, `正向条件`, `负向条件`, `构图潜空间`, `随机种子`, `步数`, `CFG引导`, `采样器`, `调度器`
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
   - `步数`: same total steps, for example `30`
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
