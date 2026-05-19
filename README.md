# Layer13 ComfyUI Nodes

这个仓库汇总了 `Layer13` 相关的 `ComfyUI` 自定义节点，包含两部分：

1. 当前主包 `layer13`
2. 独立节点目录

## 当前主包 `layer13`

当前主包位于仓库根目录，主要包含：

- 多图拖入与外部 `for` 循环取图
- 图像批次取图与循环索引
- 文本拼接与逐行前缀注入
- 随机加载 N 张图片
- 批次转网格、网格拆批次
- 长短边缩放
- VHS 输出路径提取与首中尾帧提取
- 直方图限制

### 主包节点列表

- `Layer13多图导入`
- `Layer13拖入取图`
- `Layer13拖入加载图片`
- `Layer13按索引取图像列表`
- `Layer13循环索引`
- `Layer13按索引取批次图像`
- `Layer13联结N文本`
- `Layer13前缀注入每行`
- `Layer13随机加载N图`
- `Layer13网格拆分批次`
- `Layer13批量图自动网格`
- `Layer13长短边缩放`
- `Layer13提取VHS视频路径`
- `Layer13提取首中尾帧`
- `Layer13直方图限制`

## 历史独立目录

仓库中保留并同步这些独立目录：

- `comfyui-photorealism-enhancer`：Layer13 照片增强
- `comfyui-layer13-depth-of-field`：Layer13 真实景深（深度图）
- `comfyui-jpg-convert`：Layer13 保存 JPG
- `comfyui-guided-tiled-sampler`：L13 局部重绘 4K/8K 采样器
- `comfyui-layer13-any-index-switch`
- `comfyui-layer13-auto-crop-border`
- `comfyui-layer13-dataset-tools`
- `comfyui-layer13-force-black-level`
- `comfyui-layer13-force-levels-fixed`
- `comfyui-layer13-preview`
- `comfyui-layer13-random-loader`
- `comfyui-layer13-random-video-frame`
- `comfyui-layer13-video-upload`
- `layer13_crop_info_tools`

## 安装

克隆到 `ComfyUI/custom_nodes/` 后重启 `ComfyUI`：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/guoxuebin000-max/Layer13.git
```

如果你想让主包目录名保持为 `layer13`，可以手动改名或按自己的目录结构拆分。

## 前端文件

当前前端扩展文件：

- `web/layer13_multi_image_input.js`

这个文件负责：

- 多图拖入
- 缩略图展示
- 追加上传
- 单张删除
- 拖拽排序
- 节点尺寸自适应

## 开发说明

- 主包入口：`__init__.py`
- 前端目录：`web/`
- 不要提交：
  - `__pycache__/`
  - `*.bak_*`
  - 临时测试工作流

## 当前主包源码文件

- `__init__.py`
- `layer13_apimart.py`
- `layer13_audio_tools.py`
- `layer13_for_loop_index.py`
- `layer13_fullscreen_watermark.py`
- `layer13_grid_from_batch.py`
- `layer13_grid_split_to_batch.py`
- `layer13_grok_prompt.py`
- `layer13_histogram_limit.py`
- `layer13_image_batch_append.py`
- `layer13_image_compress.py`
- `layer13_load_image_default.py`
- `layer13_masked_color_match.py`
- `layer13_masked_exposure.py`
- `layer13_metadata_cleaner.py`
- `layer13_multi_image_input.py`
- `layer13_persistent_counter.py`
- `layer13_phone_camera.py`
- `layer13_random_loader_n.py`
- `layer13_real_pixel_texture.py`
- `layer13_scale_by_long_short_edge.py`
- `layer13_text_join_n.py`
- `layer13_text_loader.py`
- `layer13_video_from_vhs.py`
