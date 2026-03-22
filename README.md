# layer13-comfyui-nodes

一组自定义 `ComfyUI` 节点，主要覆盖：

- 多图拖入与外部 `for` 循环取图
- 图像批次取图与循环索引
- 文本拼接与逐行前缀注入
- 随机加载 N 张图片
- 批次转网格、网格拆批次
- 长短边缩放
- VHS 输出路径提取与首中尾帧提取
- 直方图限制

## 节点列表

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

## 安装

把仓库克隆到 `ComfyUI/custom_nodes/` 目录，然后重启 `ComfyUI`：

```bash
cd ComfyUI/custom_nodes
git clone <your-repo-url> layer13
```

如果你不想把目录名改成 `layer13`，也可以直接保持仓库名不变，节点本身不会受影响。

## 前端文件

当前前端扩展文件：

- `web/layer13_multi_image_input.js`

这个文件用于多图拖入、缩略图展示、追加上传、删除、拖拽排序和节点尺寸自适应。

## 开发说明

- 入口文件：`__init__.py`
- 前端目录：`web/`
- 不要把 `__pycache__`、`.bak_*`、测试工作流一起提交

## 当前包含的源码文件

- `__init__.py`
- `layer13_for_loop_index.py`
- `layer13_grid_from_batch.py`
- `layer13_grid_split_to_batch.py`
- `layer13_histogram_limit.py`
- `layer13_multi_image_input.py`
- `layer13_random_loader_n.py`
- `layer13_scale_by_long_short_edge.py`
- `layer13_text_join_n.py`
- `layer13_video_from_vhs.py`
