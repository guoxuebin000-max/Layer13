# Layer13 Crop Info Tools

这组节点解决的是一个固定需求：

1. 只对第一张图做人脸检测或做人脸遮罩
2. 从第一张图生成一份稳定的 `crop_info`
3. 全部图片共用这份 `crop_info` 做同范围裁剪
4. 后续处理结束后，不走遮罩合成，直接按 `crop_info` 恢复到原图

## 节点说明

### 1. `Layer13批次取首图`
输入：`IMAGE` 批次
输出：首张图、批次数量

用途：从一组图片里只拿第一张去做人脸检测。

### 2. `Layer13从框生成裁剪信息`
输入：
- `image`
- `x/y/width/height`
- 四向 padding
- `aspect_ratio`

输出：
- `cropped_image`
- `crop_info`
- `crop_json`
- `crop_x/y/w/h`

用途：
- 手动演示版工作流直接用它
- 也可以把任何检测节点输出的脸框转换为 `crop_info`

### 3. `Layer13从遮罩生成裁剪信息`
输入：
- `image`
- `mask`
- `mask_threshold`
- 四向 padding
- `aspect_ratio`

输出同上。

用途：
- 最适合接人脸分割、人脸检测遮罩、SAM 人脸区域遮罩
- 这是你后面正式工作流最建议接的版本

### 4. `Layer13按裁剪信息批量裁剪`
输入：
- `images`
- `crop_info`

输出：
- `cropped_images`
- `crop_x/y/w/h`

用途：把第一张图算出来的裁剪范围复用到整批图片。

### 5. `Layer13从JSON恢复裁剪信息`
输入：
- `crop_json`

输出：
- `crop_info`
- `crop_json`
- `crop_x/y/w/h`

用途：
- 把第一次运行得到的 `crop_json` 固定下来
- 后续 rerun 时不再重新走遮罩检测
- 直接把保存好的裁剪信息还原成 `crop_info` 给后面的裁剪/恢复节点使用

### 6. `Layer13保存裁剪信息到文件`
输入：
- `crop_json`
- `filename`
- `subfolder`
- `overwrite`

输出：
- `file_path`
- `file_name`
- `crop_json`

用途：
- 把第一次得到的 `crop_json` 直接保存到 ComfyUI `output` 目录下
- 后续不用手动复制粘贴

### 7. `Layer13从文件读取裁剪信息`
输入：
- `file_path`

输出：
- `crop_info`
- `crop_json`
- `file_path`
- `crop_x/y/w/h`

用途：
- 直接从上次保存的 `.json` 文件恢复 `crop_info`
- 后续裁剪和恢复都用同一份裁剪信息

### 8. `Layer13按裁剪信息恢复`
输入：
- `processed_images`
- `original_images`
- `crop_info`

输出：
- `restored_images`

用途：
- 不使用遮罩恢复
- 直接按裁剪框位置贴回原图
- 如果处理后的裁剪图尺寸变了，节点会先 resize 到裁剪框尺寸再恢复

## 推荐正式工作流

```text
整批图片
  -> Layer13批次取首图
  -> 人脸检测 / 人脸遮罩
  -> Layer13从遮罩生成裁剪信息

整批图片 + crop_info
  -> Layer13按裁剪信息批量裁剪
  -> 你的后处理节点
  -> Layer13按裁剪信息恢复
```

## 固定裁剪信息的推荐用法

第一次运行：

```text
首图 -> Layer13从遮罩生成裁剪信息
     -> Layer13保存裁剪信息到文件
```

后续运行：

```text
已保存的 crop_info.json
  -> Layer13从文件读取裁剪信息
  -> Layer13按裁剪信息批量裁剪
  -> 你的后处理节点
  -> Layer13按裁剪信息恢复
```

这样后面不管你 rerun 多少次，都不会因为重新检测遮罩而得到新的裁剪框，恢复时也不会和之前裁过的视频对不上。

## 演示工作流

示例工作流文件：

- `example_workflows/Layer13_首图裁剪信息_批量复用_恢复_演示.json`

这份演示工作流默认用“手动框”模拟第一张图的人脸框，优先保证：
- 可直接导入
- 不依赖第三方检测节点
- 能完整演示“生成 crop_info -> 批量裁剪 -> 按 crop_info 恢复”

你后面只需要把“Layer13从框生成裁剪信息”替换成“Layer13从遮罩生成裁剪信息”，再把遮罩接到你的人脸检测节点即可。

## 安装方式

把整个目录复制到 ComfyUI 的 `custom_nodes` 下面：

```bash
cp -R layer13_crop_info_tools /path/to/ComfyUI/custom_nodes/
```

然后重启 ComfyUI。
