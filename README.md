# DINOv2 固废图像识别模型

这是一般工业固废图像识别模型的可复现留档与本地快速测试仓库。当前版本为 `solid_waste_dinov2_proto_v1`，识别煤矸石、矿渣、钢渣、石膏四类物料；当置信度、原型相似度或类别间隔不足时输出“其他”，提示人工复核。

本仓库包含模型源码、配置、已训练原型库、DINOv2 本地运行时和一个可上传图片或选择文件夹进行快速识别的 Tkinter 界面。没有上传原始训练图片；因此重新训练时需要按下面的数据规范自行准备图片。

## 1. 仓库结构

```text
solid_waste_model/       Python 包、推理代码和配置
artifacts/<version>/     版本化原型库、元数据、评估报告
runtime/torch_hub/       DINOv2 源码和预训练权重
data/                    训练数据入口和 manifest（不提交图片）
tests/                   单元测试
docs/                    输入输出和交付说明
scripts/                 交付包生成脚本
run_gui.bat              图形化快速识别
run_predict.bat          命令行批量识别
run_train.bat            训练和评估
```

## 2. 安装环境

推荐 Python 3.12。创建虚拟环境后安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-runtime-dinov2.txt
```

训练还需要：

```powershell
python -m pip install -r requirements-model.txt
```

有 NVIDIA GPU 时，需安装与驱动匹配的 CUDA 版 PyTorch；没有 GPU 可以将命令中的 `--device cuda` 改为 `--device cpu`，但速度会明显变慢。

## 3. 快速识别

双击 `run_gui.bat`，在界面中点击“选择图片”或“选择文件夹”，再点击“运行识别”。结果会保存到 `output/interactive_predict_results.json`。界面会显示最终类别、置信度、拒识状态、相似度、类别间隔和四类分数。启动脚本会自动检查仓库内虚拟环境、上级项目 `.venv312` 和系统 Python；没有可用依赖时会停留在窗口中显示安装提示。需要查看完整 Python 报错时可运行 `run_gui_debug.bat`。

命令行方式：

```powershell
.\.venv\Scripts\python.exe -m solid_waste_model.dinov2_proto predict path\to\images `
  --recursive `
  --prototype artifacts\solid_waste_dinov2_proto_v1\solid_waste_dinov2_proto_v1.prototypes.npz `
  --meta artifacts\solid_waste_dinov2_proto_v1\solid_waste_dinov2_proto_v1.meta.json `
  --local-repository runtime\torch_hub\facebookresearch_dinov2_main `
  --cache-dir runtime\torch_hub `
  --device cuda `
  --enable-rejection `
  --save-json output\prediction_results.json
```

首次运行会加载本地 `runtime/torch_hub` 中的 DINOv2 源码和权重，不应删除该目录。也可以在后端中使用 `solid_waste_model.dinov2_inference.WasteClassifier`；模型实例应在服务启动时创建一次，而不是每次请求重新加载权重。

## 4. 图片和数据规范

推理输入支持 `.jpg`、`.jpeg`、`.png`、`.bmp`、`.webp`、`.tif` 和 `.tiff`，可以是单张图片或文件夹。图片应尽量满足：主体物料清晰可见、避免完全黑屏或空画面、保留现场真实背景和拍摄距离，不要将同一连续视频帧同时当作独立测试样本。

重新训练时，原始图片按“场景/物料”放置：

```text
data/raw/
├─ 实验室/
│  ├─ 煤矸石/
│  ├─ 矿渣/
│  ├─ 钢渣/
│  └─ 石膏/
└─ 料棚/
   ├─ 煤矸石/
   ├─ 矿渣/
   └─ 钢渣/
```

目录名必须与 `solid_waste_model/configs/label_schema.json` 一致。每张图片在 manifest 中应有：`image_path`、`material_type`、`scene`、`split`、`is_known_class`、`sample_group`、`augmentation_source` 和 `usage_role`。训练、验证、测试图片必须按拍摄批次、摄像头或连续帧分组，不能让同一采集组跨 split。未知物料、空场景、严重模糊图和容易混淆的相似物料应单独作为拒识验证数据，不要冒充四类正样本。

当前属性字段 `particle_size`、`shape`、`color` 尚未有可靠监督，默认只能填 `unknown`；它们不能被描述为模型已经实现的真实属性识别。

## 5. 训练、评估和模型输出

先生成 manifest，再运行：

```powershell
python -m solid_waste_model.manifest
python -m solid_waste_model.inspect_dataset
python -m pytest -q
python -m solid_waste_model.dinov2_proto train `
  --model-version solid_waste_dinov2_proto_v1 `
  --manifest data\manifests\manifest.csv `
  --output-dir artifacts\solid_waste_dinov2_proto_v1 `
  --local-repository runtime\torch_hub\facebookresearch_dinov2_main `
  --cache-dir runtime\torch_hub `
  --device cuda --batch-size 8 --image-size 448 `
  --prototypes-per-class 3 --crop-ratio 0.78
```

DINOv2 这条路线不是 epoch 式端到端训练：它冻结 DINOv2，提取训练/验证/测试特征，基于训练特征构建原型，使用验证集校准拒识阈值，再用测试集评估。每次更换数据或模型版本，都必须同时生成以下同一版本产物：

```text
artifacts/<model_version>/
├─ <model_version>.prototypes.npz
├─ <model_version>.meta.json
├─ <model_version>.report.json
├─ <model_version>.validation_errors.csv
└─ <model_version>.test_errors.csv
```

`meta.json`、`prototypes.npz` 和 DINOv2 源码/权重必须来自同一次训练或同一交付版本，不能单独替换其中一个文件。`report.json` 保存数据划分、参数、准确率和拒识统计。当前版本的测试集整体准确率为 99.15%，料棚测试集准确率为 98.67%；该结果不等同于经过时间、摄像头和批次独立验证的现场准确率。

## 6. 输出格式

命令行输出 JSON 中每张图片包含 `image_path`、`result_label`、`confidence`、`similarity`、`margin`、`class_probabilities`、`crop_weights` 和 `rejected`。业务适配器 `safe_predict_payload()` 保留旧接口的顶层字段：

```json
{
  "waste_type": "矿渣",
  "confidence": 0.81,
  "features": {"particle_size": "unknown", "shape": "unknown", "color": "unknown", "source": "config_profile"},
  "model_result": {"rejected": false, "similarity_score": 0.91, "margin": 0.42},
  "model_error": null
}
```

`model_error` 非空时不得把结果写入业务数据库。`waste_type` 为“其他”或 `rejected=true` 时应转人工复核。当前版本不导出 ONNX，后端必须加载 DINOv2 本地源码/权重、原型库和元数据。

## 7. 交付和留档

运行以下命令生成完整离线交付包：

```powershell
python scripts\package_dinov2_delivery.py --zip
```

交付包输出到 `output/delivery/solid_waste_dinov2_proto_v1/`，包含模型产物、DINOv2 本地源码和权重、运行代码、依赖、启动脚本、输入输出说明及 `SHA256SUMS.txt`。建议源码提交到 GitHub，完整 ZIP 作为 GitHub Release 附件；不要提交原始数据、虚拟环境或缓存目录。上传后应执行：

```powershell
python -m pytest -q
Get-FileHash output\delivery\solid_waste_dinov2_proto_v1.zip -Algorithm SHA256
```

## 8. 当前限制

- 模型是整图分类，不是检测或分割模型。
- “其他”不是独立训练类别；当前未知物料负样本仍不足。
- 当前属性画像来自配置文件，不是粒度、形状、颜色的真实预测。
- 正式上线前应使用新采集的现场图片，按时间、摄像头和批次建立独立验收集。
