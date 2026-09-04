# DINOv2 离线交付说明

完整交付包必须将以下内容视为一个不可拆分版本：模型原型库、模型元数据、评估报告、DINOv2 源码、DINOv2 ViT-S/14 预训练权重、推理代码、依赖文件和启动脚本。

在交付包根目录执行：

```powershell
python -m pip install -r requirements-runtime-dinov2.txt
python -m solid_waste_model.dinov2_proto predict <图片或文件夹> --recursive --prototype artifacts\solid_waste_dinov2_proto_v1.prototypes.npz --meta artifacts\solid_waste_dinov2_proto_v1.meta.json --local-repository runtime\torch_hub\facebookresearch_dinov2_main --cache-dir runtime\torch_hub --device cuda --enable-rejection --save-json output\prediction_results.json
```

接收方应先校验 `SHA256SUMS.txt`，再运行一张 JPG/PNG 图片，确认 JSON 含有 `result_label`、`confidence`、`similarity`、`margin` 和 `class_probabilities`。DINOv2 版本不使用旧 ResNet 的 ONNX、阈值或原型文件。
