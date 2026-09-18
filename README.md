# DINOv2 工业固废图像识别模型

这是当前可运行的图像识别工程，使用冻结的 DINOv2 提取图像特征，再用多个物料原型完成分类、视角投票和拒识判断。当前代码支持已有四类模型（煤矸石、矿渣、钢渣、石膏），并保留第二阶段煤矸石、矿渣、钢渣三分类实验的训练和评估脚本。

## 1. 目录结构

```text
solid_waste_model/       模型代码、配置、推理接口和测试用核心模块
artifacts/               按模型版本保存的原型库、元数据和评估结果
data/                    manifest、原始数据入口和增强数据入口
docs/                    当前运行说明和交付说明
scripts/                 第二阶段清单准备、训练、实验汇总和报告脚本
tests/                   当前工程单元测试
runtime/torch_hub/       本地 DINOv2 源码和权重缓存
node_modules/            仅供报表脚本使用的本地依赖
```

训练和实验脚本集中在 `scripts/`，正式训练仍由人工确认后手动启动，不会由文档脚本自动执行。

## 2. 环境与测试

从仓库根目录运行当前工程测试：

```powershell
\.venv312\Scripts\python.exe -m pytest -q IWaste_rec_model\tests
```

进入 `IWaste_rec_model/` 后也可以使用：

```powershell
..\.venv312\Scripts\python.exe -m pytest -q tests
```

当前测试只验证清单划分、图像裁剪、原型聚类、分类和拒识逻辑，不会加载完整 DINOv2 权重，也不会修改模型产物。

## 3. 快速识别

双击 `run_gui.bat`，在界面中选择图片或文件夹后运行识别。命令行批量识别可以使用：

```powershell
python -m solid_waste_model.dinov2_proto predict path\to\images `
  --recursive `
  --prototype artifacts\solid_waste_dinov2_proto_v1\solid_waste_dinov2_proto_v1.prototypes.npz `
  --meta artifacts\solid_waste_dinov2_proto_v1\solid_waste_dinov2_proto_v1.meta.json `
  --local-repository runtime\torch_hub\facebookresearch_dinov2_main `
  --cache-dir runtime\torch_hub `
  --device cuda `
  --enable-rejection `
  --save-json output\prediction_results.json
```

首次运行需要本地 DINOv2 源码和权重。模型实例应在服务启动时创建一次，不要对每张图片重复加载权重。

## 4. 第二阶段三分类实验

完整运行顺序、输入清单、输出目录、增强种子、原型数量和裁剪视角参数见：

`docs/第二轮统一三分类基线运行说明.md`

核心脚本如下：

```text
prepare_round2_baseline_manifest.py       准备统一三分类基线清单
prepare_round2_augmented_manifest.py     准备训练集增强清单
train_round2_baseline.py                 手动启动训练和评估
prepare_round2_prototype_sweep.py       准备原型数量对比清单/参数
prepare_round2_view_crop_sweep.py       准备裁剪与视角对比参数
summarize_round2_*_sweep.py              汇总已完成实验结果
```

截至 2026 年 9 月 18 日，当前实验结论为：图像增强三次实验平均原始准确率 95.38%，每类 3 个原型的总体原始准确率 96.97%，六视图、0.78 裁剪比例仍是最佳直接三分类配置。主要残余错误为煤矸石被识别为钢渣。

## 5. 数据约定

原始数据按“场景/物料”组织，训练清单中应保留 `image_path`、`material_type`、`scene`、`split`、`is_known_class`、`sample_group`、`augmentation_source` 和 `usage_role` 等字段。同一采集批次不能跨训练、验证和测试集合。

正式实验中，验证集和测试集保持原始图片，不使用增强图片；增强只作用于训练集。测试结果必须同时查看总体准确率、三种物料准确率和混淆方向。

## 6. 模型产物

每个 `artifacts/<model_version>/` 目录应包含：

```text
<model_version>.prototypes.npz
<model_version>.meta.json
<model_version>.report.json
<model_version>.validation_errors.csv
<model_version>.test_errors.csv
```

`meta.json`、`prototypes.npz` 和 DINOv2 源码/权重必须来自同一次训练或同一交付版本。`report.json` 中的拒识阈值只用于运行口径，不能代替独立测试集上的原始分类准确率。

## 7. 依赖和提交边界

- `runtime/`、`node_modules/`、原始图片、增强图片、清单和运行输出属于本地或可再生内容，不应作为普通代码提交。
- `artifacts/` 中的模型报告和错误表是实验留档；原型库是否随交付包保留，以具体部署需求为准。
- 交付包由 `scripts/package_dinov2_delivery.py` 生成。生成前应先确认模型版本、权重和说明文档完全对应。
