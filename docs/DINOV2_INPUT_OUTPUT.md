# DINOv2 输入输出说明

输入支持 JPG、JPEG、PNG、BMP、WEBP、TIF 和 TIFF，支持单图或文件夹。图片读取后统一转为 RGB，并生成全图、中心、左上、右上、左下、右下六个视角；每个视角缩放到 448×448，使用 ImageNet mean/std 标准化后送入冻结的 DINOv2 ViT-S/14。

模型输出包括四类物料的类别概率、最高类别的原型相似度、第一和第二类别的分数间隔，以及六个裁剪视角的投票明细。置信度、相似度或类别间隔低于元数据中的阈值时，最终结果为“其他”。

业务适配器返回 `waste_type`、`confidence`、`features`、`model_result` 和 `model_error`。其中 `features` 当前来自 `material_profiles.json` 的配置画像，不能视为真实属性预测。
