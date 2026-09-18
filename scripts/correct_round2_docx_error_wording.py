"""Correct the material-confusion direction in the already updated model note."""

from __future__ import annotations

from pathlib import Path

from docx import Document


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCX_PATH = PROJECT_ROOT / "新峰测验" / "260908 识别模型实现说明.docx"
OLD = (
    "按物料分析，煤矸石、矿渣和钢渣在增强实验中的平均原始准确率分别为 92.02%、97.35% 和 96.29%，较基线分别提高 1.88、1.90 和 4.62 个百分点，其中钢渣改善最明显。"
    "三次增强实验中最稳定的错误方向仍是煤矸石被识别为矿渣，说明两者在现场外观纹理和堆体形态上仍存在相似区域；矿渣与钢渣之间的混淆明显减少。"
    "由于三次增强结果存在 94.37% 至 96.97% 的波动，报告增强效果时应采用三次实验平均值，而不宜只选取最高准确率作为最终结论。"
)
NEW = (
    "按物料分析，煤矸石、矿渣和钢渣在增强实验中的平均原始准确率分别为 92.02%、97.35% 和 96.29%，较基线分别提高 1.88、1.90 和 4.62 个百分点，其中钢渣改善最明显。"
    "三次增强实验中最稳定的错误方向仍是煤矸石被识别为钢渣（5～6 张），说明两者在现场外观纹理和堆体形态上仍存在相似区域；矿渣被识别为钢渣是另一主要错误方向，增强后由基线 4 张降至 1～3 张，但钢渣与矿渣、煤矸石之间的少量反向混淆仍然存在。"
    "由于三次增强结果存在 94.37% 至 96.97% 的波动，报告增强效果时应采用三次实验平均值，而不宜只选取最高准确率作为最终结论。"
)


def main() -> None:
    document = Document(DOCX_PATH)
    matches = [paragraph for paragraph in document.paragraphs if paragraph.text == OLD]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one target paragraph, found {len(matches)}")
    paragraph = matches[0]
    paragraph.text = NEW
    document.save(DOCX_PATH)
    print("corrected=1")


if __name__ == "__main__":
    main()
