"""Add data-distribution and joint classification/rejection metrics to the 260922 note."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCX_PATH = PROJECT_ROOT / "新峰测验" / "260922 识别模型实现说明.docx"


def set_run(run, size: float = 10.5, bold: bool = False, color: str | None = None) -> None:
    run.font.name = "宋体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def shade(cell, fill: str) -> None:
    tcpr = cell._tc.get_or_add_tcPr()
    element = OxmlElement("w:shd")
    element.set(qn("w:fill"), fill)
    tcpr.append(element)


def replace_text(paragraph, text: str, size: float = 10.5) -> None:
    paragraph.clear()
    set_run(paragraph.add_run(text), size=size)


def add_before(document: Document, anchor, text: str = "", style: str | None = None, align=None):
    paragraph = document.add_paragraph(style=style)
    if text:
        set_run(paragraph.add_run(text))
    if align is not None:
        paragraph.alignment = align
    anchor._p.addprevious(paragraph._p)
    return paragraph


def add_caption(document: Document, anchor, text: str) -> None:
    paragraph = add_before(document, anchor, text, align=WD_ALIGN_PARAGRAPH.CENTER)
    for run in paragraph.runs:
        set_run(run, size=10)


def add_table(document: Document, anchor, headers: list[str], rows: list[list[str]], caption: str) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, headers, strict=True):
        cell.text = ""
        set_run(cell.paragraphs[0].add_run(text), size=8.7, bold=True, color="FFFFFF")
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        shade(cell, "1F4E78")
    for index, values in enumerate(rows):
        cells = table.add_row().cells
        for cell, text in zip(cells, values, strict=True):
            cell.text = ""
            set_run(cell.paragraphs[0].add_run(text), size=8.4, bold=(index == len(rows) - 1 and values[0] == "合计"))
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if index % 2 == 1:
                shade(cell, "EAF2F8")
    anchor._p.addprevious(table._tbl)
    add_caption(document, anchor, caption)


def find_paragraph(document: Document, prefix: str):
    paragraph = next((item for item in document.paragraphs if item.text.startswith(prefix)), None)
    if paragraph is None:
        raise ValueError(f"cannot find paragraph starting with: {prefix}")
    return paragraph


def main() -> None:
    document = Document(DOCX_PATH)
    if any(paragraph.text.strip() == "实验数据构成与评价口径" for paragraph in document.paragraphs):
        raise ValueError("metrics section already exists")

    section_index = next(
        (index for index, paragraph in enumerate(document.paragraphs) if paragraph.text.startswith("两阶段采集图片模型优化实验")),
        None,
    )
    if section_index is None:
        raise ValueError("cannot locate the optimization-experiment heading")
    first_text = document.paragraphs[section_index + 1]
    second_text = document.paragraphs[section_index + 2]
    replace_text(
        first_text,
        "在原有第二阶段现场识别测试的基础上，本轮将第一阶段和第二阶段料棚图片按采集批次重建为煤矸石、矿渣和钢渣三分类数据集。各组优化实验均固定独立验证集和测试集，避免同一连续采集片段同时进入训练与测试；训练图像增强、原型数量和裁剪视角变化均在同一数据划分下比较。",
    )
    replace_text(
        second_text,
        "实验比较首先以原始分类准确率评价模型直接区分三种物料的能力；在此基础上，再用拒识与自动输出指标评价系统在需要人工复核时的覆盖范围和风险。以下先说明数据构成与评价口径，再依次比较图像增强、每类原型数量以及裁剪和视角设置。",
    )

    anchor = find_paragraph(document, "图像增强实验")
    heading = add_before(document, anchor, "实验数据构成与评价口径", style="Heading 2")
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_before(
        document,
        anchor,
        "本轮统一三分类优化共使用 1,364 张料棚原始图片，覆盖 25 个按采集时间划分的独立批次。每种物料均按时间顺序划分：较早批次用于训练，紧邻的 1 个批次用于验证，最新 2 个批次用于测试，因此同一批次不会跨训练、验证和测试集合。钢渣图片数量较多，煤矸石和矿渣相对较少；该不均衡反映现场实际采集量，后续必须同时报告各物料指标，不能只看总体准确率。",
    )
    add_table(
        document,
        anchor,
        ["物料", "原始图片\n批次数", "训练集\n图片/批次", "验证集\n图片/批次", "测试集\n图片/批次", "增强后训练\n图片数"],
        [
            ["煤矸石", "382 / 8", "279 / 5", "32 / 1", "71 / 2", "558"],
            ["矿渣", "301 / 6", "167 / 3", "46 / 1", "88 / 2", "334"],
            ["钢渣", "681 / 11", "544 / 8", "65 / 1", "72 / 2", "1,088"],
            ["合计", "1,364 / 25", "990 / 16", "143 / 3", "231 / 6", "1,980"],
        ],
        "表4 两阶段料棚三分类优化实验的数据与批次分布；增强仅作用于训练原图，验证和测试始终使用原始图片",
    )
    add_before(
        document,
        anchor,
        "增强实验对 990 张训练原图各生成 1 张轻度增强图，因此增强后的训练输入为 1,980 张；验证集仍为 143 张、测试集仍为 231 张原始图片。这样可以将性能变化归因于训练策略或模型参数，而不是测试图片数量或图片来源发生变化。",
    )

    heading = add_before(document, anchor, "识别与拒识联合评价指标", style="Heading 2")
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_before(
        document,
        anchor,
        "拒识机制加入后，不能只用单一“准确率”概括系统表现。原始分类准确率回答模型能否正确区分三种已知物料；拒识相关指标则回答系统自动给出结果的比例、自动输出是否可靠，以及有多少本来正确的图片被转交人工复核。为避免概念混淆，原表中的“正确接受率”统一称为“自动正确输出比例”。",
    )
    add_table(
        document,
        anchor,
        ["指标", "计算方法", "主要反映的问题", "判读原则"],
        [
            ["原始分类准确率", "分类正确图片 / 全部图片", "不考虑拒识时的三分类能力", "作为模型能力的首要比较指标"],
            ["拒识率", "拒识图片 / 全部图片", "需要人工复核的工作量", "低不一定更好，需与自动输出可靠性一起看"],
            ["自动输出覆盖率", "未拒识图片 / 全部图片", "系统可直接给出物料结果的范围", "等于 1 - 拒识率"],
            ["接受后准确率", "正确且未拒识图片 / 未拒识图片", "系统自动输出结果的可靠性", "用于控制错误自动判别风险"],
            ["自动正确输出比例", "正确且未拒识图片 / 全部图片", "系统无需人工复核即可正确完成的比例", "原“正确接受率”；体现端到端自动化贡献"],
            ["接受错误率", "错误且未拒识图片 / 未拒识图片", "被系统直接放行的错误风险", "应尽可能低，并结合错误方向分析"],
            ["正确样本误拒率", "正确但被拒识图片 / 原始分类正确图片", "拒识规则对正确图片的误伤程度", "过高说明阈值或拒识逻辑过严"],
            ["分物料指标", "对三种物料分别计算上述核心指标", "类别不均衡和特定物料短板", "总体结果不能替代煤矸石、矿渣和钢渣的分别结果"],
        ],
        "表5 三分类识别与拒识的联合评价指标及其判读方式",
    )
    add_before(
        document,
        anchor,
        "以当前最优的增强 seed 42、每类 3 个原型、六视图 0.78 裁剪配置为例：231 张测试图片中，原始分类正确 224 张，原始准确率为 96.97%；其中 49 张被拒识，自动输出覆盖率为 78.79%。最终自动正确输出 177 张，对应自动正确输出比例 76.62%，但在 182 张已自动输出图片中有 177 张正确，接受后准确率为 97.25%。被拒识的 49 张中有 47 张原始分类其实正确，说明当前拒识规则对已知物料样本偏保守；该结果应被视为阈值校准问题，而不能直接解释为三分类模型能力不足。",
    )
    add_before(
        document,
        anchor,
        "需要注意的是，本轮验证和测试集均为三种已知物料，拒识率反映的是已知物料样本被转交人工复核的比例，尚不能代表模型对未知物料、空料棚或异常画面的识别能力。后续应单独设置未知样本校准集，采用“自动输出覆盖率、接受后准确率和未知样本拒识能力”共同确定阈值，而不宜只追求较低的拒识率。",
    )

    # Update the three existing experiment tables and their interpretation so the metric name is precise.
    for table_index in (3, 4, 5):
        for cell in document.tables[table_index].rows[0].cells:
            if cell.text.strip() == "正确接受率":
                cell.text = ""
                set_run(cell.paragraphs[0].add_run("自动正确\n输出比例"), size=8.5, bold=True, color="FFFFFF")
                cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    replace_text(find_paragraph(document, "表4 图像增强实验"), "表6 图像增强实验的独立测试结果（验证集 n=143，测试集 n=231）", size=10)
    replace_text(find_paragraph(document, "表5 原型数量实验"), "表7 原型数量实验的独立测试结果（后三列为各物料原始识别准确率）", size=10)
    replace_text(find_paragraph(document, "表6 裁剪与视角变化实验"), "表8 裁剪与视角变化实验的独立测试结果", size=10)

    result_paragraph = find_paragraph(document, "三次增强实验的原始准确率")
    replace_text(
        result_paragraph,
        "三次增强实验的原始准确率均高于基线，平均值由 92.64% 提升至 95.38%，提高 2.74 个百分点；拒识率平均降低 4.33 个百分点，自动正确输出比例由 71.86% 提升至 75.76%。按物料看，煤矸石、矿渣和钢渣的增强实验平均准确率分别为 92.02%、97.35% 和 96.29%，相对基线分别提高 1.88、1.90 和 4.62 个百分点。增强后的主要残余错误仍为煤矸石被识别为钢渣（每次 5 至 6 张），说明轻度增强可提高整体稳定性，但尚不能消除两类物料的外观相似问题。由于三次结果在 94.37% 至 96.97% 之间波动，后续报告以三次平均值而非单次最高值作为增强结论。",
    )
    view_paragraph = find_paragraph(document, "原始六视图 0.78 裁剪策略")
    replace_text(
        view_paragraph,
        "原始六视图 0.78 裁剪策略（V0）仍是直接三分类性能最好的设置，总体原始准确率为 96.97%，并且煤矸石被误识为钢渣的数量最少。仅使用全图（V1）时准确率下降 5.20 个百分点，表明局部视角有助于模型提取物料纹理和颗粒细节。全图加中心视图（V2）的拒识率较低、自动正确输出比例较高，但其原始准确率较 V0 低 2.60 个百分点，同时错误自动输出数量增加，因此不能据此判定模型分类更优。将裁剪比例增大到 0.90（V3）也没有带来提升。由此保留 V0 作为后续统一三分类模型的默认视角策略。",
    )

    document.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
