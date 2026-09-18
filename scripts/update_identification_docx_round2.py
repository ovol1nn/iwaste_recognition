"""Append round-two baseline and augmentation comparison results to the model note."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCX_PATH = PROJECT_ROOT / "新峰测验" / "260908 识别模型实现说明.docx"
ASSET_DIR = PROJECT_ROOT / "IWaste_rec_model" / "output" / "round2_docx_figures"
FONT_PATH = Path(r"C:\Windows\Fonts\msyh.ttc")
COLORS = {
    "baseline": "#4E79A7",
    "aug1": "#59A14F",
    "aug2": "#F28E2B",
    "aug3": "#B07AA1",
    "grid": "#D9E2F3",
    "axis": "#52616B",
    "text": "#1F2933",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    index = 1 if bold else 0
    return ImageFont.truetype(str(FONT_PATH), size=size, index=index)


def hex_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))


def draw_text_center(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, text_font: ImageFont.FreeTypeFont, fill: str) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    draw.text((left + (right - left - width) / 2, top + (bottom - top - height) / 2), text, font=text_font, fill=hex_rgb(fill))


def draw_bar_panel(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    size: tuple[int, int],
    title: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    y_max: float,
    value_suffix: str = "%",
) -> None:
    x, y = origin
    width, height = size
    left, right = x + 92, x + width - 25
    top, bottom = y + 68, y + height - 74
    draw.text((x, y), title, font=font(28, True), fill=hex_rgb(COLORS["text"]))
    for tick in range(0, 101, 20):
        py = bottom - (bottom - top) * tick / y_max
        draw.line((left, py, right, py), fill=hex_rgb(COLORS["grid"]), width=2)
        draw.text((x + 15, py - 12), str(tick), font=font(18), fill=hex_rgb(COLORS["axis"]))
    draw.line((left, top, left, bottom), fill=hex_rgb(COLORS["axis"]), width=2)
    draw.line((left, bottom, right, bottom), fill=hex_rgb(COLORS["axis"]), width=2)
    step = (right - left) / len(values)
    bar_width = step * 0.50
    for index, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        center = left + step * (index + 0.5)
        bar_top = bottom - (bottom - top) * value / y_max
        draw.rounded_rectangle((center - bar_width / 2, bar_top, center + bar_width / 2, bottom), radius=5, fill=hex_rgb(color))
        draw_text_center(draw, (int(center - bar_width), int(bar_top - 38), int(center + bar_width), int(bar_top - 2)), f"{value:.2f}{value_suffix}", font(18, True), COLORS["text"])
        draw_text_center(draw, (int(center - step / 2), bottom + 15, int(center + step / 2), bottom + 55), label, font(18), COLORS["text"])


def make_overall_figure(path: Path) -> None:
    image = Image.new("RGB", (1800, 760), "white")
    draw = ImageDraw.Draw(image)
    labels = ["基线", "增强\nseed 42", "增强\nseed 2026", "增强\nseed 2027"]
    colors = [COLORS["baseline"], COLORS["aug1"], COLORS["aug2"], COLORS["aug3"]]
    draw_bar_panel(draw, (55, 45), (830, 650), "（a）独立测试集原始准确率", labels, [92.64, 96.97, 94.81, 94.37], colors, 100)
    draw_bar_panel(draw, (915, 45), (830, 650), "（b）独立测试集拒识率", labels, [26.41, 21.21, 22.08, 22.94], colors, 100)
    draw.text((570, 710), "测试集固定为第二阶段独立采集批次，共 231 张图片", font=font(19), fill=hex_rgb(COLORS["axis"]))
    image.save(path, quality=95)


def make_material_figure(path: Path) -> None:
    image = Image.new("RGB", (1800, 780), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 150, 1730, 130, 635
    draw.text((60, 35), "各物料原始识别准确率：基线与增强实验平均值", font=font(32, True), fill=hex_rgb(COLORS["text"]))
    for tick in range(80, 101, 5):
        py = bottom - (bottom - top) * (tick - 80) / 20
        draw.line((left, py, right, py), fill=hex_rgb(COLORS["grid"]), width=2)
        draw.text((72, py - 12), str(tick), font=font(18), fill=hex_rgb(COLORS["axis"]))
    draw.line((left, top, left, bottom), fill=hex_rgb(COLORS["axis"]), width=2)
    draw.line((left, bottom, right, bottom), fill=hex_rgb(COLORS["axis"]), width=2)
    groups = [
        ("煤矸石", 90.14, 92.02, 91.55, 92.96),
        ("矿渣", 95.45, 97.35, 96.59, 98.86),
        ("钢渣", 91.67, 96.29, 94.44, 98.61),
    ]
    group_width = (right - left) / len(groups)
    scale = lambda value: bottom - (bottom - top) * (value - 80) / 20
    for index, (label, baseline, average, low, high) in enumerate(groups):
        center = left + group_width * (index + 0.5)
        baseline_x = center - 80
        average_x = center + 80
        for bx, value, color in [(baseline_x, baseline, COLORS["baseline"]), (average_x, average, COLORS["aug1"])]:
            top_y = scale(value)
            draw.rounded_rectangle((bx - 44, top_y, bx + 44, bottom), radius=5, fill=hex_rgb(color))
            draw_text_center(draw, (int(bx - 62), int(top_y - 38), int(bx + 62), int(top_y - 2)), f"{value:.2f}%", font(18, True), COLORS["text"])
        low_y, high_y = scale(low), scale(high)
        draw.line((average_x, high_y, average_x, low_y), fill=hex_rgb(COLORS["text"]), width=3)
        draw.line((average_x - 13, high_y, average_x + 13, high_y), fill=hex_rgb(COLORS["text"]), width=3)
        draw.line((average_x - 13, low_y, average_x + 13, low_y), fill=hex_rgb(COLORS["text"]), width=3)
        draw_text_center(draw, (int(center - 150), bottom + 22, int(center + 150), bottom + 66), label, font(23, True), COLORS["text"])
    draw.rectangle((520, 690, 550, 720), fill=hex_rgb(COLORS["baseline"]))
    draw.text((560, 689), "基线", font=font(19), fill=hex_rgb(COLORS["text"]))
    draw.rectangle((760, 690, 790, 720), fill=hex_rgb(COLORS["aug1"]))
    draw.text((800, 689), "增强实验平均值", font=font(19), fill=hex_rgb(COLORS["text"]))
    draw.text((1090, 689), "误差线：三次增强实验最小值至最大值", font=font(19), fill=hex_rgb(COLORS["text"]))
    image.save(path, quality=95)


def set_cell_shading(cell, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def format_run(run, size: float = 10.5, bold: bool = False, color: str | None = None) -> None:
    run.font.name = "宋体"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def insert_before(anchor, paragraph) -> None:
    anchor._p.addprevious(paragraph._p)


def insert_table_before(anchor, table) -> None:
    anchor._p.addprevious(table._tbl)


def add_inserted_paragraph(document: Document, anchor, text: str = "", style: str | None = None, align=None):
    paragraph = document.add_paragraph(style=style)
    if text:
        format_run(paragraph.add_run(text))
    if align is not None:
        paragraph.alignment = align
    insert_before(anchor, paragraph)
    return paragraph


def add_caption(document: Document, anchor, text: str):
    paragraph = add_inserted_paragraph(document, anchor, align=WD_ALIGN_PARAGRAPH.CENTER)
    format_run(paragraph.add_run(text), size=10)
    return paragraph


def add_figure(document: Document, anchor, image_path: Path, caption: str, width_cm: float = 15.0) -> None:
    paragraph = add_inserted_paragraph(document, anchor, align=WD_ALIGN_PARAGRAPH.CENTER)
    paragraph.add_run().add_picture(str(image_path), width=Cm(width_cm))
    add_caption(document, anchor, caption)


def add_result_table(document: Document, anchor) -> None:
    table = document.add_table(rows=1, cols=6)
    table.style = "Table Grid"
    headers = ["实验组", "训练图片数", "验证/测试图片数", "测试准确率", "拒识率", "正确接受率"]
    for cell, text in zip(table.rows[0].cells, headers, strict=True):
        cell.text = ""
        run = cell.paragraphs[0].add_run(text)
        format_run(run, size=9.5, bold=True, color="FFFFFF")
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_shading(cell, "1F4E78")
    rows = [
        ("基线模型", "990", "143 / 231", "92.64%", "26.41%", "71.86%"),
        ("增强 seed 42", "1,980", "143 / 231", "96.97%", "21.21%", "76.62%"),
        ("增强 seed 2026", "1,980", "143 / 231", "94.81%", "22.08%", "75.76%"),
        ("增强 seed 2027", "1,980", "143 / 231", "94.37%", "22.94%", "74.89%"),
        ("增强组平均值", "1,980", "143 / 231", "95.38%", "22.08%", "75.76%"),
    ]
    for row_index, values in enumerate(rows, start=1):
        cells = table.add_row().cells
        for cell, text in zip(cells, values, strict=True):
            cell.text = ""
            run = cell.paragraphs[0].add_run(text)
            format_run(run, size=9.5, bold=row_index == len(rows))
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            if row_index % 2 == 0:
                set_cell_shading(cell, "EAF2F8")
    table.autofit = True
    insert_table_before(anchor, table)


def main() -> None:
    if not DOCX_PATH.exists():
        raise FileNotFoundError(DOCX_PATH)
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    overall_path = ASSET_DIR / "round2_overall_comparison.png"
    material_path = ASSET_DIR / "round2_material_comparison.png"
    make_overall_figure(overall_path)
    make_material_figure(material_path)

    document = Document(DOCX_PATH)
    conclusion = next((paragraph for paragraph in document.paragraphs if paragraph.text.strip() == "结论"), None)
    if conclusion is None:
        raise ValueError("cannot locate the 结论 heading")
    if any("两阶段采集图片基线与增强对比结果" in paragraph.text for paragraph in document.paragraphs):
        raise ValueError("comparison section already exists")

    heading = add_inserted_paragraph(document, conclusion, "两阶段采集图片基线与增强对比结果", style="Heading 1")
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_inserted_paragraph(
        document,
        conclusion,
        "在原有第二阶段识别测试的基础上，进一步将第一阶段和第二阶段料棚图片按采集批次重建为三分类训练数据集。训练集同时包含两个阶段的图片，验证集和测试集均保留为第二阶段的独立采集批次，以降低同一连续采集片段同时进入训练和测试所造成的结果偏高风险。",
    )
    subheading = add_inserted_paragraph(document, conclusion, "实验设计与评价口径", style="Heading 2")
    subheading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_inserted_paragraph(
        document,
        conclusion,
        "基线模型采用 DINOv2 特征提取、每类 3 个原型和六视图投票；训练、验证和测试图片分别为 990、143 和 231 张。增强组保持类别、批次划分、模型结构、原型数量和测试集完全一致，仅对训练集原图施加轻度随机裁剪、水平翻转、小角度旋转、亮度与对比度调整及轻度模糊，每张训练图片生成 1 张增强图。为降低单次随机增强带来的偶然性，分别采用随机种子 42、2026 和 2027 重复训练。原始准确率用于评价模型直接分类能力；拒识率和正确接受率用于评价阈值校准后的实际输出表现。",
    )
    add_result_table(document, conclusion)
    add_caption(document, conclusion, "表4 两阶段料棚图片基线模型与增强模型的独立测试结果")
    add_figure(document, conclusion, overall_path, "图9 基线模型与三次增强实验在固定独立测试集上的总体准确率和拒识率")
    subheading = add_inserted_paragraph(document, conclusion, "结果分析", style="Heading 2")
    subheading.alignment = WD_ALIGN_PARAGRAPH.LEFT
    add_inserted_paragraph(
        document,
        conclusion,
        "基线模型在 231 张第二阶段独立测试图片上的原始准确率为 92.64%，拒识率为 26.41%。三次增强实验的原始准确率分别为 96.97%、94.81% 和 94.37%，均高于基线；平均值为 95.38%，标准差为 1.39 个百分点。增强组平均拒识率降至 22.08%，正确接受率由 71.86% 提升至 75.76%。这表明在固定测试批次下，轻度图像增强能够提高模型对采集距离、局部遮挡、光照和清晰度变化的适应能力。",
    )
    add_figure(document, conclusion, material_path, "图10 不同物料在基线与增强实验中的原始识别准确率对比；误差线表示三次增强实验的最小值至最大值")
    add_inserted_paragraph(
        document,
        conclusion,
        "按物料分析，煤矸石、矿渣和钢渣在增强实验中的平均原始准确率分别为 92.02%、97.35% 和 96.29%，较基线分别提高 1.88、1.90 和 4.62 个百分点，其中钢渣改善最明显。三次增强实验中最稳定的错误方向仍是煤矸石被识别为钢渣（5～6 张），说明两者在现场外观纹理和堆体形态上仍存在相似区域；矿渣被识别为钢渣是另一主要错误方向，增强后由基线 4 张降至 1～3 张，但钢渣与矿渣、煤矸石之间的少量反向混淆仍然存在。由于三次增强结果存在 94.37% 至 96.97% 的波动，报告增强效果时应采用三次实验平均值，而不宜只选取最高准确率作为最终结论。",
    )
    add_inserted_paragraph(
        document,
        conclusion,
        "综合来看，批次独立测试验证了模型在第二阶段现场新采图片上的泛化能力；在此基础上加入轻度训练集增强后，三分类性能获得稳定提升。增强策略可作为后续模型优化的基础设置，但仍应在更多独立采集批次上持续验证，并围绕煤矸石与矿渣的混淆开展下一步原型数量和特征分支优化实验。",
    )

    document.save(DOCX_PATH)
    print(DOCX_PATH)


if __name__ == "__main__":
    main()
