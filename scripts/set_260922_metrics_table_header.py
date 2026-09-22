"""Repeat the metrics-table header when the table flows to a new page."""

from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn


DOCX_PATH = Path(__file__).resolve().parents[2] / "新峰测验" / "260922 识别模型实现说明.docx"


def main() -> None:
    document = Document(DOCX_PATH)
    table = next((item for item in document.tables if item.rows[0].cells[0].text.strip() == "指标"), None)
    if table is None:
        raise ValueError("cannot locate metrics table")
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    existing = tr_pr.find(qn("w:tblHeader"))
    if existing is None:
        marker = OxmlElement("w:tblHeader")
        marker.set(qn("w:val"), "true")
        tr_pr.append(marker)
    document.save(DOCX_PATH)
    print("metrics_table_header_repeated=true")


if __name__ == "__main__":
    main()
