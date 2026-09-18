import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const scriptPath = fileURLToPath(import.meta.url);
const projectRoot = path.resolve(path.dirname(scriptPath), "..", "..");
const manifestDir = path.join(projectRoot, "ml_data", "manifests");
const outputPath = path.join(projectRoot, "新峰测验", "料棚图像数据库.xlsx");
const qaDir = path.join(projectRoot, "output", "qa_stockyard_database");
const images = JSON.parse(await fs.readFile(path.join(manifestDir, "stockyard_image_database.json"), "utf8"));
const batches = JSON.parse(await fs.readFile(path.join(manifestDir, "stockyard_capture_batches.json"), "utf8"));
const summary = JSON.parse(await fs.readFile(path.join(manifestDir, "stockyard_image_database_summary.json"), "utf8"));

const wb = Workbook.create();
const overview = wb.worksheets.add("总览");
const imageSheet = wb.worksheets.add("图片清单");
const batchSheet = wb.worksheets.add("采集批次");
const issueSheet = wb.worksheets.add("异常待核对");
const font = "Arial";
const titleStyle = { font: { name: font, size: 14, bold: true, color: "#1F4E78" } };
const headerStyle = { fill: "#1F4E78", font: { name: font, size: 10, bold: true, color: "#FFFFFF" }, horizontalAlignment: "center", verticalAlignment: "center" };
const noteStyle = { font: { name: font, size: 10, italic: true, color: "#595959" } };

function writeTable(sheet, headers, rows) {
  sheet.getRangeByIndexes(0, 0, 1, headers.length).values = [headers];
  sheet.getRangeByIndexes(0, 0, 1, headers.length).format = headerStyle;
  if (rows.length) {
    sheet.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows.map(row => headers.map(header => row[header] ?? ""));
  }
  sheet.getRangeByIndexes(0, 0, Math.max(1, rows.length + 1), headers.length).format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = false;
}

overview.showGridLines = false;
overview.getRange("A2").values = [["料棚图像数据库｜第一轮数据整理"]];
overview.getRange("A2").format = titleStyle;
overview.getRange("A4:B9").values = [
  ["指标", "数值"],
  ["图片总数", summary.total_images],
  ["文件名时间解析成功", summary.parsed_filename_timestamps],
  ["采集批次阈值（分钟）", summary.batch_gap_minutes],
  ["自动划分采集批次", summary.batch_count],
  ["昼夜字段", "已保留，仅用于后续描述统计"],
];
overview.getRange("A4:B4").format = headerStyle;
overview.getRange("A4:B9").format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
overview.getRange("A12").values = [["数据规则"]];
overview.getRange("A12").format = { font: { name: font, size: 11, bold: true, color: "#1F4E78" } };
overview.getRange("A13:A17").values = [
  ["1. 文件名中的采集时间用于排序与采集批次划分；全部图片均可解析。"],
  ["2. 相同摄像头、相同物料、相同阶段下，相邻图片间隔超过30分钟即视为新批次。"],
  ["3. 第一阶段焦距记为“未标注”；第二阶段焦距取自采集文件夹。"],
  ["4. 昼夜字段以19:00为界保存，第一轮不用于样本平衡或模型条件输入。"],
];
overview.getRange("A13:A17").format = noteStyle;
overview.getRange("A20:D20").values = [["物料", "阶段", "图片数", "采集批次数"]];
overview.getRange("A20:D20").format = headerStyle;
const phaseCounts = summary.counts_by_material_phase.map(item => {
  const batchCount = batches.filter(batch => batch.material === item.material && batch.phase === item.phase).length;
  return [item.material, item.phase, item.image_count, batchCount];
});
overview.getRangeByIndexes(20, 0, phaseCounts.length, 4).values = phaseCounts;
overview.getRangeByIndexes(19, 0, phaseCounts.length + 1, 4).format.borders = { preset: "all", style: "thin", color: "#D9E2F3" };
overview.getRange("A:A").format.columnWidth = 42;
overview.getRange("B:B").format.columnWidth = 22;
overview.getRange("C:D").format.columnWidth = 18;

const imageHeaders = ["image_id", "relative_path", "file_name", "material", "phase", "camera_id", "file_timestamp", "capture_date", "capture_time", "day_night", "focus_group", "capture_folder", "folder_declared_datetime", "batch_id", "batch_gap_minutes", "file_size_bytes", "parse_status", "notes"];
writeTable(imageSheet, imageHeaders, images);
imageSheet.freezePanes.freezeColumns(3);
imageSheet.getRange("A:R").format.font = { name: font, size: 9 };
imageSheet.getRange("A:A").format.columnWidth = 16;
imageSheet.getRange("B:B").format.columnWidth = 64;
imageSheet.getRange("C:C").format.columnWidth = 42;
imageSheet.getRange("D:K").format.columnWidth = 16;
imageSheet.getRange("L:L").format.columnWidth = 42;
imageSheet.getRange("M:M").format.columnWidth = 24;
imageSheet.getRange("N:N").format.columnWidth = 42;
imageSheet.getRange("O:R").format.columnWidth = 20;

const batchHeaders = ["batch_id", "material", "phase", "camera_id", "batch_start", "batch_end", "duration_minutes", "image_count", "day_night_values", "focus_groups", "capture_folders", "rule"];
writeTable(batchSheet, batchHeaders, batches);
batchSheet.getRange("A:L").format.font = { name: font, size: 10 };
batchSheet.getRange("A:A").format.columnWidth = 42;
batchSheet.getRange("B:D").format.columnWidth = 18;
batchSheet.getRange("E:F").format.columnWidth = 22;
batchSheet.getRange("G:H").format.columnWidth = 16;
batchSheet.getRange("I:J").format.columnWidth = 20;
batchSheet.getRange("K:K").format.columnWidth = 70;
batchSheet.getRange("L:L").format.columnWidth = 48;

const issues = images.filter(row => row.parse_status !== "文件名时间已解析");
issueSheet.getRange("A1").values = [["说明：本页仅列出文件名时间未成功解析的图片。"]];
issueSheet.getRange("A1").format = noteStyle;
if (issues.length) {
  issueSheet.getRangeByIndexes(2, 0, 1, imageHeaders.length).values = [imageHeaders];
  issueSheet.getRangeByIndexes(2, 0, 1, imageHeaders.length).format = headerStyle;
  issueSheet.getRangeByIndexes(3, 0, issues.length, imageHeaders.length).values = issues.map(row => imageHeaders.map(header => row[header] ?? ""));
} else {
  issueSheet.getRange("A3").values = [["本轮无文件名时间解析异常：1,364 张图片均已完成时间、物料和阶段归档。"]];
  issueSheet.getRange("A3").format = { font: { name: font, size: 11, bold: true, color: "#1F4E78" } };
}
issueSheet.getRange("A:R").format.font = { name: font, size: 9 };
issueSheet.freezePanes.freezeRows(3);
issueSheet.showGridLines = false;

wb.recalculate();
const overviewCheck = await wb.inspect({
  kind: "table",
  range: "总览!A2:D26",
  include: "values,formulas",
  table_max_rows: 30,
  table_max_cols: 6,
});
console.log(overviewCheck.ndjson);
const formulaErrorCheck = await wb.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { use_regex: true, max_results: 50 },
  summary: "final formula error scan",
});
console.log(formulaErrorCheck.ndjson);
await fs.mkdir(qaDir, { recursive: true });
for (const [sheetName, range] of [["总览", "A1:D26"], ["图片清单", "A1:T30"], ["采集批次", "A1:L26"], ["异常待核对", "A1:T12"]]) {
  const image = await wb.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(path.join(qaDir, `${sheetName}.png`), Buffer.from(await image.arrayBuffer()));
}
await fs.mkdir(path.dirname(outputPath), { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(wb);
await xlsx.save(outputPath);
console.log(outputPath);
