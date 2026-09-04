from __future__ import annotations

import json
import csv
import random
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import PROJECT_ROOT, SUPPORTED_IMAGE_SUFFIXES
from .dinov2_inference import WasteClassifier


MODEL_VERSION = "solid_waste_dinov2_proto_v1"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / MODEL_VERSION
META_PATH = ARTIFACT_DIR / f"{MODEL_VERSION}.meta.json"
PROTOTYPE_PATH = ARTIFACT_DIR / f"{MODEL_VERSION}.prototypes.npz"
PROFILE_PATH = PROJECT_ROOT / "solid_waste_model" / "configs" / "material_profiles.json"
LOCAL_REPOSITORY = PROJECT_ROOT / "runtime" / "torch_hub" / "facebookresearch_dinov2_main"
CACHE_DIR = PROJECT_ROOT / "runtime" / "torch_hub"
OUTPUT_JSON = PROJECT_ROOT / "output" / "interactive_predict_results.json"


def collect_images(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES else []
    if not path.is_dir():
        raise FileNotFoundError(f"路径不存在：{path}")
    iterator = path.rglob("*") if recursive else path.glob("*")
    return sorted(item for item in iterator if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES)


def load_random_test_rows(manifest_path: Path, count: int) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row.get("usage_role") == "known_test"
            and row.get("is_known_class") == "True"
            and row.get("augmentation_source") == "original"
        ]
    if not rows:
        raise ValueError("manifest 中没有 known_test 原始图片")
    return random.sample(rows, k=min(max(count, 1), len(rows)))


class ModelTesterApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("固废物料图像识别测试")
        self.geometry("1360x720")
        self.minsize(1080, 600)
        self.selected_images: list[Path] = []
        self.results: list[dict[str, object]] = []
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.path_var = tk.StringVar(value=str(PROJECT_ROOT / "data" / "incoming"))
        self.recursive_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="请选择图片或文件夹，然后点击运行识别。")
        self.progress_var = tk.DoubleVar(value=0.0)
        self._build_ui()
        self.after(100, self._drain_events)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        top = ttk.Frame(root)
        top.pack(fill=tk.X)
        ttk.Label(top, text="测试路径").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        ttk.Entry(top, textvariable=self.path_var).grid(row=0, column=1, sticky=tk.EW)
        top.columnconfigure(1, weight=1)
        ttk.Button(top, text="选择图片", command=self.select_images).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(top, text="选择文件夹", command=self.select_folder).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(top, text="载入路径", command=self.load_path).grid(row=0, column=4, padx=(8, 0))

        controls = ttk.Frame(root)
        controls.pack(fill=tk.X, pady=(10, 8))
        ttk.Checkbutton(controls, text="递归文件夹", variable=self.recursive_var).pack(side=tk.LEFT)
        ttk.Button(controls, text="运行识别", command=self.run_predict).pack(side=tk.LEFT, padx=(14, 0))
        ttk.Button(controls, text="保存结果", command=self.save_results).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(controls, text="打开结果目录", command=self.open_output_folder).pack(side=tk.LEFT, padx=(8, 0))

        columns = ("image", "result", "confidence", "rejected", "similarity", "margin", "scores", "error")
        self.table = ttk.Treeview(root, columns=columns, show="headings", height=20)
        headings = {"image": "图片", "result": "识别结果", "confidence": "置信度", "rejected": "拒识", "similarity": "相似度", "margin": "类别间隔", "scores": "四类分数", "error": "错误"}
        widths = {"image": 330, "result": 90, "confidence": 80, "rejected": 70, "similarity": 80, "margin": 80, "scores": 390, "error": 220}
        for column in columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=widths[column], minwidth=60, anchor=tk.W)
        y_scroll = ttk.Scrollbar(root, orient=tk.VERTICAL, command=self.table.yview)
        x_scroll = ttk.Scrollbar(root, orient=tk.HORIZONTAL, command=self.table.xview)
        self.table.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
        self.table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        x_scroll.pack(fill=tk.X)
        bottom = ttk.Frame(root)
        bottom.pack(fill=tk.X, pady=(8, 0))
        ttk.Progressbar(bottom, variable=self.progress_var, maximum=100).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(bottom, textvariable=self.status_var, width=70).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(root, text=f"模型：{MODEL_VERSION} | 原型库：{PROTOTYPE_PATH}").pack(fill=tk.X, pady=(8, 0))

    def select_images(self) -> None:
        names = filedialog.askopenfilenames(title="选择测试图片", filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("全部文件", "*.*")])
        if names:
            self.selected_images = [Path(name) for name in names]
            self.path_var.set(str(self.selected_images[0].parent))
            self._reset_table()
            self.status_var.set(f"已选择 {len(self.selected_images)} 张图片。")

    def select_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择测试文件夹")
        if folder:
            self.path_var.set(folder)
            self.load_path()

    def load_path(self) -> None:
        try:
            self.selected_images = collect_images(Path(self.path_var.get()), self.recursive_var.get())
        except Exception as exc:
            messagebox.showerror("载入失败", str(exc))
            return
        self._reset_table()
        self.status_var.set(f"已载入 {len(self.selected_images)} 张图片。")

    def run_predict(self) -> None:
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("正在运行", "当前识别任务还没有结束。")
            return
        if not self.selected_images:
            self.load_path()
        if not self.selected_images:
            messagebox.showwarning("没有图片", "请先选择图片或文件夹。")
            return
        self._reset_table()
        self.progress_var.set(0)
        self.status_var.set("正在加载 DINOv2 并识别...")
        self.worker = threading.Thread(target=self._predict_worker, daemon=True)
        self.worker.start()

    def _predict_worker(self) -> None:
        try:
            classifier = WasteClassifier(meta_path=META_PATH, prototype_path=PROTOTYPE_PATH, profile_path=PROFILE_PATH, local_repository=LOCAL_REPOSITORY, cache_dir=CACHE_DIR)
            total = len(self.selected_images)
            for index, image_path in enumerate(self.selected_images, start=1):
                payload = classifier.safe_predict_payload(image_path)
                payload["image_path"] = str(image_path)
                self.events.put(("result", payload))
                self.events.put(("progress", index / total * 100))
            self.events.put(("done", None))
        except Exception as exc:
            self.events.put(("failed", str(exc)))

    def _drain_events(self) -> None:
        while True:
            try:
                event, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if event == "result":
                self._add_result(payload)  # type: ignore[arg-type]
            elif event == "progress":
                self.progress_var.set(float(payload))
            elif event == "done":
                self.save_results(show_message=False)
                self.status_var.set(f"完成：{len(self.results)} 张图片，结果已保存。")
            elif event == "failed":
                self.status_var.set("识别失败。")
                messagebox.showerror("识别失败", str(payload))
        self.after(100, self._drain_events)

    def _add_result(self, payload: dict[str, object]) -> None:
        self.results.append(payload)
        details = payload.get("model_result") or {}
        scores = details.get("material_scores") or {}  # type: ignore[union-attr]
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)  # type: ignore[union-attr]
        summary = "  ".join(f"{label}:{float(score):.3f}" for label, score in ranked)
        values = (payload.get("image_path", ""), payload.get("waste_type", "-"), f"{float(payload.get('confidence', 0)):.4f}", details.get("rejected", "-"), f"{float(details.get('similarity_score', 0)):.4f}", f"{float(details.get('margin', 0)):.4f}", summary, payload.get("model_error", ""))  # type: ignore[union-attr]
        self.table.insert("", tk.END, values=values)
        self.status_var.set(f"已完成 {len(self.results)} / {len(self.selected_images)}")

    def _reset_table(self) -> None:
        self.results = []
        for item in self.table.get_children():
            self.table.delete(item)

    def save_results(self, show_message: bool = True) -> None:
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps(self.results, ensure_ascii=False, indent=2), encoding="utf-8")
        if show_message:
            messagebox.showinfo("保存完成", f"结果已保存到：\n{OUTPUT_JSON}")

    def open_output_folder(self) -> None:
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        import os
        os.startfile(OUTPUT_JSON.parent)


def main() -> None:
    ModelTesterApp().mainloop()


if __name__ == "__main__":
    main()
