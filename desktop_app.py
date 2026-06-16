import atexit
import math
import queue
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
import shutil
import tkinter as tk
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


APP_TITLE = "RC Finish Analyzer Desktop"
APP_DIR = Path(__file__).resolve().parent
TEMP_DIR = Path(tempfile.mkdtemp(prefix="rc_finish_analyzer_"))
MIN_LAYOUT_WIDTH = 980
CONTROL_MIN_WIDTH = 300
GAP_WIDTH = 18
YOLO_QUEUE_LIMIT = 3
VEHICLE_CLASS_NAMES = {"car", "truck", "bus", "motorcycle"}
MIN_BLOB_AREA = 450
MIN_BLOB_HEIGHT = 22
LINE_CLEAR_FRAMES = 3
MAX_CONFIRMED_EVENTS = 500
TRACK_FEATURE_BINS = 48
TRACK_MATCH_THRESHOLD = 0.22
TRACK_REUSE_MIN_GAP_MS = 1200
WORKER_STOP = object()
YOLO_DOWNLOAD_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"


def cleanup_temp_dir():
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


atexit.register(cleanup_temp_dir)


@dataclass
class DetectionEvent:
    event_id: str
    track_id: str
    rank: int
    time_ms: int
    lap: int
    close_finish: bool
    review_status: str
    yolo_profile: str
    confidence: float
    crop_path: Path
    review_path: Path | None
    feature_vector: tuple[float, ...]
    yolo_boxes: list[tuple[float, float, float, float, float, str]]


@dataclass
class CandidateEvent:
    generation: int
    event_id: str
    batch_id: str
    time_ms: int
    crop_path: Path
    review_path: Path | None


@dataclass(frozen=True)
class YoloResult:
    profile: str
    confidence: float
    executed: bool
    boxes: list[tuple[float, float, float, float, float, str]]


@dataclass(frozen=True)
class AnalysisConfig:
    generation: int
    race_mode: str
    lap_count: int
    direction: str
    yolo_profile: str
    analysis_fps: int
    close_finish_ms: int
    motion_threshold: int
    line_band: int
    finish_line: tuple[tuple[int, int], tuple[int, int]]


@dataclass
class TrackProfile:
    track_id: str
    feature_vector: tuple[float, ...]
    last_seen_ms: int
    lap_count: int


class DesktopApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1540x960")
        self.root.minsize(900, 680)
        self.root.configure(bg="#090b0f")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_path: Path | None = None
        self.cap: cv2.VideoCapture | None = None
        self.preview_frame_bgr: np.ndarray | None = None
        self.preview_photo = None
        self.preview_box = None
        self.viewer_canvas_width = 960
        self.viewer_canvas_height = 540
        self.results_photo_refs = []
        self.close_photo_refs = []
        self.analyze_thread: threading.Thread | None = None
        self.stop_requested = False
        self.main_thread_id = threading.get_ident()
        self.ui_queue: queue.Queue[tuple[int, object]] = queue.Queue()
        self.yolo_queue: queue.Queue[CandidateEvent] = queue.Queue(maxsize=YOLO_QUEUE_LIMIT)
        self.yolo_worker_thread = threading.Thread(target=self.yolo_worker, daemon=True)
        self.pending_yolo_lock = threading.Lock()
        self.pending_yolo_count = 0
        self.analysis_video_done_generation = -1
        self.active_config: AnalysisConfig | None = None
        self.shutdown_requested = False
        self.analysis_generation = 0
        self.run_temp_dir = TEMP_DIR / "run_0"
        self.yolo_models = {}
        self.yolo_class = None
        self.yolo_import_error = None
        self.preview_update_interval_s = 0.16
        self.last_preview_update_s = 0.0

        self.frame_size = (960, 540)
        self.start_line = ((180, 0), (180, 540))
        self.finish_line = ((760, 0), (760, 540))
        self.pending_line_points: list[tuple[int, int]] = []
        self.draw_mode: str | None = None

        self.events: list[DetectionEvent] = []
        self.close_events: list[DetectionEvent] = []
        self.event_by_id: dict[str, DetectionEvent] = {}
        self.track_profiles: dict[str, TrackProfile] = {}
        self.close_event_ids: set[str] = set()
        self.manual_event_ids: set[str] = set()
        self.next_event_id = 1
        self.next_track_id = 1
        self.last_event_time_ms: int | None = None
        self.started = False
        self.started_by: str | None = None
        self.manual_start_armed = False

        self.race_mode = tk.StringVar(value="finish")
        self.lap_count = tk.IntVar(value=3)
        self.start_mode = tk.StringVar(value="auto")
        self.direction = tk.StringVar(value="ltr")
        self.camera_angle = tk.StringVar(value="right_front")
        self.edit_line_target = tk.StringVar(value="finish")
        self.yolo_profile = tk.StringVar(value="yolo26n")
        self.analysis_fps = tk.IntVar(value=60)
        self.close_finish_ms = tk.IntVar(value=50)
        self.motion_threshold = tk.IntVar(value=26)
        self.line_band = tk.IntVar(value=120)

        self.status_text = tk.StringVar(value="動画を選択してください")
        self.auto_state = tk.StringVar(value="動画読込で自動開始")
        self.yolo_state = tk.StringVar(value="未実行")
        self.summary_text = tk.StringVar(value="単純ゴール順 / レース未開始")
        self.metric_events = tk.StringVar(value="0")
        self.metric_close = tk.StringVar(value="0")
        self.metric_manual = tk.StringVar(value="0")

        self.build_ui()
        self.update_summary()
        self.root.after(30, self.process_ui_queue)
        self.yolo_worker_thread.start()

    def build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#090b0f")
        style.configure("Panel.TFrame", background="#12161d", relief="flat")
        style.configure("TLabel", background="#12161d", foreground="#f3f7fb")
        style.configure("Muted.TLabel", background="#12161d", foreground="#8e99a8")
        style.configure("Header.TLabel", background="#090b0f", foreground="#f3f7fb", font=("Helvetica Neue", 28, "bold"))
        style.configure("SubHeader.TLabel", background="#090b0f", foreground="#8e99a8", font=("Helvetica Neue", 11))
        style.configure("MetricValue.TLabel", background="#171c24", foreground="#f3f7fb", font=("Helvetica Neue", 22, "bold"))
        style.configure("MetricLabel.TLabel", background="#171c24", foreground="#8e99a8", font=("Helvetica Neue", 10))
        style.configure("TButton", background="#1a6cff", foreground="#f3f7fb", borderwidth=0, focusthickness=0, padding=8)
        style.map("TButton", background=[("active", "#2b7cff"), ("disabled", "#20242b")], foreground=[("disabled", "#6c7380")])
        style.configure("TCombobox", fieldbackground="#0f1319", background="#0f1319", foreground="#f3f7fb", arrowcolor="#8e99a8")
        style.configure("Treeview", background="#0f1319", fieldbackground="#0f1319", foreground="#f3f7fb", bordercolor="#20242b", rowheight=28)
        style.map("Treeview", background=[("selected", "#1a6cff")], foreground=[("selected", "#f3f7fb")])
        style.configure("Treeview.Heading", background="#171c24", foreground="#b8c2cf", relief="flat")

        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.scroll_canvas = tk.Canvas(self.root, bg="#090b0f", highlightthickness=0)
        self.scroll_canvas.grid(row=0, column=0, sticky="nsew")
        scroll_y = ttk.Scrollbar(self.root, orient="vertical", command=self.scroll_canvas.yview)
        scroll_y.grid(row=0, column=1, sticky="ns")
        scroll_x = ttk.Scrollbar(self.root, orient="horizontal", command=self.scroll_canvas.xview)
        scroll_x.grid(row=1, column=0, sticky="ew")
        self.scroll_canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)

        self.outer = ttk.Frame(self.scroll_canvas, padding=18)
        self.outer_id = self.scroll_canvas.create_window((0, 0), window=self.outer, anchor="nw")
        self.outer.bind("<Configure>", self.on_outer_configure)
        self.scroll_canvas.bind("<Configure>", self.on_canvas_configure)
        self.scroll_canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.scroll_canvas.bind_all("<Shift-MouseWheel>", self.on_shift_mousewheel)
        self.scroll_canvas.bind_all("<Button-4>", self.on_linux_scroll_up)
        self.scroll_canvas.bind_all("<Button-5>", self.on_linux_scroll_down)

        header = ttk.Frame(self.outer)
        header.pack(fill="x")
        ttk.Label(header, text=APP_TITLE, style="Header.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="録画動画のスタート/ゴール検出、順位確認、僅差レビューを行うデスクトップ検証ツール",
            style="SubHeader.TLabel",
        ).pack(anchor="w", pady=(4, 14))

        self.top_frame = ttk.Frame(self.outer)
        self.top_frame.pack(fill="both", expand=True)
        self.top_frame.rowconfigure(0, weight=1)
        self.top_frame.columnconfigure(0, weight=1, minsize=320)
        self.top_frame.columnconfigure(1, weight=3, minsize=640)

        left = ttk.Frame(self.top_frame)
        left.grid(row=0, column=0, sticky="nsew")

        right = ttk.Frame(self.top_frame)
        right.grid(row=0, column=1, sticky="nsew", padx=(18, 0))

        self.build_controls(left)
        self.build_viewer(right)
        self.build_results(self.outer)

    def safe_after(self, delay_ms: int, callback):
        if self.shutdown_requested:
            return
        if threading.get_ident() != self.main_thread_id:
            self.ui_queue.put((delay_ms, callback))
            return
        try:
            self.root.after(delay_ms, callback)
        except tk.TclError:
            return

    def process_ui_queue(self):
        if self.shutdown_requested:
            return
        while True:
            try:
                delay_ms, callback = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            try:
                self.root.after(delay_ms, callback)
            except tk.TclError:
                return
        try:
            self.root.after(30, self.process_ui_queue)
        except tk.TclError:
            return

    def build_controls(self, parent):
        panel = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        panel.pack(fill="both")

        ttk.Label(panel, text="Race Setup", font=("Helvetica Neue", 18, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(panel, text="動画を選ぶと自動で解析します。ラインはプレビュー上で2点クリックして合わせます。", style="Muted.TLabel", wraplength=320).pack(anchor="w", pady=(0, 12))

        self.video_name_label = ttk.Label(panel, text="動画未選択", style="Muted.TLabel", wraplength=320)
        self.video_name_label.pack(anchor="w", pady=(0, 8))
        panel.bind("<Configure>", lambda event: self.video_name_label.configure(wraplength=max(220, event.width - 24)))

        ttk.Button(panel, text="動画を開いて解析", command=self.open_video).pack(fill="x", pady=(0, 8))
        action_row = ttk.Frame(panel, style="Panel.TFrame")
        action_row.pack(fill="x", pady=(0, 12))
        self.analyze_button = ttk.Button(action_row, text="再解析", command=self.start_analysis)
        self.analyze_button.pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(action_row, text="リセット", command=self.reset_state).pack(side="left", fill="x", expand=True, padx=(6, 0))

        form = ttk.Frame(panel)
        form.pack(fill="x", pady=(12, 0))

        self.add_section_title(form, "Line")
        self.add_labeled_combo(
            form,
            "編集するライン",
            self.edit_line_target,
            [("ゴールライン", "finish"), ("スタートライン", "start"), ("編集しない", "none")],
            self.on_edit_line_target_changed,
        )
        self.add_labeled_combo(
            form,
            "カメラ位置",
            self.camera_angle,
            [
                ("車両の右前から見る", "right_front"),
                ("車両の左前から見る", "left_front"),
                ("車両の右後ろから見る", "right_rear"),
                ("車両の左後ろから見る", "left_rear"),
                ("真横", "side"),
            ],
            self.on_camera_angle_changed,
        )
        self.add_labeled_spin(form, "ライン帯域(px)", self.line_band, 10, 320)
        self.add_labeled_scale(form, "差分しきい値", self.motion_threshold, 5, 80)

        self.add_section_title(form, "Race")
        self.add_labeled_combo(form, "レースモード", self.race_mode, [("単純ゴール順", "finish"), ("周回モード", "lap")])
        self.add_labeled_spin(form, "周回数", self.lap_count, 1, 99)
        self.add_labeled_combo(form, "進行方向", self.direction, [("左 → 右", "ltr"), ("右 → 左", "rtl")])

        self.add_section_title(form, "Model")
        self.add_labeled_combo(
            form,
            "YOLOプロファイル",
            self.yolo_profile,
            [("Fast / yolo26n", "yolo26n"), ("Balanced / yolo26s", "yolo26s"), ("Accurate / yolo26m", "yolo26m")],
            self.on_yolo_changed,
        )
        self.add_labeled_spin(form, "解析FPS", self.analysis_fps, 5, 120)
        self.add_labeled_spin(form, "僅差閾値(ms)", self.close_finish_ms, 1, 500)

        status = ttk.Frame(panel, style="Panel.TFrame")
        status.pack(fill="x", pady=(14, 0))
        for label, variable in [
            ("状態", self.status_text),
            ("自動開始", self.auto_state),
            ("YOLO", self.yolo_state),
        ]:
            row = ttk.Frame(status, style="Panel.TFrame")
            row.pack(fill="x", pady=2)
            ttk.Label(row, text=f"{label}:", width=10).pack(side="left")
            ttk.Label(row, textvariable=variable, style="Muted.TLabel", wraplength=260).pack(side="left", fill="x", expand=True)

    def build_viewer(self, parent):
        viewer = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        viewer.pack(fill="both", expand=True)
        ttk.Label(viewer, text="Viewer", font=("Helvetica Neue", 16, "bold")).pack(anchor="w")

        self.canvas = tk.Canvas(viewer, width=960, height=540, bg="#05070a", highlightthickness=0)
        self.canvas.pack(fill="x", expand=False, pady=(10, 0))
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda _event: self.draw_preview())
        self.draw_placeholder()

        legend = ttk.Label(
            viewer,
            text="青: スタートライン  /  桃: ゴールライン  /  黄: ゴール検出帯域",
            style="Muted.TLabel",
        )
        legend.pack(anchor="w", pady=(10, 0))

        summary = ttk.Frame(parent, style="Panel.TFrame", padding=14)
        summary.pack(fill="x", pady=(18, 0))
        ttk.Label(summary, text="Summary", font=("Helvetica Neue", 16, "bold")).pack(anchor="w")

        metrics = ttk.Frame(summary, style="Panel.TFrame")
        metrics.pack(fill="x", pady=(10, 0))
        for label, variable in [
            ("イベント数", self.metric_events),
            ("僅差件数", self.metric_close),
            ("手動確認", self.metric_manual),
        ]:
            card = ttk.Frame(metrics, style="Panel.TFrame", padding=10)
            card.pack(side="left", fill="both", expand=True, padx=(0, 8))
            card.configure(style="Panel.TFrame")
            tk.Frame(card, bg="#171c24").pack(fill="both", expand=True)
            inner = card.winfo_children()[0]
            ttk.Label(inner, text=label, style="MetricLabel.TLabel").pack(anchor="w", padx=10, pady=(10, 2))
            ttk.Label(inner, textvariable=variable, style="MetricValue.TLabel").pack(anchor="w", padx=10, pady=(0, 10))

        ttk.Label(summary, textvariable=self.summary_text, style="Muted.TLabel", wraplength=760).pack(anchor="w", pady=(8, 0))
        summary.bind(
            "<Configure>",
            lambda event: summary.winfo_children()[-1].configure(wraplength=max(280, event.width - 28))
        )

    def build_results(self, parent):
        lower = ttk.Frame(parent)
        lower.pack(fill="both", expand=True, pady=(18, 0))
        lower.columnconfigure(0, weight=1)
        lower.columnconfigure(1, weight=1, minsize=360)
        lower.rowconfigure(0, weight=1)

        results = ttk.Frame(lower, style="Panel.TFrame", padding=14)
        results.grid(row=0, column=0, sticky="nsew")
        ttk.Label(results, text="Rankings", font=("Helvetica Neue", 16, "bold")).pack(anchor="w")

        self.results_tree = ttk.Treeview(
            results,
            columns=("rank", "event", "track", "time", "lap", "flag", "yolo"),
            show="headings",
            height=10,
        )
        for column, title, width in [
            ("rank", "Rank", 60),
            ("event", "Event", 80),
            ("track", "Track", 80),
            ("time", "Time(ms)", 110),
            ("lap", "Lap", 60),
            ("flag", "Review", 120),
            ("yolo", "YOLO", 130),
        ]:
            self.results_tree.heading(column, text=title)
            self.results_tree.column(column, width=width, anchor="center")
        self.results_tree.pack(fill="both", expand=True, pady=(10, 0))
        self.results_tree.bind("<<TreeviewSelect>>", self.on_result_select)

        right = ttk.Frame(lower)
        right.grid(row=0, column=1, sticky="nsew", padx=(18, 0))
        right.rowconfigure(0, weight=1)

        close_panel = ttk.Frame(right, style="Panel.TFrame", padding=14)
        close_panel.pack(fill="both", expand=True)
        ttk.Label(close_panel, text="Detection Review", font=("Helvetica Neue", 16, "bold")).pack(anchor="w")

        self.close_canvas = tk.Canvas(close_panel, width=420, height=330, bg="#0f1319", highlightthickness=0)
        self.close_canvas.pack(fill="x", expand=False, pady=(10, 0))

        self.close_list = tk.Listbox(
            close_panel,
            height=6,
            bg="#0f1319",
            fg="#f3f7fb",
            selectbackground="#1a6cff",
            selectforeground="#f3f7fb",
            highlightthickness=0,
            borderwidth=0,
        )
        self.close_list.pack(fill="x", pady=(10, 0))
        self.close_list.bind("<<ListboxSelect>>", self.on_close_select)

    def add_labeled_combo(self, parent, label, variable, values, on_change=None):
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text=label, style="Muted.TLabel").pack(anchor="w")
        combo = ttk.Combobox(frame, textvariable=variable, state="readonly", values=[text for text, _ in values])
        combo.pack(fill="x")
        reverse = {text: value for text, value in values}
        combo.set(next(text for text, value in values if value == variable.get()))

        def handler(_event=None):
            variable.set(reverse[combo.get()])
            if on_change:
                on_change()

        combo.bind("<<ComboboxSelected>>", handler)

    def add_section_title(self, parent, title):
        ttk.Label(parent, text=title, font=("Helvetica Neue", 13, "bold")).pack(anchor="w", pady=(14, 4))

    def add_labeled_spin(self, parent, label, variable, start, end):
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text=label, style="Muted.TLabel").pack(anchor="w")
        tk.Spinbox(
            frame,
            from_=start,
            to=end,
            textvariable=variable,
            width=10,
            bg="#0f1319",
            fg="#f3f7fb",
            buttonbackground="#171c24",
            highlightthickness=0,
            relief="flat",
        ).pack(fill="x")

    def add_labeled_scale(self, parent, label, variable, start, end):
        frame = ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="x", pady=4)
        ttk.Label(frame, text=label, style="Muted.TLabel").pack(anchor="w")
        tk.Scale(
            frame,
            from_=start,
            to=end,
            orient="horizontal",
            variable=variable,
            bg="#12161d",
            fg="#f3f7fb",
            troughcolor="#0b0f14",
            highlightthickness=0,
            activebackground="#1a6cff",
        ).pack(fill="x")

    def on_outer_configure(self, _event=None):
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def on_canvas_configure(self, event):
        target_width = max(event.width, MIN_LAYOUT_WIDTH)
        self.scroll_canvas.itemconfigure(self.outer_id, width=target_width)
        self.resize_responsive_regions(target_width)
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def resize_responsive_regions(self, content_width):
        inner_width = max(MIN_LAYOUT_WIDTH, content_width - 36)
        controls_width = max(CONTROL_MIN_WIDTH, min(420, int(inner_width * 0.28)))
        viewer_width = max(520, inner_width - controls_width - GAP_WIDTH)
        canvas_width = max(480, viewer_width - 28)
        canvas_height = max(270, min(680, int(canvas_width * 9 / 16)))
        if hasattr(self, "top_frame"):
            self.top_frame.columnconfigure(0, minsize=controls_width)
            self.top_frame.columnconfigure(1, minsize=viewer_width)
        self.viewer_canvas_width = canvas_width
        self.viewer_canvas_height = canvas_height
        if hasattr(self, "canvas"):
            self.canvas.configure(width=canvas_width, height=canvas_height)
            self.draw_preview()
        if hasattr(self, "close_canvas"):
            close_width = max(320, min(520, int(inner_width * 0.32)))
            self.close_canvas.configure(width=close_width, height=max(260, int(close_width * 0.72)))

    def on_mousewheel(self, event):
        if event.state & 0x0001:
            self.scroll_by_pixels("x", -event.delta)
            return
        self.scroll_by_pixels("y", -event.delta)

    def on_shift_mousewheel(self, event):
        self.scroll_by_pixels("x", -event.delta)

    def on_linux_scroll_up(self, _event):
        self.scroll_canvas.yview_scroll(-3, "units")

    def on_linux_scroll_down(self, _event):
        self.scroll_canvas.yview_scroll(3, "units")

    def scroll_by_pixels(self, axis: str, delta: int):
        bbox = self.scroll_canvas.bbox("all")
        if not bbox:
            return
        if axis == "y":
            content_size = max(1, bbox[3] - bbox[1])
            viewport_size = max(1, self.scroll_canvas.winfo_height())
            current = self.scroll_canvas.yview()[0]
            next_pos = current + (delta / max(content_size - viewport_size, 1))
            self.scroll_canvas.yview_moveto(min(1.0, max(0.0, next_pos)))
            return
        content_size = max(1, bbox[2] - bbox[0])
        viewport_size = max(1, self.scroll_canvas.winfo_width())
        current = self.scroll_canvas.xview()[0]
        next_pos = current + (delta / max(content_size - viewport_size, 1))
        self.scroll_canvas.xview_moveto(min(1.0, max(0.0, next_pos)))

    def set_draw_mode(self, mode: str):
        self.draw_mode = mode
        self.pending_line_points = []
        self.status_text.set("プレビュー上でラインの始点と終点を順にクリックしてください")

    def on_canvas_click(self, event):
        if self.preview_frame_bgr is None or self.preview_box is None:
            return
        target = self.draw_mode or self.edit_line_target.get()
        if target not in {"start", "finish"}:
            return
        point = self.canvas_to_frame_point(event.x, event.y)
        if point is None:
            return
        self.pending_line_points.append(point)
        if len(self.pending_line_points) == 1:
            self.status_text.set("終点をクリックしてください")
            return
        line = (self.pending_line_points[0], self.pending_line_points[1])
        if target == "start":
            self.start_line = line
        elif target == "finish":
            self.finish_line = line
        self.pending_line_points = []
        self.draw_mode = None
        self.draw_preview()
        self.status_text.set("ラインを更新しました")

    def canvas_to_frame_point(self, canvas_x: int, canvas_y: int):
        box_x, box_y, box_w, box_h = self.preview_box
        if not (box_x <= canvas_x <= box_x + box_w and box_y <= canvas_y <= box_y + box_h):
            return None
        img_width, img_height = self.frame_size
        x = (canvas_x - box_x) * (img_width / max(box_w, 1e-6))
        y = (canvas_y - box_y) * (img_height / max(box_h, 1e-6))
        return (
            int(max(0, min(img_width - 1, x))),
            int(max(0, min(img_height - 1, y))),
        )

    def on_start_mode_changed(self):
        self.manual_start_armed = False
        self.auto_state.set("動画読込で自動開始")

    def on_edit_line_target_changed(self):
        self.pending_line_points = []
        self.draw_mode = None
        if self.edit_line_target.get() == "none":
            self.status_text.set("ライン編集を停止しました")
            return
        label = "ゴールライン" if self.edit_line_target.get() == "finish" else "スタートライン"
        self.status_text.set(f"{label}をプレビュー上で2点クリックして調整できます")

    def on_yolo_changed(self):
        self.yolo_state.set(f"{self.yolo_profile.get()} selected")
        self.update_summary()

    def on_camera_angle_changed(self):
        self.reset_default_lines()
        self.draw_preview()
        self.status_text.set(f"カメラ位置を更新しました: {self.camera_angle_label()}")

    def camera_angle_label(self):
        labels = {
            "right_front": "車両の右前から見る",
            "left_front": "車両の左前から見る",
            "right_rear": "車両の右後ろから見る",
            "left_rear": "車両の左後ろから見る",
            "side": "真横",
        }
        return labels.get(self.camera_angle.get(), self.camera_angle.get())

    def open_video(self):
        path = filedialog.askopenfilename(
            title="動画を選択",
            filetypes=[("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv"), ("All files", "*.*")],
        )
        if not path:
            return
        self.load_video(Path(path))

    def load_video(self, path: Path):
        try:
            self.stop_analysis(wait=True)
            if self.cap is not None:
                self.cap.release()
            self.video_path = path
            self.cap = cv2.VideoCapture(str(path))
            ok, frame = self.cap.read()
            if not ok or frame is None or frame.size == 0:
                self.status_text.set("動画を読み込めませんでした")
                return
            self.reset_runtime_only()
            self.frame_size = (frame.shape[1], frame.shape[0])
            self.reset_default_lines()
            self.preview_frame_bgr = frame
            self.video_name_label.configure(text=path.name)
            self.status_text.set(f"動画読込完了、自動解析開始: {path.name}")
            self.draw_preview()
            self.safe_after(150, self.start_analysis)
        except Exception as exc:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.video_path = None
            self.status_text.set(f"動画読込でエラー: {type(exc).__name__}")

    def draw_placeholder(self):
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), self.viewer_canvas_width)
        height = max(self.canvas.winfo_height(), self.viewer_canvas_height)
        self.canvas.create_rectangle(0, 0, width, height, fill="#05070a", outline="")
        self.canvas.create_text(width / 2, min(140, height / 3), text="動画を読み込むとここにプレビューを表示します", fill="#dbe6f2", font=("Helvetica Neue", 24))
        self.preview_box = (0, 0, width, height)
        self.draw_line_overlay(960, 540)

    def draw_line_overlay(self, width, height):
        if self.preview_box is None:
            canvas_w = max(self.canvas.winfo_width(), self.viewer_canvas_width)
            canvas_h = max(self.canvas.winfo_height(), self.viewer_canvas_height)
            self.preview_box = (0, 0, canvas_w, canvas_h)
        self.draw_line_band(self.finish_line, width, height, "#e8b94a")
        self.draw_line(self.start_line, width, height, "#5aa2ff")
        self.draw_line(self.finish_line, width, height, "#ff5db1")

    def frame_to_canvas_point(self, point, width, height):
        box_x, box_y, box_w, box_h = self.preview_box
        x, y = point
        return (
            box_x + x * (box_w / max(width, 1)),
            box_y + y * (box_h / max(height, 1)),
        )

    def draw_line(self, line, width, height, color):
        p1 = self.frame_to_canvas_point(line[0], width, height)
        p2 = self.frame_to_canvas_point(line[1], width, height)
        self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill=color, width=3)

    def draw_line_band(self, line, width, height, color):
        p1 = self.frame_to_canvas_point(line[0], width, height)
        p2 = self.frame_to_canvas_point(line[1], width, height)
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1)
        nx = -dy / length
        ny = dx / length
        band = self.line_band.get() * (self.preview_box[2] / max(width, 1))
        points = [
            p1[0] + nx * band / 2, p1[1] + ny * band / 2,
            p2[0] + nx * band / 2, p2[1] + ny * band / 2,
            p2[0] - nx * band / 2, p2[1] - ny * band / 2,
            p1[0] - nx * band / 2, p1[1] - ny * band / 2,
        ]
        self.canvas.create_polygon(points, fill=color, stipple="gray25", outline="")

    def reset_default_lines(self):
        width, height = self.frame_size
        self.start_line = self.default_line_for_angle(width, height, start=True)
        self.finish_line = self.default_line_for_angle(width, height, start=False)

    def default_line_for_angle(self, width, height, start=False):
        # These presets approximate finish/start lines seen from a low,
        # diagonal trackside camera. Users should still align the two points
        # to the actual painted line in the video.
        y_near = int(height * (0.76 if not start else 0.58))
        y_far = int(height * (0.58 if not start else 0.43))
        angle = self.camera_angle.get()
        if angle == "right_front":
            return ((0, y_far), (width - 1, y_near))
        if angle == "left_front":
            return ((0, y_near), (width - 1, y_far))
        if angle == "right_rear":
            return ((0, y_near), (width - 1, y_far))
        if angle == "left_rear":
            return ((0, y_far), (width - 1, y_near))
        y = int(height * (0.70 if not start else 0.50))
        return ((0, y), (width - 1, y))

    def draw_preview(self):
        if self.preview_frame_bgr is None:
            self.draw_placeholder()
            return
        image = cv2.cvtColor(self.preview_frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image)
        width = max(self.canvas.winfo_width(), self.viewer_canvas_width)
        height = max(self.canvas.winfo_height(), self.viewer_canvas_height)
        src_w, src_h = pil.size
        scale = min(width / src_w, height / src_h)
        display_w = max(1, int(src_w * scale))
        display_h = max(1, int(src_h * scale))
        pil = pil.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        offset_x = (width - display_w) // 2
        offset_y = (height - display_h) // 2
        self.preview_box = (offset_x, offset_y, display_w, display_h)
        self.canvas.create_rectangle(0, 0, width, height, fill="#05070a", outline="")
        self.canvas.create_image(offset_x, offset_y, image=self.preview_photo, anchor="nw")
        self.draw_line_overlay(src_w, src_h)

    def reset_runtime_only(self):
        self.analysis_generation += 1
        self.cleanup_run_files()
        self.run_temp_dir = TEMP_DIR / f"run_{self.analysis_generation}"
        self.run_temp_dir.mkdir(parents=True, exist_ok=True)
        with self.pending_yolo_lock:
            self.pending_yolo_count = 0
        self.analysis_video_done_generation = -1
        self.active_config = None
        self.events.clear()
        self.close_events.clear()
        self.event_by_id.clear()
        self.track_profiles.clear()
        self.close_event_ids.clear()
        self.manual_event_ids.clear()
        self.next_event_id = 1
        self.next_track_id = 1
        self.last_event_time_ms = None
        self.started = False
        self.started_by = None
        self.manual_start_armed = False
        self.close_photo_refs = []
        self.metric_events.set("0")
        self.metric_close.set("0")
        self.metric_manual.set("0")
        self.results_tree.delete(*self.results_tree.get_children())
        self.close_list.delete(0, "end")
        self.close_canvas.delete("all")
        self.auto_state.set("動画読込で自動開始")
        self.yolo_state.set("未実行")
        self.drain_yolo_queue()
        self.update_summary()
        self.close_canvas.create_text(210, 165, text="僅差イベントがここに表示されます", fill="#8e99a8", font=("Helvetica Neue", 14))

    def cleanup_run_files(self):
        for path in TEMP_DIR.glob("run_*"):
            shutil.rmtree(path, ignore_errors=True)

    def reset_state(self):
        self.stop_analysis(wait=True)
        self.reset_runtime_only()
        self.status_text.set("リセットしました")

    def update_summary(self):
        mode = "単純ゴール順" if self.race_mode.get() == "finish" else f"周回 {self.lap_count.get()}周"
        if self.events:
            self.summary_text.set(f"{mode} / 開始: {self.started_by or '未開始'} / プロファイル: {self.yolo_profile.get()}")
        else:
            self.summary_text.set(f"{mode} / レース未開始")

    def start_analysis(self):
        if self.video_path is None:
            self.status_text.set("先に動画を選択してください")
            return
        if self.analyze_thread and self.analyze_thread.is_alive():
            return
        self.reset_runtime_only()
        self.stop_requested = False
        self.active_config = self.snapshot_config()
        self.analyze_button.state(["disabled"])
        self.status_text.set("解析中...")
        self.yolo_state.set("YOLO待機中")
        self.analyze_thread = threading.Thread(target=self.analyze_worker, args=(self.active_config,), daemon=True)
        self.analyze_thread.start()

    def snapshot_config(self):
        return AnalysisConfig(
            generation=self.analysis_generation,
            race_mode=self.race_mode.get(),
            lap_count=self.lap_count.get(),
            direction=self.direction.get(),
            yolo_profile=self.yolo_profile.get(),
            analysis_fps=max(5, self.analysis_fps.get()),
            close_finish_ms=self.close_finish_ms.get(),
            motion_threshold=self.motion_threshold.get(),
            line_band=self.line_band.get(),
            finish_line=self.finish_line,
        )

    def stop_analysis(self, wait: bool):
        self.stop_requested = True
        thread = self.analyze_thread
        if thread and thread.is_alive() and wait:
            thread.join(timeout=1.5)
        if thread and not thread.is_alive():
            self.analyze_thread = None

    def analyze_worker(self, config: AnalysisConfig):
        cap = None
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            fps_native = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_interval = max(1, int(round(fps_native / config.analysis_fps)))

            prev_finish = None
            frame_index = 0
            line_occupied = False
            clear_frames = LINE_CLEAR_FRAMES

            self.started = True
            self.started_by = "video"
            self.safe_after(0, lambda: self.auto_state.set("動画読込で自動開始"))

            while not self.stop_requested:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                if frame_index % frame_interval != 0:
                    continue

                current_ms = int((frame_index / fps_native) * 1000)
                now = time.monotonic()
                if now - self.last_preview_update_s >= self.preview_update_interval_s:
                    self.last_preview_update_s = now
                    self.preview_frame_bgr = frame.copy()
                    self.safe_after(0, self.draw_preview)

                finish_band = self.extract_line_region(frame, config.finish_line, config.line_band)
                diff = self.compute_diff(prev_finish, finish_band)
                prev_finish = finish_band
                if diff is None:
                    continue

                blobs = self.build_blobs(diff, config.motion_threshold)
                if not blobs:
                    clear_frames += 1
                    if clear_frames >= LINE_CLEAR_FRAMES:
                        line_occupied = False
                    continue

                if line_occupied:
                    continue

                line_occupied = True
                clear_frames = 0
                self.submit_candidates(frame, blobs, current_ms, config)
        except Exception as exc:
            self.safe_after(0, lambda: self.status_text.set(f"解析中エラー: {type(exc).__name__}"))
        finally:
            if cap is not None:
                cap.release()
            self.analysis_video_done_generation = config.generation
            self.safe_after(0, lambda g=config.generation: self.maybe_finish_analysis(g))

    def maybe_finish_analysis(self, generation: int | None = None):
        config = self.active_config
        if config is None:
            return
        if generation is not None and config.generation != generation:
            return
        if self.analysis_video_done_generation != config.generation:
            return
        with self.pending_yolo_lock:
            if self.pending_yolo_count != 0:
                return
        self.analyze_button.state(["!disabled"])
        self.status_text.set(f"解析完了: 確定 {len(self.events)} events")
        self.update_summary()

    def submit_candidates(self, frame: np.ndarray, blobs: list[dict], current_ms: int, config: AnalysisConfig):
        if self.yolo_queue.full():
            self.safe_after(0, lambda: self.yolo_state.set("YOLO待ち行列が満杯: 候補破棄"))
            return

        batch_id = f"B{config.generation:03d}_{current_ms:08d}"
        for blob in self.sort_blobs_for_finish(blobs, config.finish_line, config.direction):
            if self.yolo_queue.full():
                break
            event_id = f"E{self.next_event_id:04d}"
            self.next_event_id += 1
            crop_path = self.save_crop(frame, blob, event_id, config.generation)
            if crop_path is None:
                self.safe_after(0, lambda eid=event_id: self.status_text.set(f"画像保存失敗のため候補を破棄: {eid}"))
                continue
            review_path = self.save_review_image(frame, event_id, config.generation) if len(blobs) > 1 else None
            if len(blobs) > 1 and review_path is None:
                self.safe_unlink(crop_path)
                self.safe_after(0, lambda eid=event_id: self.status_text.set(f"レビュー画像保存失敗のため候補を破棄: {eid}"))
                continue
            candidate = CandidateEvent(
                generation=config.generation,
                event_id=event_id,
                batch_id=batch_id,
                time_ms=current_ms,
                crop_path=crop_path,
                review_path=review_path,
            )
            try:
                self.yolo_queue.put_nowait(candidate)
                with self.pending_yolo_lock:
                    self.pending_yolo_count += 1
            except queue.Full:
                self.safe_unlink(crop_path)
                self.safe_unlink(review_path)
                break

    def sort_blobs_for_finish(self, blobs: list[dict], line, direction: str):
        return sorted(
            blobs,
            key=lambda blob: (-self.finish_progress_score(blob, line, direction), -blob["area"]),
        )

    def drain_yolo_queue(self):
        while True:
            try:
                item = self.yolo_queue.get_nowait()
                if isinstance(item, CandidateEvent):
                    self.delete_candidate_files(item)
            except queue.Empty:
                return

    def decrement_pending_yolo(self):
        with self.pending_yolo_lock:
            self.pending_yolo_count = max(0, self.pending_yolo_count - 1)
        self.safe_after(0, self.maybe_finish_analysis)

    def finish_progress_score(self, blob: dict, line, direction: str):
        center = (float(blob["center_x"]), float(blob["center_y"]))
        side = self.line_side(line, center)
        score = -side if direction == "ltr" else side
        return score / max(1.0, math.sqrt(blob["area"]))

    def yolo_worker(self):
        while True:
            candidate = self.yolo_queue.get()
            if candidate is WORKER_STOP:
                return
            if candidate.generation != self.analysis_generation or self.stop_requested:
                self.delete_candidate_files(candidate)
                self.decrement_pending_yolo()
                continue
            yolo_result = self.run_yolo(candidate.crop_path)
            if candidate.generation != self.analysis_generation or self.stop_requested:
                self.delete_candidate_files(candidate)
                self.decrement_pending_yolo()
                continue
            if not yolo_result.executed or yolo_result.confidence <= 0.0:
                self.delete_candidate_files(candidate)
                self.decrement_pending_yolo()
                continue
            event = self.register_confirmed_event(candidate, yolo_result)
            if event is not None:
                self.safe_after(0, lambda e=event: self.finalize_event_ui(e))
            else:
                self.decrement_pending_yolo()

    def extract_line_region(self, frame: np.ndarray, line, band: int):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.line(mask, line[0], line[1], 255, max(1, int(band)))
        masked = cv2.bitwise_and(gray, gray, mask=mask)
        return masked

    def compute_diff(self, previous: np.ndarray | None, current: np.ndarray):
        if previous is None or previous.shape != current.shape:
            return None
        return cv2.absdiff(previous, current)

    def build_blobs(self, diff: np.ndarray, motion_threshold: int):
        _, mask = cv2.threshold(diff, motion_threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        blobs = []
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < MIN_BLOB_AREA or h < MIN_BLOB_HEIGHT:
                continue
            cx, cy = centroids[label]
            blobs.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "center_x": cx,
                "center_y": cy,
                "center": (float(cx), float(cy)),
                "area": area,
            })
        return blobs

    def is_crossing_line(self, line, previous_point, next_point):
        prev_side = self.line_side(line, previous_point)
        next_side = self.line_side(line, next_point)
        if abs(prev_side) < 1 or abs(next_side) < 1:
            return True
        if prev_side * next_side > 0:
            return False
        if self.direction.get() == "ltr":
            return prev_side < next_side
        return prev_side > next_side

    def line_side(self, line, point):
        (x1, y1), (x2, y2) = line
        px, py = point
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

    def run_yolo(self, image_path: Path):
        config = self.active_config
        profile = config.yolo_profile if config is not None else "yolo26n"
        try:
            effective_profile, model = self.get_yolo_model(profile)
        except Exception as exc:
            message = f"YOLO unavailable: {type(exc).__name__}: {exc}"
            self.safe_after(0, lambda msg=message: self.yolo_state.set(msg))
            return YoloResult(profile, 0.0, False, [])
        try:
            results = model.predict(str(image_path), verbose=False, imgsz=960, conf=0.08)
            boxes = results[0].boxes if results else None
            if boxes is None or len(boxes) == 0:
                confidence = 0.0
                yolo_boxes = []
            else:
                confidence, yolo_boxes = self.extract_yolo_boxes(results[0])
            self.safe_after(0, lambda p=effective_profile, count=len(yolo_boxes), conf=confidence: self.yolo_state.set(f"{p} boxes={count} conf={conf:.2f}"))
            return YoloResult(effective_profile, confidence, True, yolo_boxes)
        except Exception as exc:
            message = f"YOLO failed: {type(exc).__name__}"
            self.safe_after(0, lambda msg=message: self.yolo_state.set(msg))
            return YoloResult(profile, 0.0, False, [])

    def get_yolo_model(self, profile: str):
        if self.yolo_class is None:
            try:
                from ultralytics import YOLO as UltralyticsYOLO
                self.yolo_class = UltralyticsYOLO
            except Exception as exc:
                self.yolo_import_error = exc
                raise
        weights_path = self.ensure_yolo_weights(profile)
        if profile not in self.yolo_models:
            self.yolo_models[profile] = self.yolo_class(str(weights_path))
        return profile, self.yolo_models[profile]

    def ensure_yolo_weights(self, profile: str):
        weights_path = APP_DIR / f"{profile}.pt"
        if weights_path.exists() and weights_path.stat().st_size > 0:
            return weights_path

        url = f"{YOLO_DOWNLOAD_BASE}/{profile}.pt"
        temp_path = weights_path.with_suffix(".pt.download")
        self.safe_after(0, lambda p=profile: self.yolo_state.set(f"{p}.pt を取得中"))
        try:
            urllib.request.urlretrieve(url, temp_path)
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise FileNotFoundError(f"{profile}.pt の取得に失敗しました")
            temp_path.replace(weights_path)
            self.safe_after(0, lambda p=profile: self.yolo_state.set(f"{p}.pt 取得完了"))
            return weights_path
        except Exception:
            self.safe_unlink(temp_path)
            raise

    def extract_yolo_boxes(self, result):
        names = getattr(result, "names", {}) or {}
        vehicle_boxes = []
        fallback_boxes = []
        for xyxy, cls_id, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            name = str(names.get(int(cls_id.item()), "")).lower()
            box = (
                float(xyxy[0].item()),
                float(xyxy[1].item()),
                float(xyxy[2].item()),
                float(xyxy[3].item()),
                float(conf.item()),
                name or "object",
            )
            fallback_boxes.append(box)
            if (
                name in VEHICLE_CLASS_NAMES
                or "car" in name
                or "truck" in name
                or "bus" in name
                or "motor" in name
                or "vehicle" in name
            ):
                vehicle_boxes.append(box)
        boxes = vehicle_boxes or fallback_boxes
        if boxes:
            return max(box[4] for box in boxes), boxes
        if result.boxes is None or len(result.boxes) == 0:
            return 0.0, []
        return float(result.boxes.conf.max().item()), fallback_boxes

    def register_confirmed_event(self, candidate: CandidateEvent, yolo_result: YoloResult):
        if len(self.events) >= MAX_CONFIRMED_EVENTS:
            self.delete_candidate_files(candidate)
            self.safe_after(0, lambda: self.status_text.set("確定イベント上限に達したため追加を停止しました"))
            return None

        config = self.active_config
        if config is None:
            self.delete_candidate_files(candidate)
            return None

        feature_vector = self.extract_feature_vector(candidate.crop_path)
        self.draw_yolo_boxes(candidate.crop_path, yolo_result.boxes)
        track_id, lap = self.assign_track(feature_vector, candidate.time_ms, config)
        close_finish = self.update_close_finish_state(candidate.time_ms, config.close_finish_ms)
        review_status = "manual_review" if config.race_mode == "lap" and close_finish else "auto"

        event = DetectionEvent(
            event_id=candidate.event_id,
            track_id=track_id,
            rank=len(self.events) + 1,
            time_ms=candidate.time_ms,
            lap=lap,
            close_finish=close_finish,
            review_status=review_status,
            yolo_profile=yolo_result.profile,
            confidence=yolo_result.confidence,
            crop_path=candidate.crop_path,
            review_path=candidate.review_path,
            feature_vector=feature_vector,
            yolo_boxes=yolo_result.boxes,
        )
        self.events.append(event)
        self.event_by_id[event.event_id] = event
        self.apply_close_finish(event, close_finish, config.race_mode == "lap")
        return event

    def assign_track(self, feature_vector: tuple[float, ...], time_ms: int, config: AnalysisConfig):
        if config.race_mode != "lap":
            track_id = f"T{self.next_track_id:03d}"
            self.next_track_id += 1
            return track_id, 1

        best_track: TrackProfile | None = None
        best_distance = float("inf")
        for profile in self.track_profiles.values():
            if time_ms - profile.last_seen_ms < TRACK_REUSE_MIN_GAP_MS:
                continue
            distance = self.feature_distance(feature_vector, profile.feature_vector)
            if distance < TRACK_MATCH_THRESHOLD and distance < best_distance:
                best_track = profile
                best_distance = distance

        if best_track is None:
            track_id = f"T{self.next_track_id:03d}"
            self.next_track_id += 1
            self.track_profiles[track_id] = TrackProfile(
                track_id=track_id,
                feature_vector=feature_vector,
                last_seen_ms=time_ms,
                lap_count=1,
            )
            return track_id, 1

        best_track.feature_vector = feature_vector
        best_track.last_seen_ms = time_ms
        best_track.lap_count = min(config.lap_count, best_track.lap_count + 1)
        return best_track.track_id, best_track.lap_count

    def feature_distance(self, left: tuple[float, ...], right: tuple[float, ...]):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def extract_feature_vector(self, image_path: Path):
        image = cv2.imread(str(image_path))
        if image is None:
            return tuple(0.0 for _ in range(TRACK_FEATURE_BINS))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [12, 4], [0, 180, 0, 256]).flatten()
        norm = float(np.linalg.norm(hist))
        if norm > 0:
            hist = hist / norm
        return tuple(float(value) for value in hist.tolist())

    def update_close_finish_state(self, time_ms: int, close_finish_ms: int):
        return any(abs(time_ms - event.time_ms) <= close_finish_ms for event in self.events)

    def apply_close_finish(self, event: DetectionEvent, is_close: bool, lap_mode: bool):
        if not is_close:
            return
        event.close_finish = True
        if lap_mode:
            event.review_status = "manual_review"
        for previous in self.events:
            if previous.event_id == event.event_id:
                continue
            if abs(event.time_ms - previous.time_ms) > (self.active_config.close_finish_ms if self.active_config else 0):
                continue
            previous.close_finish = True
            if lap_mode:
                previous.review_status = "manual_review"
            self.safe_after(0, lambda e=previous: self.refresh_event_ui(e))

    def save_crop(self, frame: np.ndarray, blob: dict, event_id: str, generation: int):
        h, w = frame.shape[:2]
        crop_w = min(w, max(420, int(blob["w"] * 4.0)))
        crop_h = min(h, max(300, int(blob["h"] * 4.0)))
        x = max(0, min(w - crop_w, int(blob["center_x"] - crop_w // 2)))
        y = max(0, min(h - crop_h, int(blob["center_y"] - crop_h / 2)))
        crop = frame[y:y + crop_h, x:x + crop_w]
        path = self.candidate_file_path(generation, f"{event_id}_crop.jpg")
        try:
            ok = cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 86])
        except Exception:
            self.safe_unlink(path)
            return None
        if not ok or not path.exists() or path.stat().st_size == 0:
            self.safe_unlink(path)
            return None
        return path

    def draw_yolo_boxes(self, image_path: Path, boxes: list[tuple[float, float, float, float, float, str]]):
        image = cv2.imread(str(image_path))
        if image is None:
            return
        for x1, y1, x2, y2, conf, label in boxes:
            p1 = (int(max(0, x1)), int(max(0, y1)))
            p2 = (int(min(image.shape[1] - 1, x2)), int(min(image.shape[0] - 1, y2)))
            cv2.rectangle(image, p1, p2, (0, 255, 180), 3)
            text = f"{label} {conf:.2f}"
            text_y = max(18, p1[1] - 8)
            cv2.putText(image, text, (p1[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2, cv2.LINE_AA)
        cv2.imwrite(str(image_path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])

    def save_review_image(self, frame: np.ndarray, event_id: str, generation: int):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.thumbnail((960, 540))
        path = self.candidate_file_path(generation, f"{event_id}_review.jpg")
        try:
            image.save(path, "JPEG", quality=82)
        except Exception:
            self.safe_unlink(path)
            return None
        if not path.exists() or path.stat().st_size == 0:
            self.safe_unlink(path)
            return None
        return path

    def candidate_file_path(self, generation: int, filename: str):
        directory = TEMP_DIR / f"run_{generation}"
        directory.mkdir(parents=True, exist_ok=True)
        return directory / filename

    def delete_candidate_files(self, candidate: CandidateEvent):
        self.safe_unlink(candidate.crop_path)
        self.safe_unlink(candidate.review_path)

    def safe_unlink(self, path: Path | None):
        if path is not None:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def append_event_ui(self, event: DetectionEvent):
        self.metric_events.set(str(len(self.events)))
        self.results_tree.insert("", "end", iid=event.event_id, values=(
            event.rank, event.event_id, event.track_id, event.time_ms, event.lap, "", f"{event.yolo_profile} {event.confidence:.2f}"
        ))
        self.refresh_event_ui(event)
        self.yolo_state.set(f"{event.yolo_profile} confirmed {event.confidence:.2f}")
        self.update_summary()

    def finalize_event_ui(self, event: DetectionEvent):
        self.append_event_ui(event)
        if event.crop_path.exists():
            self.show_close_image(event.crop_path)
        self.decrement_pending_yolo()

    def refresh_event_ui(self, event: DetectionEvent):
        if not self.results_tree.exists(event.event_id):
            return
        values = list(self.results_tree.item(event.event_id, "values"))
        if len(values) >= 7:
            flag = event.review_status if event.review_status != "auto" else ("close_finish" if event.close_finish else "auto")
            values[5] = flag
            values[6] = f"{event.yolo_profile} {event.confidence:.2f}"
            self.results_tree.item(event.event_id, values=values)
        if event.close_finish and event.event_id not in self.close_event_ids:
            self.close_event_ids.add(event.event_id)
            self.close_events.append(event)
            self.close_list.insert("end", f"{event.event_id} / {event.track_id} / {event.time_ms}ms")
            self.metric_close.set(str(len(self.close_event_ids)))
        if event.review_status == "manual_review" and event.event_id not in self.manual_event_ids:
            self.manual_event_ids.add(event.event_id)
            self.metric_manual.set(str(len(self.manual_event_ids)))

    def on_close_select(self, _event=None):
        selection = self.close_list.curselection()
        if not selection:
            return
        event = self.close_events[selection[0]]
        if event.review_path is not None and event.review_path.exists():
            self.show_close_image(event.review_path)
        elif event.crop_path.exists():
            self.show_close_image(event.crop_path)

    def on_result_select(self, _event=None):
        selection = self.results_tree.selection()
        if not selection:
            return
        event = self.event_by_id.get(selection[0])
        if event is None:
            return
        if event.crop_path.exists():
            self.show_close_image(event.crop_path)

    def show_close_image(self, image_path: Path):
        self.close_canvas.delete("all")
        with Image.open(image_path) as image:
            preview = image.copy()
            preview.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(preview)
        self.close_photo_refs = [photo]
        x = max(1, self.close_canvas.winfo_width()) // 2
        y = max(1, self.close_canvas.winfo_height()) // 2
        self.close_canvas.create_image(x, y, image=photo)

    def on_close(self):
        self.shutdown_requested = True
        self.stop_analysis(wait=True)
        self.drain_ui_queue()
        try:
            self.yolo_queue.put_nowait(WORKER_STOP)
        except queue.Full:
            pass
        if self.cap is not None:
            self.cap.release()
        cleanup_temp_dir()
        self.root.destroy()

    def drain_ui_queue(self):
        while True:
            try:
                self.ui_queue.get_nowait()
            except queue.Empty:
                return


def main():
    root = tk.Tk()
    app = DesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
