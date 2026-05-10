import json
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

import cv2

TRASH_CLASSES = ["plastic", "metal", "paper", "glass", "organic", "generic"]
DEFAULT_BIN_SETTINGS = {
    "bin_floor_distance": 1000.0,
    "empty_threshold_percent": 80.0,
}
DEFAULT_EXPECTED_CLASS = "generic"


@dataclass
class DashboardState:
    frame_bgr: Optional[Any] = None
    detections: List[Dict[str, Any]] = field(default_factory=list)
    depth_mm: Optional[float] = None
    updated_at: float = 0.0


class DashboardStore:
    def __init__(self):
        self._state = DashboardState()
        self._lock = threading.Lock()

    def update_frame(self, frame_bgr: Any) -> None:
        with self._lock:
            self._state.frame_bgr = frame_bgr
            self._state.updated_at = time.time()

    def update_detections(self, detections: List[Dict[str, Any]]) -> None:
        with self._lock:
            self._state.detections = detections
            self._state.updated_at = time.time()

    def update_depth(self, depth_mm: Optional[float]) -> None:
        with self._lock:
            self._state.depth_mm = depth_mm
            self._state.updated_at = time.time()

    def snapshot(self) -> DashboardState:
        with self._lock:
            return DashboardState(
                frame_bgr=self._state.frame_bgr,
                detections=list(self._state.detections),
                depth_mm=self._state.depth_mm,
                updated_at=self._state.updated_at,
            )


def _annotate_frame(
    frame,
    detections: List[Dict[str, Any]],
    expected_class: Optional[str] = None,
    highlight_wrong_only: bool = False,
):
    height, width = frame.shape[:2]
    selected_class = normalize_expected_class(expected_class)
    for detection in detections:
      yolo_label = detection.get("yolo_label")
      trash_label = detection.get("trash_label")
      display_label = yolo_label or trash_label or detection.get("label") or "unknown"
      is_wrong = expected_class is not None and trash_label != selected_class
      if highlight_wrong_only and not is_wrong:
        continue

      x1 = max(0, min(width - 1, int(detection["xmin"] * width)))
      y1 = max(0, min(height - 1, int(detection["ymin"] * height)))
      x2 = max(0, min(width - 1, int(detection["xmax"] * width)))
      y2 = max(0, min(height - 1, int(detection["ymax"] * height)))

      color = (92, 231, 167) if not is_wrong else (74, 74, 255)
      thickness = 2 if not is_wrong else 3
      cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
      label_text = f"{display_label} {int(detection.get('confidence', 0) * 100)}%"
      if yolo_label and trash_label:
        label_text = f"{label_text} -> {trash_label}"
      if is_wrong:
        label_text = f"{label_text} wrong"
      (text_width, text_height), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
      )
      top = max(0, y1 - text_height - baseline - 8)
      cv2.rectangle(
        frame,
        (x1, top),
        (x1 + text_width + 10, top + text_height + baseline + 8),
        (17, 24, 39),
        -1,
      )
      cv2.putText(
        frame,
        label_text,
        (x1 + 5, top + text_height + 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (235, 243, 255),
        2,
      )

    return frame


def _encode_annotated_frame(
    state: DashboardState,
    expected_class: Optional[str],
    highlight_wrong_only: bool,
) -> Optional[bytes]:
    if state.frame_bgr is None:
        return None
    frame = state.frame_bgr.copy()
    annotated = _annotate_frame(
        frame,
        state.detections,
        expected_class=expected_class,
        highlight_wrong_only=highlight_wrong_only,
    )
    success, encoded = cv2.imencode(
        ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 82]
    )
    if not success:
        return None
    return encoded.tobytes()


def _default_config_path() -> Path:
    return Path(__file__).with_name("bin_config.json")


def _coerce_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def load_bin_config(config_path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    path = config_path or _default_config_path()
    try:
        raw_config = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        raw_config = {}

    if isinstance(raw_config, dict) and isinstance(raw_config.get("bins"), dict):
        raw_config = raw_config["bins"]

    config: Dict[str, Dict[str, float]] = {}
    for class_name in TRASH_CLASSES:
        raw_settings = raw_config.get(class_name, {}) if isinstance(raw_config, dict) else {}
        max_depth = _coerce_float(
            raw_settings.get("bin_floor_distance"),
            DEFAULT_BIN_SETTINGS["bin_floor_distance"],
        )
        threshold = _coerce_float(
            raw_settings.get("empty_threshold_percent"),
            DEFAULT_BIN_SETTINGS["empty_threshold_percent"],
        )
        config[class_name] = {
            "bin_floor_distance": max(max_depth, 1.0),
            "empty_threshold_percent": min(max(threshold, 0.0), 100.0),
        }
    return config


def normalize_expected_class(value: Optional[str]) -> str:
    if not value:
        return DEFAULT_EXPECTED_CLASS
    normalized = value.strip().lower()
    if normalized in TRASH_CLASSES:
        return normalized
    return DEFAULT_EXPECTED_CLASS


def compute_fullness_percent(
    average_depth_mm: Optional[float],
    bin_floor_distance: float,
) -> Optional[float]:
    if average_depth_mm is None:
        return None
    fullness = ((bin_floor_distance - average_depth_mm) / bin_floor_distance) * 100.0
    return round(min(max(fullness, 0.0), 100.0), 1)


def build_state_payload(
    state: DashboardState,
    bin_config: Dict[str, Dict[str, float]],
    expected_class: Optional[str],
) -> Dict[str, Any]:
    selected_class = normalize_expected_class(expected_class)
    selected_config = bin_config.get(selected_class, DEFAULT_BIN_SETTINGS)
    max_depth = selected_config["bin_floor_distance"]
    threshold = selected_config["empty_threshold_percent"]
    fullness_percent = compute_fullness_percent(state.depth_mm, max_depth)
    wrong_objects = [
      detection
      for detection in state.detections
      if detection.get("trash_label") != selected_class
    ]

    return {
        "expected_class": selected_class,
        "classes": TRASH_CLASSES,
        "bin_config": selected_config,
        "average_depth_mm": state.depth_mm,
        "depth_mm": state.depth_mm,
        "bin_floor_distance": max_depth,
        "empty_threshold_percent": threshold,
        "fullness_percent": fullness_percent,
        "is_full_enough_to_empty": (
            fullness_percent is not None and fullness_percent >= threshold
        ),
        "has_wrong_object": bool(wrong_objects),
        "wrong_objects": wrong_objects,
        "detections": state.detections,
        "updated_at": state.updated_at,
    }


_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Bin Monitor</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101417;
      --surface: #171d21;
      --surface-2: #20282d;
      --border: #344148;
      --text: #eef3f5;
      --muted: #a9b6bc;
      --ok: #5ee0a0;
      --warn: #ffc857;
      --bad: #ff6b6b;
      --info: #76c7f2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
    }
    header {
      max-width: 1360px;
      margin: 0 auto;
      padding: 24px 20px 8px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
    }
    .tab-bar {
      max-width: 1360px;
      margin: 0 auto;
      padding: 0 20px 12px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .tab-button {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 8px 18px;
      background: var(--surface-2);
      color: var(--text);
      font-weight: 700;
      cursor: pointer;
    }
    .tab-button.active {
      border-color: var(--info);
      background: #143142;
      color: #d7f2ff;
    }
    h1 {
      margin: 0;
      font-size: clamp(1.7rem, 4vw, 3rem);
      letter-spacing: 0;
    }
    .timestamp {
      color: var(--muted);
      font-size: 0.94rem;
      white-space: nowrap;
    }
    main {
      max-width: 1360px;
      margin: 0 auto;
      padding: 12px 20px 28px;
    }
    .tab-panel {
      display: none;
      grid-template-columns: minmax(0, 1.65fr) minmax(340px, 0.9fr);
      gap: 16px;
    }
    .tab-panel.active {
      display: grid;
    }
    .panel {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .panel-head {
      min-height: 58px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    h2 {
      margin: 0;
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }
    .viewer {
      padding: 12px;
    }
    .viewer img {
      width: 100%;
      display: block;
      border-radius: 8px;
      background: #050708;
      min-height: 260px;
      aspect-ratio: 16 / 9;
      object-fit: cover;
    }
    .side {
      display: grid;
      gap: 16px;
      align-content: start;
    }
    .control-grid {
      padding: 14px;
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .bin-button {
      min-height: 42px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface-2);
      color: var(--text);
      cursor: pointer;
      text-transform: capitalize;
      font-weight: 700;
    }
    .bin-button.active {
      border-color: var(--info);
      background: #143142;
      color: #d7f2ff;
    }
    .summary {
      padding: 14px;
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }
    .metric {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface-2);
      padding: 14px;
      min-height: 116px;
    }
    .metric-label {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.72rem;
      margin-bottom: 10px;
    }
    .metric-value {
      font-size: clamp(1.45rem, 4vw, 2.3rem);
      line-height: 1.05;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .muted { color: var(--muted); }
    .recap {
      padding: 0 14px 14px;
    }
    .recap-box {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #14191d;
      padding: 14px;
      line-height: 1.45;
      color: var(--text);
    }
    .detection-list {
      display: grid;
      gap: 8px;
      max-height: 360px;
      overflow: auto;
      padding: 14px;
    }
    .detection-item {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface-2);
      padding: 11px 12px;
    }
    .detection-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-weight: 700;
      text-transform: capitalize;
      margin-bottom: 6px;
    }
    .detection-meta {
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.35;
    }
    .pill {
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 5px 10px;
      color: var(--muted);
      font-size: 0.82rem;
      white-space: nowrap;
    }
    .empty {
      color: var(--muted);
      padding: 4px 0;
    }
    @media (max-width: 980px) {
      header {
        align-items: start;
        flex-direction: column;
      }
      .tab-panel {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 560px) {
      .control-grid,
      .summary {
        grid-template-columns: 1fr;
      }
      .panel-head {
        align-items: start;
        flex-direction: column;
      }
      .timestamp {
        white-space: normal;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>Bin Monitor</h1>
    <div id="lastUpdate" class="timestamp">Last update: --</div>
  </header>
  <nav class="tab-bar">
    <button id="tabIdentification" class="tab-button active" type="button">Identification</button>
    <button id="tabBinState" class="tab-button" type="button">Bin State</button>
  </nav>
  <main>
    <section id="panelIdentification" class="tab-panel active">
      <section class="panel">
        <div class="panel-head">
          <h2>Computer Vision Image</h2>
          <div id="streamStatus" class="pill">Live stream</div>
        </div>
        <div class="viewer">
          <img id="identificationStream" src="/stream.mjpg" alt="Annotated computer vision stream" />
        </div>
      </section>
      <div class="side">
        <section class="panel">
          <div class="panel-head">
            <h2>Detected Objects</h2>
            <div id="detectionCountIdentification" class="pill">0 objects</div>
          </div>
          <div id="detectionListIdentification" class="detection-list"></div>
        </section>
      </div>
    </section>

    <section id="panelBinState" class="tab-panel">
      <section class="panel">
        <div class="panel-head">
          <h2>Wrong Items Highlight</h2>
          <div id="binStreamStatus" class="pill">Bin view</div>
        </div>
        <div class="viewer">
          <img id="binStream" src="/stream-wrong.mjpg" alt="Bin view with wrong items highlighted" />
        </div>
      </section>
      <div class="side">
        <section class="panel">
          <div class="panel-head">
            <h2>Expected Bin</h2>
            <div id="selectedBin" class="pill">generic</div>
          </div>
          <div id="classControls" class="control-grid"></div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Recap</h2>
            <div id="emptyThreshold" class="pill">Threshold --</div>
          </div>
          <div class="summary">
            <div class="metric">
              <div class="metric-label">Wrong object</div>
              <div id="wrongStatus" class="metric-value muted">--</div>
            </div>
            <div class="metric">
              <div class="metric-label">Fullness</div>
              <div id="fullnessValue" class="metric-value muted">--</div>
            </div>
            <div class="metric">
              <div class="metric-label">Average depth</div>
              <div id="depthValue" class="metric-value muted">--</div>
            </div>
            <div class="metric">
              <div class="metric-label">Empty bin</div>
              <div id="emptyStatus" class="metric-value muted">--</div>
            </div>
          </div>
          <div class="recap">
            <div id="recapText" class="recap-box">Waiting for detections.</div>
          </div>
        </section>

        <section class="panel">
          <div class="panel-head">
            <h2>Bin Contents</h2>
            <div id="detectionCountBin" class="pill">0 objects</div>
          </div>
          <div id="detectionListBin" class="detection-list"></div>
        </section>
      </div>
    </section>
  </main>
  <script>
    const tabIdentification = document.getElementById('tabIdentification');
    const tabBinState = document.getElementById('tabBinState');
    const panelIdentification = document.getElementById('panelIdentification');
    const panelBinState = document.getElementById('panelBinState');
    const binStream = document.getElementById('binStream');

    const classControls = document.getElementById('classControls');
    const selectedBin = document.getElementById('selectedBin');
    const wrongStatus = document.getElementById('wrongStatus');
    const fullnessValue = document.getElementById('fullnessValue');
    const depthValue = document.getElementById('depthValue');
    const emptyStatus = document.getElementById('emptyStatus');
    const emptyThreshold = document.getElementById('emptyThreshold');
    const recapText = document.getElementById('recapText');
    const detectionCountIdentification = document.getElementById('detectionCountIdentification');
    const detectionListIdentification = document.getElementById('detectionListIdentification');
    const detectionCountBin = document.getElementById('detectionCountBin');
    const detectionListBin = document.getElementById('detectionListBin');
    const lastUpdate = document.getElementById('lastUpdate');

    const classes = ['plastic', 'metal', 'paper', 'glass', 'organic', 'generic'];
    let expectedClass = new URLSearchParams(window.location.search).get('expected_class') || 'generic';
    if (!classes.includes(expectedClass)) expectedClass = 'generic';

    function setActiveTab(tab) {
      const isIdentification = tab === 'identification';
      tabIdentification.classList.toggle('active', isIdentification);
      tabBinState.classList.toggle('active', !isIdentification);
      panelIdentification.classList.toggle('active', isIdentification);
      panelBinState.classList.toggle('active', !isIdentification);
    }

    tabIdentification.addEventListener('click', () => setActiveTab('identification'));
    tabBinState.addEventListener('click', () => setActiveTab('bin'));

    function setValue(node, value, tone) {
      node.textContent = value;
      node.className = `metric-value ${tone || 'muted'}`;
    }

    function renderControls() {
      classControls.innerHTML = '';
      classes.forEach((className) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = className === expectedClass ? 'bin-button active' : 'bin-button';
        button.textContent = className;
        button.addEventListener('click', () => {
          expectedClass = className;
          const url = new URL(window.location.href);
          url.searchParams.set('expected_class', expectedClass);
          window.history.replaceState({}, '', url);
          renderControls();
          binStream.src = `/stream-wrong.mjpg?expected_class=${encodeURIComponent(expectedClass)}&t=${Date.now()}`;
          refreshState();
        });
        classControls.appendChild(button);
      });
      selectedBin.textContent = expectedClass;
    }

    function renderDetections(detections, listNode, countNode) {
      listNode.innerHTML = '';
      countNode.textContent = `${detections.length} ${detections.length === 1 ? 'object' : 'objects'}`;
      if (!detections.length) {
        listNode.innerHTML = '<div class="empty">No detections yet.</div>';
        return;
      }

      detections.forEach((det) => {
        const hasYolo = !!det.yolo_label;
        const yoloLabel = det.yolo_label || det.label || 'unknown';
        const trashLabel = det.trash_label || 'unmapped';
        const titleLabel = hasYolo ? `${yoloLabel} - ${trashLabel}` : trashLabel;
        const item = document.createElement('div');
        item.className = 'detection-item';

        const title = document.createElement('div');
        title.className = 'detection-title';
        title.innerHTML = `<span>${titleLabel}</span><span>${Math.round(det.confidence * 100)}%</span>`;

        const meta = document.createElement('div');
        meta.className = 'detection-meta';
        meta.textContent = `bbox x=${det.xmin.toFixed(2)} y=${det.ymin.toFixed(2)} w=${det.width.toFixed(2)} h=${det.height.toFixed(2)}`;

        item.appendChild(title);
        item.appendChild(meta);
        listNode.appendChild(item);
      });
    }

    function renderRecap(state) {
      const depth = state.average_depth_mm;
      const fullness = state.fullness_percent;
      const wrongLabels = [...new Set(state.wrong_objects.map((item) => item.trash_label || 'unmapped'))];

      if (state.has_wrong_object) {
        setValue(wrongStatus, 'Yes', 'bad');
      } else if (state.detections.length) {
        setValue(wrongStatus, 'No', 'ok');
      } else {
        setValue(wrongStatus, '--', 'muted');
      }

      if (fullness === null || fullness === undefined) {
        setValue(fullnessValue, '--', 'muted');
      } else {
        setValue(fullnessValue, `${fullness.toFixed(1)}%`, fullness >= state.empty_threshold_percent ? 'warn' : 'ok');
      }

      if (depth === null || depth === undefined) {
        setValue(depthValue, '--', 'muted');
      } else {
        setValue(depthValue, `${depth.toFixed(1)} mm`, 'ok');
      }

      setValue(
        emptyStatus,
        state.is_full_enough_to_empty ? 'Yes' : 'No',
        state.is_full_enough_to_empty ? 'warn' : 'ok'
      );

      emptyThreshold.textContent = `Threshold ${state.empty_threshold_percent.toFixed(0)}%`;

      if (!state.detections.length) {
        recapText.textContent = 'No object is currently detected in this bin view.';
      } else if (state.has_wrong_object) {
        recapText.textContent = `Expected ${state.expected_class}, but detected ${wrongLabels.join(', ')}. Average depth is ${depth === null || depth === undefined ? 'not available' : `${depth.toFixed(1)} mm`}.`;
      } else {
        recapText.textContent = `All detected objects match ${state.expected_class}. Average depth is ${depth === null || depth === undefined ? 'not available' : `${depth.toFixed(1)} mm`}.`;
      }
    }

    async function refreshState() {
      try {
        const response = await fetch(`/api/state?expected_class=${encodeURIComponent(expectedClass)}`, { cache: 'no-store' });
        const state = await response.json();

        const previousClass = expectedClass;
        expectedClass = state.expected_class || expectedClass;
        if (expectedClass !== previousClass) {
          renderControls();
          binStream.src = `/stream-wrong.mjpg?expected_class=${encodeURIComponent(expectedClass)}&t=${Date.now()}`;
        }
        selectedBin.textContent = expectedClass;
        renderDetections(state.detections, detectionListIdentification, detectionCountIdentification);
        renderDetections(state.detections, detectionListBin, detectionCountBin);
        renderRecap(state);
        lastUpdate.textContent = state.updated_at ? `Last update: ${new Date(state.updated_at * 1000).toLocaleTimeString()}` : 'Last update: --';
      } catch (error) {
        console.error(error);
      }
    }

    renderControls();
    binStream.src = `/stream-wrong.mjpg?expected_class=${encodeURIComponent(expectedClass)}`;
    refreshState();
    setInterval(refreshState, 350);
  </script>
</body>
</html>
"""


def _send_json(handler: BaseHTTPRequestHandler, payload: Dict[str, Any]) -> None:
    encoded = json.dumps(payload).encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _build_handler(store: DashboardStore, bin_config: Dict[str, Dict[str, float]]):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)

            if parsed.path in {"/", "/bin-monitor"}:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(_DASHBOARD_HTML.encode("utf-8"))
                return

            if parsed.path == "/api/state":
                query = parse_qs(parsed.query)
                expected_class = query.get("expected_class", [None])[0]
                payload = build_state_payload(store.snapshot(), bin_config, expected_class)
                _send_json(self, payload)
                return

            if parsed.path == "/stream.mjpg":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()

                try:
                    while True:
                        state = store.snapshot()
                        frame = _encode_annotated_frame(
                            state, expected_class=None, highlight_wrong_only=False
                        )
                        if frame:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

            if parsed.path == "/stream-wrong.mjpg":
                query = parse_qs(parsed.query)
                expected_class = query.get("expected_class", [None])[0]

                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()

                try:
                    while True:
                        state = store.snapshot()
                        frame = _encode_annotated_frame(
                            state,
                            expected_class=expected_class,
                            highlight_wrong_only=True,
                        )
                        if frame:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("utf-8"))
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return

            self.send_error(HTTPStatus.NOT_FOUND)

        def log_message(self, format, *args):
            return

    return DashboardHandler


class DashboardServer:
    def __init__(
        self,
        host: str,
        port: int,
        store: DashboardStore,
        bin_config_path: Optional[Path] = None,
    ):
        self.host = host
        self.port = port
        self.store = store
        self.bin_config = load_bin_config(bin_config_path)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        handler = _build_handler(self.store, self.bin_config)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
