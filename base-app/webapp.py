import json
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse


TRASH_CLASSES = ["plastic", "metal", "paper", "glass", "organic", "generic"]
DEFAULT_BIN_SETTINGS = {
    "max_average_depth_mm": 1000.0,
    "empty_threshold_percent": 80.0,
}
DEFAULT_EXPECTED_CLASS = "generic"


@dataclass
class DashboardState:
    frame_jpeg: Optional[bytes] = None
    detections: List[Dict[str, Any]] = field(default_factory=list)
    depth_mm: Optional[float] = None
    updated_at: float = 0.0


class DashboardStore:
    def __init__(self):
        self._state = DashboardState()
        self._lock = threading.Lock()

    def update_frame(self, frame_jpeg: bytes) -> None:
        with self._lock:
            self._state.frame_jpeg = frame_jpeg
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
                frame_jpeg=self._state.frame_jpeg,
                detections=list(self._state.detections),
                depth_mm=self._state.depth_mm,
                updated_at=self._state.updated_at,
            )


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
            raw_settings.get("max_average_depth_mm"),
            DEFAULT_BIN_SETTINGS["max_average_depth_mm"],
        )
        threshold = _coerce_float(
            raw_settings.get("empty_threshold_percent"),
            DEFAULT_BIN_SETTINGS["empty_threshold_percent"],
        )
        config[class_name] = {
            "max_average_depth_mm": max(max_depth, 1.0),
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
    max_average_depth_mm: float,
) -> Optional[float]:
    if average_depth_mm is None:
        return None
    fullness = ((max_average_depth_mm - average_depth_mm) / max_average_depth_mm) * 100.0
    return round(min(max(fullness, 0.0), 100.0), 1)


def build_state_payload(
    state: DashboardState,
    bin_config: Dict[str, Dict[str, float]],
    expected_class: Optional[str],
) -> Dict[str, Any]:
    selected_class = normalize_expected_class(expected_class)
    selected_config = bin_config.get(selected_class, DEFAULT_BIN_SETTINGS)
    max_depth = selected_config["max_average_depth_mm"]
    threshold = selected_config["empty_threshold_percent"]
    fullness_percent = compute_fullness_percent(state.depth_mm, max_depth)
    wrong_objects = [
        detection
        for detection in state.detections
        if detection.get("label") != selected_class
    ]

    return {
        "expected_class": selected_class,
        "classes": TRASH_CLASSES,
        "bin_config": selected_config,
        "average_depth_mm": state.depth_mm,
        "depth_mm": state.depth_mm,
        "max_average_depth_mm": max_depth,
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
      display: grid;
      grid-template-columns: minmax(0, 1.65fr) minmax(340px, 0.9fr);
      gap: 16px;
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
      main {
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
  <main>
    <section class="panel">
      <div class="panel-head">
        <h2>Computer Vision Image</h2>
        <div id="streamStatus" class="pill">Live stream</div>
      </div>
      <div class="viewer">
        <img src="/stream.mjpg" alt="Annotated computer vision stream" />
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
          <h2>Detected Objects</h2>
          <div id="detectionCount" class="pill">0 objects</div>
        </div>
        <div id="detectionList" class="detection-list"></div>
      </section>
    </div>
  </main>
  <script>
    const classControls = document.getElementById('classControls');
    const selectedBin = document.getElementById('selectedBin');
    const wrongStatus = document.getElementById('wrongStatus');
    const fullnessValue = document.getElementById('fullnessValue');
    const depthValue = document.getElementById('depthValue');
    const emptyStatus = document.getElementById('emptyStatus');
    const emptyThreshold = document.getElementById('emptyThreshold');
    const recapText = document.getElementById('recapText');
    const detectionCount = document.getElementById('detectionCount');
    const detectionList = document.getElementById('detectionList');
    const lastUpdate = document.getElementById('lastUpdate');

    const classes = ['plastic', 'metal', 'paper', 'glass', 'organic', 'generic'];
    let expectedClass = new URLSearchParams(window.location.search).get('expected_class') || 'generic';
    if (!classes.includes(expectedClass)) expectedClass = 'generic';

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
          refreshState();
        });
        classControls.appendChild(button);
      });
      selectedBin.textContent = expectedClass;
    }

    function renderDetections(detections) {
      detectionList.innerHTML = '';
      detectionCount.textContent = `${detections.length} ${detections.length === 1 ? 'object' : 'objects'}`;
      if (!detections.length) {
        detectionList.innerHTML = '<div class="empty">No detections yet.</div>';
        return;
      }

      detections.forEach((det) => {
        const item = document.createElement('div');
        item.className = 'detection-item';

        const title = document.createElement('div');
        title.className = 'detection-title';
        title.innerHTML = `<span>${det.label}</span><span>${Math.round(det.confidence * 100)}%</span>`;

        const meta = document.createElement('div');
        meta.className = 'detection-meta';
        meta.textContent = `bbox x=${det.xmin.toFixed(2)} y=${det.ymin.toFixed(2)} w=${det.width.toFixed(2)} h=${det.height.toFixed(2)}`;

        item.appendChild(title);
        item.appendChild(meta);
        detectionList.appendChild(item);
      });
    }

    function renderRecap(state) {
      const depth = state.average_depth_mm;
      const fullness = state.fullness_percent;
      const wrongLabels = [...new Set(state.wrong_objects.map((item) => item.label))];

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

        expectedClass = state.expected_class || expectedClass;
        selectedBin.textContent = expectedClass;
        renderDetections(state.detections);
        renderRecap(state);
        lastUpdate.textContent = state.updated_at ? `Last update: ${new Date(state.updated_at * 1000).toLocaleTimeString()}` : 'Last update: --';
      } catch (error) {
        console.error(error);
      }
    }

    renderControls();
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
                        frame = state.frame_jpeg
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
