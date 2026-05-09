import json
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


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


_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>DepthAI Dashboard</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #09111d;
      --panel: rgba(16, 24, 39, 0.88);
      --panel-border: rgba(148, 163, 184, 0.18);
      --text: #e5eefb;
      --muted: #93a4bf;
      --accent: #66e3b4;
      --accent-2: #7dd3fc;
      --warn: #f59e0b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(102, 227, 180, 0.20), transparent 30%),
        radial-gradient(circle at top right, rgba(125, 211, 252, 0.18), transparent 22%),
        linear-gradient(180deg, #071019 0%, #0b1422 55%, #060b12 100%);
      color: var(--text);
      min-height: 100vh;
    }
    header {
      padding: 28px 24px 12px;
      max-width: 1400px;
      margin: 0 auto;
    }
    h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      letter-spacing: -0.04em;
    }
    .subhead {
      margin-top: 10px;
      color: var(--muted);
      max-width: 860px;
      line-height: 1.5;
    }
    main {
      max-width: 1400px;
      margin: 0 auto;
      padding: 12px 24px 28px;
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 20px;
      overflow: hidden;
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.28);
      backdrop-filter: blur(16px);
    }
    .panel h2 {
      margin: 0;
      padding: 18px 20px;
      border-bottom: 1px solid var(--panel-border);
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
    }
    .viewer {
      padding: 16px;
    }
    .viewer img {
      width: 100%;
      display: block;
      border-radius: 14px;
      background: #02050a;
      min-height: 240px;
      object-fit: cover;
    }
    .stats {
      padding: 16px 18px 20px;
      display: grid;
      gap: 14px;
    }
    .metric {
      border: 1px solid var(--panel-border);
      border-radius: 16px;
      padding: 16px;
      background: rgba(255, 255, 255, 0.03);
    }
    .metric-label {
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 0.72rem;
      margin-bottom: 10px;
    }
    .metric-value {
      font-size: 2rem;
      font-weight: 700;
      line-height: 1.1;
    }
    .metric-value.ok {
      color: var(--accent);
    }
    .metric-value.warn {
      color: var(--warn);
    }
    .detection-list {
      display: grid;
      gap: 10px;
      max-height: 520px;
      overflow: auto;
      padding-right: 2px;
    }
    .detection-item {
      border-radius: 14px;
      padding: 12px 14px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(148, 163, 184, 0.14);
    }
    .detection-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      font-weight: 600;
      margin-bottom: 6px;
    }
    .detection-meta {
      color: var(--muted);
      font-size: 0.92rem;
      line-height: 1.4;
    }
    .empty {
      color: var(--muted);
      padding: 6px 2px;
    }
    @media (max-width: 980px) {
      main {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>DepthAI Dashboard</h1>
    <div class="subhead">
      Live camera view, YOLO detections, and average depth from the active pipeline.
      The stream updates continuously while the device is running.
    </div>
  </header>
  <main>
    <section class="panel">
      <h2>Camera</h2>
      <div class="viewer">
        <img src="/stream.mjpg" alt="Live camera stream" />
      </div>
    </section>
    <section class="panel">
      <h2>Stats</h2>
      <div class="stats">
        <div class="metric">
          <div class="metric-label">Average depth</div>
          <div id="depthValue" class="metric-value warn">--</div>
        </div>
        <div class="metric">
          <div class="metric-label">Detections</div>
          <div id="detectionCount" class="metric-value ok">0</div>
          <div id="detectionList" class="detection-list"></div>
        </div>
        <div class="metric">
          <div class="metric-label">Last update</div>
          <div id="lastUpdate" class="metric-value">--</div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const depthValue = document.getElementById('depthValue');
    const detectionCount = document.getElementById('detectionCount');
    const detectionList = document.getElementById('detectionList');
    const lastUpdate = document.getElementById('lastUpdate');

    function renderDetections(detections) {
      detectionList.innerHTML = '';
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
        meta.textContent = `bbox: x=${det.xmin.toFixed(2)} y=${det.ymin.toFixed(2)} w=${det.width.toFixed(2)} h=${det.height.toFixed(2)}`;

        item.appendChild(title);
        item.appendChild(meta);
        detectionList.appendChild(item);
      });
    }

    async function refreshState() {
      try {
        const response = await fetch('/api/state', { cache: 'no-store' });
        const state = await response.json();

        if (state.depth_mm === null || state.depth_mm === undefined) {
          depthValue.textContent = '--';
          depthValue.className = 'metric-value warn';
        } else {
          depthValue.textContent = `${state.depth_mm.toFixed(1)} mm`;
          depthValue.className = 'metric-value ok';
        }

        detectionCount.textContent = String(state.detections.length);
        renderDetections(state.detections);
        lastUpdate.textContent = state.updated_at ? new Date(state.updated_at * 1000).toLocaleTimeString() : '--';
      } catch (error) {
        console.error(error);
      }
    }

    refreshState();
    setInterval(refreshState, 300);
  </script>
</body>
</html>
"""


def _build_handler(store: DashboardStore):
    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(_DASHBOARD_HTML.encode("utf-8"))
                return

            if self.path == "/api/state":
                state = store.snapshot()
                payload = json.dumps(
                    {
                        "depth_mm": state.depth_mm,
                        "detections": state.detections,
                        "updated_at": state.updated_at,
                    }
                ).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            if self.path == "/stream.mjpg":
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
    def __init__(self, host: str, port: int, store: DashboardStore):
        self.host = host
        self.port = port
        self.store = store
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return

        handler = _build_handler(self.store)
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
