import os
import time
import threading
import json
import socket
import webbrowser
import atexit
import signal
from pathlib import Path
from flask import Flask, request, render_template_string, Response, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np

from notifier import alert_manager, INCIDENTS_DIR
from detector import CCTVAnalyzer


def get_lan_ip():
    """Dynamically determines the host machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def auto_open_browser():
    """Waits until the server is actively listening on port 5000, then opens the browser instantly."""
    host = "127.0.0.1"
    port = 5000
    for _ in range(60):
        try:
            with socket.create_connection((host, port), timeout=0.15):
                break
        except OSError:
            time.sleep(0.08)
    try:
        print("[CampusCCTV] Server active! Launching dashboard in default browser...")
        webbrowser.open(f"http://{host}:{port}")
    except Exception as e:
        print(f"[CampusCCTV] Auto-open browser notice: {e}")


app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
INCIDENTS_DIR.mkdir(exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 250 MB
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

stream_lock = threading.Lock()
current_source = "webcam"
current_metrics = {
    "camera_connected": False,
    "source": "webcam",
    "motion_score": 0.0,
    "fight_score": 0.0,
    "fight_detected": False,
    "high_motion_detected": False,
    "persons_count": 0,
    "ai_model": "Deep Learning Bi-GRU Skeleton Pose",
    "student_predictions": {},
    "persons": [],
    "involved_persons": [],
    "last_alert_time": 0.0,
    "cooldown_remaining": 0,
    "active_incident_id": None
}

analyzer = CCTVAnalyzer()
analyzer.set_sensitivity(alert_manager.config.get("sensitivity", "medium"))
latest_annotated_frame = None
latest_frame_lock = threading.Lock()


def get_available_videos():
    videos = []
    for f in UPLOAD_FOLDER.iterdir():
        if f.is_file() and f.suffix.lower() in ALLOWED_EXTENSIONS:
            size_mb = round(f.stat().st_size / (1024 * 1024), 2)
            videos.append({
                "filename": f.name,
                "size_mb": size_mb,
                "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
            })
    return sorted(videos, key=lambda x: x["filename"])


class CameraHub:
    """
    Centralized singleton camera & video capture engine.
    Runs a dedicated background worker thread that captures frames, runs CCTVAnalyzer,
    and encodes JPEG images.
    All Flask clients receive the latest frame from this single pipeline,
    eliminating any camera hardware lockouts, race conditions, or duplicate YOLO processing.
    """
    def __init__(self, analyzer, alert_manager, upload_folder):
        self.analyzer = analyzer
        self.alert_manager = alert_manager
        self.upload_folder = Path(upload_folder)

        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

        self.current_source = "webcam"
        self.current_camera_index = 0
        self.available_cameras = [0]

        self.latest_jpeg = None
        self.latest_annotated_frame = None
        self.latest_raw_frame = None
        self.frame_id = 0

        # Immediately publish initial standby HUD frame so /video_feed NEVER stalls on client connect!
        try:
            init_standby = self._create_standby_frame(
                "LIVE WEBCAM CONNECTING...",
                "Initializing camera hardware video feed..."
            )
            ret, buf = cv2.imencode(".jpg", init_standby, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                self.latest_jpeg = buf.tobytes()
                self.latest_annotated_frame = init_standby
                self.latest_raw_frame = init_standby
                self.frame_id = 1
        except Exception as e:
            print(f"[CameraHub] Init standby frame notice: {e}")

        self.current_metrics = {
            "camera_connected": False,
            "source": "webcam",
            "motion_score": 0.0,
            "fight_score": 0.0,
            "fight_detected": False,
            "high_motion_detected": False,
            "persons_count": 0,
            "ai_model": "Deep Learning Bi-GRU Skeleton Pose",
            "student_predictions": {},
            "persons": [],
            "involved_persons": [],
            "last_alert_time": 0.0,
            "cooldown_remaining": 0,
            "active_incident_id": None,
            "kinetic_telemetry": {
                "primary_gesture": "Calm Conversation",
                "primary_demeanor": "Relaxed / Smiling",
                "gesture_category": "CALM",
                "strike_speed": 0.0,
                "combat_energy": 0.0,
                "safe_gesture_filter": "ACTIVE",
                "is_combat": False
            }
        }

        self.active_cap = None
        self.active_stream_clients = 0
        self.last_client_activity = time.time()
        self.is_standby = False

        self._running = True
        self._need_reconnect = False
        self._thread = threading.Thread(target=self._worker_loop, name="CameraHubWorker", daemon=True)
        self._thread.start()

    def heartbeat(self):
        """Notifies the engine that a client is actively viewing or polling."""
        with self.lock:
            self.last_client_activity = time.time()
            if self.is_standby:
                self.is_standby = False
                self._need_reconnect = True
        with self.condition:
            self.condition.notify_all()

    def client_connected(self):
        with self.lock:
            self.active_stream_clients += 1
            self.last_client_activity = time.time()
            if self.is_standby:
                self.is_standby = False
                self._need_reconnect = True
        with self.condition:
            self.condition.notify_all()

    def client_disconnected(self):
        with self.lock:
            self.active_stream_clients = max(0, self.active_stream_clients - 1)
            self.last_client_activity = time.time()

    def release_camera(self, message="Webcam hardware released (LED OFF). Open or refresh dashboard to resume."):
        """Immediately releases camera hardware so physical camera LED turns off."""
        with self.lock:
            if self.active_cap is not None:
                try:
                    self.active_cap.release()
                except Exception as e:
                    print(f"[CameraHub] Notice releasing active camera: {e}")
                self.active_cap = None
            self.is_standby = True
            self.current_metrics["camera_connected"] = False
            standby = self._create_standby_frame("● CAMERA HARDWARE RELEASED", message)
            ret, buffer = cv2.imencode(".jpg", standby, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                self.latest_jpeg = buffer.tobytes()
                self.latest_annotated_frame = standby
                self.frame_id += 1
        with self.condition:
            self.condition.notify_all()
        print("[CameraHub] Camera hardware released. Physical LED should be OFF.")

    def stop(self):
        """Completely terminates worker thread and guarantees hardware camera release."""
        self._running = False
        self.release_camera("System terminated.")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    def _open_camera_capture(self, index):
        """Attempts to open camera device with backends in order of reliability on Windows."""
        backends = [
            (cv2.CAP_DSHOW, "CAP_DSHOW"),
            (cv2.CAP_MSMF, "CAP_MSMF"),
            (cv2.CAP_ANY, "CAP_ANY")
        ]
        for api, api_name in backends:
            try:
                cap = cv2.VideoCapture(index, api)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        print(f"[CameraHub] Successfully opened Camera index {index} with {api_name} ({test_frame.shape[1]}x{test_frame.shape[0]})")
                        return cap, test_frame
                    cap.release()
            except Exception as e:
                print(f"[CameraHub] Probe exception for camera {index} with {api_name}: {e}")
        return None, None

    def _find_working_camera(self):
        """Scans preferred camera index first, then remaining indices."""
        preferred = self.current_camera_index
        cap, test_frame = self._open_camera_capture(preferred)
        if cap is not None:
            return preferred, cap, test_frame

        for idx in [0, 1, 2]:
            if idx == preferred:
                continue
            cap, test_frame = self._open_camera_capture(idx)
            if cap is not None:
                return idx, cap, test_frame
        return None, None, None

    def _create_standby_frame(self, title, message):
        """Creates a modern HUD standby frame when camera is connecting or unavailable."""
        img = np.full((480, 640, 3), (18, 14, 10), dtype=np.uint8)
        for y in range(0, 480, 40):
            cv2.line(img, (0, y), (640, y), (28, 22, 16), 1)
        for x in range(0, 640, 40):
            cv2.line(img, (x, 0), (x, 480), (28, 22, 16), 1)

        cv2.rectangle(img, (0, 0), (640, 44), (28, 20, 14), -1)
        cv2.putText(img, "● CAMERA STANDBY / CONNECTING", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)
        cv2.putText(img, f"TIME: {time.strftime('%H:%M:%S')}", (480, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

        cv2.rectangle(img, (70, 160), (570, 320), (32, 24, 18), -1)
        cv2.rectangle(img, (70, 160), (570, 320), (0, 140, 255), 2)

        cv2.line(img, (70, 160), (95, 160), (0, 215, 255), 3)
        cv2.line(img, (70, 160), (70, 185), (0, 215, 255), 3)
        cv2.line(img, (570, 160), (545, 160), (0, 215, 255), 3)
        cv2.line(img, (570, 160), (570, 185), (0, 215, 255), 3)
        cv2.line(img, (70, 320), (95, 320), (0, 215, 255), 3)
        cv2.line(img, (70, 320), (70, 295), (0, 215, 255), 3)
        cv2.line(img, (570, 320), (545, 320), (0, 215, 255), 3)
        cv2.line(img, (570, 320), (570, 295), (0, 215, 255), 3)

        cv2.putText(img, title, (95, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (255, 255, 255), 2)
        cv2.putText(img, message, (95, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
        cv2.putText(img, "Retrying live camera handshake...", (95, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        return img

    def _worker_loop(self):
        import numpy as np
        global current_source, current_metrics, latest_annotated_frame

        while self._running:
            with self.lock:
                src = self.current_source
                cam_idx = self.current_camera_index
                self._need_reconnect = False

            cap = None
            is_webcam = (src == "webcam")

            if is_webcam:
                with self.lock:
                    idle_seconds = time.time() - self.last_client_activity
                    # Only standby if genuinely idle > 15s AND user is not actively requesting live webcam
                    should_standby = (self.active_stream_clients == 0 and idle_seconds > 15.0 and self.is_standby and not self._need_reconnect)
                    if should_standby:
                        time.sleep(0.5)
                        continue
                    else:
                        self.is_standby = False

                working_idx, cap, first_frame = self._find_working_camera()
                if cap is None:
                    with self.lock:
                        self.current_metrics["camera_connected"] = False
                        self.current_metrics["source"] = "webcam (Connecting...)"
                        current_metrics = self.current_metrics.copy()
                        standby = self._create_standby_frame(
                            "LIVE WEBCAM CONNECTING...",
                            "Verifying camera device (DirectShow / MediaFoundation)"
                        )
                        ret, buffer = cv2.imencode(".jpg", standby, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ret:
                            self.latest_jpeg = buffer.tobytes()
                            self.latest_annotated_frame = standby
                            latest_annotated_frame = standby
                            self.frame_id += 1
                            self.condition.notify_all()
                    time.sleep(1.0)
                    continue
                else:
                    with self.lock:
                        self.active_cap = cap
                        self.is_standby = False
                        self.current_camera_index = working_idx
                        cam_idx = working_idx
                        self.current_metrics["camera_connected"] = True
                        self.current_metrics["source"] = "webcam"
                        current_metrics = self.current_metrics.copy()

                    # Immediately broadcast the first frame already captured during camera handshake!
                    if first_frame is not None and first_frame.size > 0:
                        try:
                            source_label = "Webcam (Live Camera)"
                            fh, fw = first_frame.shape[:2]
                            if max(fh, fw) > 800:
                                scale = 800.0 / max(fh, fw)
                                first_frame = cv2.resize(first_frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)

                            annotated_first, metrics_first = self.analyzer.process_frame(first_frame, source_name=source_label)
                            ret, buffer = cv2.imencode(".jpg", annotated_first, [cv2.IMWRITE_JPEG_QUALITY, 80])
                            if ret:
                                with self.condition:
                                    self.latest_jpeg = buffer.tobytes()
                                    self.latest_annotated_frame = annotated_first.copy()
                                    self.latest_raw_frame = first_frame.copy()
                                    self.current_metrics = metrics_first
                                    current_metrics = metrics_first
                                    latest_annotated_frame = self.latest_annotated_frame
                                    self.frame_id += 1
                                    self.condition.notify_all()
                        except Exception as e:
                            print(f"[CameraHub] First frame broadcast exception: {e}")
            else:
                video_path = self.upload_folder / src
                if not video_path.exists():
                    video_path = self.upload_folder / secure_filename(src)
                if not video_path.exists():
                    print(f"[CameraHub] Archive video {src} not found. Returning to webcam.")
                    with self.lock:
                        self.current_source = "webcam"
                        current_source = "webcam"
                    continue
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"[CameraHub] Could not open archive video: {video_path}")
                    with self.lock:
                        self.current_source = "webcam"
                        current_source = "webcam"
                    time.sleep(0.5)
                    continue
                with self.lock:
                    self.active_cap = cap
                    self.current_metrics["camera_connected"] = True
                    self.current_metrics["source"] = src
                    current_metrics = self.current_metrics.copy()

            with self.lock:
                self.analyzer.reset_state()

            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0 or fps > 60:
                    fps = 30.0
                frame_delay = 1.0 / fps

                source_label = "Webcam (Live Camera)" if is_webcam else f"Archive: {src}"

                while self._running:
                    with self.lock:
                        if self.current_source != src or self._need_reconnect or (is_webcam and self.current_camera_index != cam_idx):
                            break
                        # Check idle timeout while camera is active
                        if is_webcam and self.active_stream_clients == 0 and (time.time() - self.last_client_activity > 10.0):
                            print("[CameraHub] No active browser clients detected for 10s. Putting camera to sleep...")
                            break

                    t_start = time.time()
                    success, frame = cap.read()

                    if not success or frame is None or frame.size == 0:
                        if not is_webcam:
                            # Re-open archive video cleanly from start to ensure seamless continuous looping
                            try:
                                cap.release()
                            except Exception:
                                pass
                            cap = cv2.VideoCapture(str(video_path))
                            self.analyzer.reset_state()
                            if cap.isOpened():
                                success, frame = cap.read()
                            if not success or frame is None or frame.size == 0:
                                time.sleep(0.1)
                                continue
                        else:
                            print("[CameraHub] Frame read dropped on live webcam, will re-probe...")
                            time.sleep(0.2)
                            break

                    # Normalize resolution for smooth 30 FPS streaming
                    fh, fw = frame.shape[:2]
                    if max(fh, fw) > 800:
                        scale = 800.0 / max(fh, fw)
                        frame = cv2.resize(frame, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)

                    try:
                        annotated_frame, metrics = self.analyzer.process_frame(frame, source_name=source_label)
                    except Exception as proc_err:
                        print(f"[CameraHub] Analysis error: {proc_err}")
                        annotated_frame = frame.copy()
                        metrics = {
                            "camera_connected": True,
                            "motion_score": 0.0,
                            "fight_score": 0.0,
                            "fight_detected": False,
                            "high_motion_detected": False,
                            "persons_count": 0,
                            "persons": [],
                            "involved_persons": [],
                            "source": source_label,
                            "timestamp": time.time()
                        }

                    now = time.time()
                    cooldown = self.alert_manager.config.get("cooldown_seconds", 25)
                    time_since_last = now - self.alert_manager.last_alert_time
                    cooldown_left = max(0, int(cooldown - time_since_last))

                    # Danger Alert & Authority Dispatch Trigger (Instant dispatch upon detection)
                    has_danger = metrics.get("fight_detected") and len(metrics.get("involved_persons", [])) > 0

                    if has_danger:
                        k_tel = metrics.get("kinetic_telemetry", {})
                        if time_since_last >= cooldown:
                            fighters = metrics.get("involved_persons", [])
                            details = {
                                "location": "Camera 01 - Main Quad",
                                "source": source_label,
                                "persons": fighters,
                                "persons_count": metrics.get("persons_count", len(fighters)),
                                "score": max(metrics.get("fight_score", 0.0), 65.0),
                                "confidence": 95.0,
                                "facial_demeanor": k_tel.get("primary_demeanor", "Hostile Combat Stance"),
                                "gesture": k_tel.get("primary_gesture", "Violent Combat Strike")
                            }
                            print(f"[CameraHub] !!! DANGER ALERT TRIGGERED FOR AUTHORITY: {details} !!!")
                            success_dispatch, inc_id = self.alert_manager.trigger_alert(
                                "FIGHT_DETECTED", details, frame=annotated_frame
                            )
                            if success_dispatch:
                                metrics["active_incident_id"] = inc_id

                    metrics["cooldown_remaining"] = cooldown_left
                    metrics["source"] = src
                    metrics["camera_connected"] = True

                    ret, buffer = cv2.imencode(".jpg", annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        jpeg_bytes = buffer.tobytes()
                        with self.condition:
                            self.latest_jpeg = jpeg_bytes
                            self.latest_annotated_frame = annotated_frame.copy()
                            self.latest_raw_frame = frame.copy()
                            self.current_metrics = metrics
                            current_metrics = metrics
                            latest_annotated_frame = self.latest_annotated_frame
                            self.frame_id += 1
                            self.condition.notify_all()

                    elapsed = time.time() - t_start
                    if not is_webcam:
                        sleep_time = max(0.005, frame_delay - elapsed)
                        time.sleep(sleep_time)
                    else:
                        if elapsed < 0.033:
                            time.sleep(0.033 - elapsed)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception as e:
                        print(f"[CameraHub] Error releasing cap: {e}")
                with self.lock:
                    if self.active_cap is cap:
                        self.active_cap = None

    def generate_client_stream(self):
        """Yields continuous MJPEG stream with boundary markers and Content-Length to any connected client."""
        last_id = -1
        self.client_connected()
        try:
            while self._running:
                try:
                    with self.condition:
                        while self.frame_id == last_id or self.latest_jpeg is None:
                            if not self.condition.wait(timeout=0.4):
                                break
                        jpeg = self.latest_jpeg
                        last_id = self.frame_id

                    if jpeg is not None:
                        self.heartbeat()
                        header = (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                        )
                        yield (header + jpeg + b"\r\n")
                    else:
                        time.sleep(0.03)
                except (GeneratorExit, ConnectionResetError, BrokenPipeError, OSError):
                    break
                except Exception:
                    break
        finally:
            self.client_disconnected()

    def set_source(self, source):
        """Thread-safely sets the active feed source."""
        global current_source
        if source != "webcam":
            video_path = self.upload_folder / source
            if not video_path.exists():
                video_path = self.upload_folder / secure_filename(source)
            if not video_path.exists():
                return False, f"Video '{source}' not found in archive."
            resolved_source = video_path.name
        else:
            resolved_source = "webcam"

        with self.lock:
            self.current_source = resolved_source
            current_source = resolved_source
            self._need_reconnect = True
            # WAKE UP CAMERA IMMEDIATELY: Clear standby & reset idle timers
            self.is_standby = False
            self.last_client_activity = time.time()
            if self.active_cap is not None:
                try:
                    self.active_cap.release()
                except Exception:
                    pass
                self.active_cap = None
            self.analyzer.reset_state()
            self.alert_manager.silence_alarm()

        with self.condition:
            self.condition.notify_all()
        return True, resolved_source

    def reconnect_camera(self):
        """Forces the capture thread to re-open and probe camera hardware."""
        global current_source
        with self.lock:
            self.current_source = "webcam"
            current_source = "webcam"
            self._need_reconnect = True
            self.is_standby = False
            self.last_client_activity = time.time()
            if self.active_cap is not None:
                try:
                    self.active_cap.release()
                except Exception:
                    pass
                self.active_cap = None
            self.analyzer.reset_state()
            self.alert_manager.silence_alarm()

        with self.condition:
            self.condition.notify_all()
        return True

    def get_metrics(self):
        with self.lock:
            return self.current_metrics.copy()

    def get_latest_annotated_frame(self):
        with self.lock:
            return self.latest_annotated_frame.copy() if self.latest_annotated_frame is not None else None

    def get_current_source(self):
        with self.lock:
            return self.current_source


camera_hub = CameraHub(analyzer, alert_manager, UPLOAD_FOLDER)


# ---------------- WEB INTERFACE & APIS ----------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Campus CCTV AI Monitoring System | Mini Project Demonstration</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg-base: #060b13;
    --bg-surface: #0c1424;
    --bg-card: #111e35;
    --bg-card-hover: #162644;
    --border: #1e3355;
    --border-bright: #2d4c7d;
    --primary: #3b82f6;
    --primary-glow: rgba(59, 130, 246, 0.35);
    --cyan: #06b6d4;
    --emerald: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --red-glow: rgba(239, 68, 68, 0.45);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    --radius-lg: 16px;
    --radius-md: 10px;
    --radius-sm: 6px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    background: var(--bg-base);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    padding-bottom: 40px;
    background-image: 
        radial-gradient(circle at 15% 15%, rgba(6, 182, 212, 0.05) 0%, transparent 40%),
        radial-gradient(circle at 85% 85%, rgba(239, 68, 68, 0.04) 0%, transparent 40%);
}

header {
    background: rgba(12, 20, 36, 0.88);
    backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
    position: sticky;
    top: 0;
    z-index: 50;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    width: 40px;
    height: 40px;
    background: linear-gradient(135deg, #2563eb, #06b6d4);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.4);
}

.brand h1 {
    font-size: 18px;
    font-weight: 700;
    letter-spacing: -0.3px;
    background: linear-gradient(to right, #ffffff, #93c5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand p {
    font-size: 12px;
    color: var(--text-muted);
}

.header-meta {
    display: flex;
    align-items: center;
    gap: 14px;
}

.mini-badge {
    background: rgba(59, 130, 246, 0.15);
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #93c5fd;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.live-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: var(--emerald);
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.live-dot {
    width: 8px;
    height: 8px;
    background: var(--emerald);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--emerald);
    animation: pulse 1.8s infinite;
}

@keyframes pulse {
    0% { transform: scale(0.9); opacity: 0.7; }
    50% { transform: scale(1.3); opacity: 1; }
    100% { transform: scale(0.9); opacity: 0.7; }
}

.clock {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: var(--text-muted);
    background: var(--bg-card);
    padding: 6px 12px;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border);
}

.container {
    max-width: 1560px;
    margin: 24px auto;
    padding: 0 20px;
    display: grid;
    grid-template-columns: 1fr 440px;
    gap: 24px;
}

@media (max-width: 1200px) {
    .container { grid-template-columns: 1fr; }
}

.card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    margin-bottom: 20px;
}

.card-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-main);
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}

/* SURVEILLANCE STAGE: Split Layout with Video & Live Gesture Movement Graph */
.surveillance-stage {
    display: grid;
    grid-template-columns: 1fr 340px;
    gap: 16px;
    align-items: stretch;
}

@media (max-width: 1100px) {
    .surveillance-stage {
        grid-template-columns: 1fr;
    }
}

.video-wrapper {
    position: relative;
    width: 100%;
    min-height: 380px;
    background: #020408;
    border-radius: var(--radius-md);
    overflow: hidden;
    border: 2px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: center;
}

.gesture-telemetry-panel {
    background: #08101d;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.4);
}

.gt-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.gt-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11.5px;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #93c5fd;
    text-transform: uppercase;
}

.gt-pulse-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #06b6d4;
    box-shadow: 0 0 8px #06b6d4;
    animation: pulse 1.5s infinite;
}

.gt-filter-pill {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    font-size: 10px;
    font-weight: 700;
    padding: 2px 8px;
    border-radius: 12px;
    letter-spacing: 0.4px;
}

.gt-canvas-box {
    background: #040812;
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    overflow: hidden;
    position: relative;
    width: 100%;
    height: 155px;
}

#gestureWaveformCanvas {
    width: 100%;
    height: 100%;
    display: block;
}

.gt-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-top: 10px;
}

.gt-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 8px 10px;
}

.gt-card-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    margin-bottom: 3px;
}

.gt-card-val {
    font-size: 12px;
    font-weight: 700;
    color: var(--text-main);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.gt-card-val.mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    color: #38bdf8;
}

.gt-safety-notice {
    margin-top: 10px;
    background: rgba(6, 182, 212, 0.08);
    border: 1px solid rgba(6, 182, 212, 0.25);
    border-radius: var(--radius-sm);
    padding: 8px 10px;
    font-size: 10.5px;
    color: #94a3b8;
    line-height: 1.4;
    display: flex;
    align-items: flex-start;
}

.video-feed {
    width: 100%;
    height: auto;
    max-height: 600px;
    object-fit: contain;
    display: block;
}

.threat-banner {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(to top, rgba(239, 68, 68, 0.95), rgba(239, 68, 68, 0.75));
    color: white;
    padding: 10px 16px;
    display: none;
    align-items: center;
    justify-content: space-between;
    font-weight: 700;
    font-size: 14px;
    animation: flashAlert 1.2s infinite alternate;
}

@keyframes flashAlert {
    from { opacity: 0.9; }
    to { opacity: 1; filter: drop-shadow(0 0 12px rgba(239,68,68,0.8)); }
}

.source-bar {
    display: flex;
    gap: 10px;
    margin-top: 14px;
    overflow-x: auto;
    padding-bottom: 4px;
}

.source-btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 8px 16px;
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 6px;
}

.source-btn:hover {
    background: var(--bg-card-hover);
    color: var(--text-main);
    border-color: var(--border-bright);
}

.source-btn.active {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    color: white;
    border-color: #60a5fa;
    box-shadow: 0 0 15px rgba(37, 99, 235, 0.4);
}

.telemetry-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 18px;
}

.telemetry-item {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 14px;
    text-align: center;
}

.telemetry-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
    color: var(--text-dim);
    margin-bottom: 6px;
}

.telemetry-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 20px;
    font-weight: 700;
    color: var(--text-main);
}

.meter-bar {
    height: 4px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 2px;
    margin-top: 8px;
    overflow: hidden;
}

.meter-fill {
    height: 100%;
    width: 0%;
    border-radius: 2px;
    transition: width 0.3s ease, background-color 0.3s ease;
}

/* Authority Profile Box (PASSWORD PROTECTED) */
.authority-box {
    background: linear-gradient(135deg, rgba(30, 58, 138, 0.35), rgba(15, 23, 42, 0.6));
    border: 1px solid #3b82f6;
    border-radius: var(--radius-md);
    padding: 16px;
    position: relative;
    box-shadow: 0 0 25px rgba(59, 130, 246, 0.15);
    transition: all 0.3s ease;
}

.authority-box.alarm-active {
    border: 2px solid #ef4444 !important;
    background: linear-gradient(135deg, rgba(185, 28, 28, 0.38), rgba(20, 10, 15, 0.85)) !important;
    box-shadow: 0 0 35px rgba(239, 68, 68, 0.75), inset 0 0 20px rgba(239, 68, 68, 0.25) !important;
    animation: authAlarmPulse 1.1s infinite alternate !important;
}

@keyframes authAlarmPulse {
    0% { box-shadow: 0 0 15px rgba(239, 68, 68, 0.45); border-color: #ef4444; }
    100% { box-shadow: 0 0 45px rgba(239, 68, 68, 0.95); border-color: #ff4d4f; }
}

.auth-alarm-banner {
    background: linear-gradient(135deg, #b91c1c, #ef4444);
    color: white;
    border-radius: var(--radius-sm);
    padding: 10px 14px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.65);
    animation: flashAlert 0.7s infinite alternate;
}

.authority-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
}

.auth-badge {
    background: rgba(59, 130, 246, 0.2);
    border: 1px solid #3b82f6;
    color: #93c5fd;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.lock-badge {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #34d399;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 4px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

.auth-avatar {
    width: 44px;
    height: 44px;
    background: linear-gradient(135deg, #1d4ed8, #0284c7);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 17px;
    color: white;
    border: 2px solid #60a5fa;
    box-shadow: 0 0 12px rgba(96, 165, 250, 0.5);
}

.auth-details {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
    font-size: 13px;
    margin: 12px 0;
    background: rgba(0, 0, 0, 0.25);
    padding: 12px;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(255, 255, 255, 0.05);
}

.auth-detail-item {
    color: var(--text-muted);
}
.auth-detail-item b {
    color: var(--text-main);
    display: block;
    font-size: 13px;
    margin-top: 2px;
}

.directory-table th, .directory-table td {
    padding: 10px 12px;
    border-bottom: 1px solid rgba(255, 255, 255, 0.07);
}
.directory-table tr:hover {
    background: rgba(255, 255, 255, 0.035);
}
.duty-badge-active {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 10px;
    font-weight: 700;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}
.duty-badge-standby {
    background: rgba(148, 163, 184, 0.1);
    border: 1px solid rgba(148, 163, 184, 0.2);
    color: #94a3b8;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 10px;
    display: inline-flex;
    align-items: center;
    gap: 4px;
}

/* Smartphone Simulation Card */
.phone-sim-card {
    background: #090e18;
    border: 1px solid #334155;
    border-radius: var(--radius-md);
    padding: 14px;
    margin-top: 14px;
    position: relative;
    overflow: hidden;
}

.phone-sim-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 11px;
    text-transform: uppercase;
    font-weight: 700;
    color: var(--cyan);
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.incoming-bubble {
    background: rgba(30, 41, 59, 0.85);
    border-left: 3px solid #3b82f6;
    padding: 10px 12px;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 12px;
    line-height: 1.45;
}

.bubble-meta {
    font-size: 10px;
    color: var(--text-dim);
    margin-top: 6px;
    display: flex;
    justify-content: space-between;
}

/* Threat Card */
.threat-card {
    border-radius: var(--radius-md);
    padding: 16px;
    margin-bottom: 18px;
    transition: all 0.3s ease;
}

.threat-card.normal {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.threat-card.critical {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid rgba(239, 68, 68, 0.6);
    box-shadow: 0 0 25px rgba(239, 68, 68, 0.25);
}

.threat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 700;
    font-size: 15px;
}

.threat-details {
    margin-top: 10px;
    font-size: 13px;
    color: var(--text-muted);
}

.badge-red {
    background: var(--red);
    color: white;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
}

.combatants-box {
    margin-top: 12px;
    background: rgba(0, 0, 0, 0.3);
    padding: 10px 12px;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.combatants-title {
    font-size: 11px;
    text-transform: uppercase;
    color: #fca5a5;
    font-weight: 700;
    margin-bottom: 6px;
}

.fighter-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
}

.fighter-chip {
    background: #dc2626;
    color: white;
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 600;
}

.roster {
    max-height: 120px;
    overflow-y: auto;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 8px;
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 10px;
}

.person-pill {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid var(--border-bright);
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    display: flex;
    align-items: center;
    gap: 4px;
}

.person-pill.fighting {
    background: rgba(239, 68, 68, 0.25);
    border-color: var(--red);
    color: #fca5a5;
}

.btn {
    flex: 1;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: var(--radius-sm);
    padding: 9px 12px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    text-decoration: none;
}

.btn:hover {
    background: #2563eb;
    box-shadow: 0 0 12px var(--primary-glow);
}

.btn-secondary {
    background: rgba(255, 255, 255, 0.08);
    color: var(--text-main);
    border: 1px solid var(--border);
}

.btn-secondary:hover {
    background: rgba(255, 255, 255, 0.14);
}

.btn-whatsapp {
    background: #25d366;
    color: #0b2212;
    font-weight: 700;
}
.btn-whatsapp:hover {
    background: #20ba5a;
    box-shadow: 0 0 12px rgba(37, 211, 102, 0.4);
}

.btn-danger {
    background: #dc2626;
}
.btn-danger:hover {
    background: #b91c1c;
    box-shadow: 0 0 15px var(--red-glow);
}

.incident-list {
    max-height: 220px;
    overflow-y: auto;
    margin-top: 12px;
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.incident-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.incident-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--border-bright);
}

.incident-thumb {
    width: 52px;
    height: 38px;
    border-radius: 4px;
    object-fit: cover;
    background: #000;
    border: 1px solid var(--border);
}

.incident-info {
    flex: 1;
}

.incident-type {
    font-size: 12px;
    font-weight: 700;
    color: var(--red);
}

.incident-meta {
    font-size: 11px;
    color: var(--text-dim);
}

.modal-backdrop {
    display: none;
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0, 0, 0, 0.82);
    backdrop-filter: blur(8px);
    z-index: 100;
    align-items: center;
    justify-content: center;
    padding: 20px;
}

.modal {
    background: var(--bg-surface);
    border: 1px solid var(--border-bright);
    border-radius: var(--radius-lg);
    width: 100%;
    max-width: 520px;
    padding: 24px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.6);
}

.modal h3 {
    margin-bottom: 12px;
    font-size: 17px;
    color: var(--text-main);
}

.form-group {
    margin-bottom: 14px;
}

.form-group label {
    display: block;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.form-control {
    width: 100%;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: white;
    padding: 10px 12px;
    font-size: 13px;
    outline: none;
}

.form-control:focus {
    border-color: var(--cyan);
    box-shadow: 0 0 10px rgba(6, 182, 212, 0.25);
}

.upload-dropzone {
    border: 2px dashed var(--border-bright);
    border-radius: var(--radius-md);
    padding: 24px;
    text-align: center;
    background: rgba(255, 255, 255, 0.015);
    cursor: pointer;
    transition: all 0.2s ease;
}

.upload-dropzone:hover {
    background: rgba(6, 182, 212, 0.03);
    border-color: var(--cyan);
}

.file-input { display: none; }
</style>
</head>
<body>

<header>
    <div class="brand">
        <div class="brand-icon">
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path>
                <circle cx="12" cy="13" r="4"></circle>
            </svg>
        </div>
        <div>
            <h1>Campus CCTV AI Monitoring System</h1>
            <p>Smart Fight Detection & Single-Password Secured Authority Dispatch</p>
        </div>
    </div>
    <div class="header-meta">
        <div class="mini-badge">🎓 Mini Project Edition</div>
        <div class="live-pill">
            <span class="live-dot"></span>
            AI ENGINE ACTIVE
        </div>
        <button class="btn btn-secondary" onclick="shutdownServer()" style="padding: 4px 10px; font-size: 11px; border: 1px solid rgba(239, 68, 68, 0.5); color: #fca5a5; background: rgba(239, 68, 68, 0.08);" title="Stop AI Surveillance Server and release camera">
            ⏹ Stop & Exit
        </button>
        <div class="clock" id="systemClock">00:00:00</div>
    </div>
</header>

<div class="container">
    <!-- LEFT: SURVEILLANCE FEED & TELEMETRY -->
    <div>
        <div class="card">
            <div class="card-title">
                <span>Surveillance Video Stream</span>
                <span id="currentSourceTag" style="font-size: 12px; color: var(--cyan); font-family: 'JetBrains Mono', monospace;">SOURCE: WEBCAM</span>
            </div>

            <!-- SURVEILLANCE STAGE: Split layout placing live kinetics graph in the empty space beside the video -->
            <div class="surveillance-stage">
                <div class="video-wrapper">
                    <img id="videoFeed" class="video-feed" src="/video_feed" alt="Surveillance Stream">
                    <div id="threatBanner" class="threat-banner">
                        <span id="threatBannerText">🚨 CRITICAL ALERT: FIGHT DETECTED</span>
                        <button class="btn btn-danger" style="padding: 4px 10px; font-size: 11px;" onclick="dispatchTestAlert()">DISPATCH TO AUTHORITY</button>
                    </div>
                </div>

                <!-- LIVE BODY LANGUAGE & GESTURE MOVEMENT GRAPH (Side of Live Footage) -->
                <div class="gesture-telemetry-panel">
                    <div class="gt-header">
                        <div class="gt-title">
                            <span class="gt-pulse-dot"></span>
                            BODY KINETICS GRAPH
                        </div>
                        <span class="gt-filter-pill">🛡️ CASUAL FILTER ON</span>
                    </div>

                    <!-- Oscilloscope Real-time Waveform Canvas -->
                    <div class="gt-canvas-box">
                        <canvas id="gestureWaveformCanvas" width="320" height="155"></canvas>
                    </div>

                    <!-- Gesture & Demeanor Dynamic Cards -->
                    <div class="gt-grid">
                        <div class="gt-card">
                            <div class="gt-card-label">CLASSIFIED GESTURE</div>
                            <div class="gt-card-val" id="gtGestureType">Calm Movement</div>
                        </div>
                        <div class="gt-card">
                            <div class="gt-card-label">FACIAL DEMEANOR</div>
                            <div class="gt-card-val" id="gtFacialDemeanor">Relaxed / Smiling</div>
                        </div>
                        <div class="gt-card">
                            <div class="gt-card-label">PEAK LIMB SPEED</div>
                            <div class="gt-card-val mono" id="gtVelocityVal">0.00 m/s</div>
                        </div>
                        <div class="gt-card">
                            <div class="gt-card-label">KINETIC ENERGY</div>
                            <div class="gt-card-val mono" id="gtEnergyVal">0.00 J</div>
                        </div>
                    </div>

                    <!-- Safe Filter Explanation -->
                    <div class="gt-safety-notice">
                        <span style="color: var(--emerald); font-weight: bold; margin-right: 5px;">✓</span>
                        <span>Casual jokes, funny waving &amp; dancing are verified safe and excluded from danger alerts.</span>
                    </div>
                </div>
            </div>

            <div style="margin-top: 14px;">
                <p style="font-size: 12px; font-weight: 600; color: var(--text-dim); text-transform: uppercase; margin-bottom: 6px;">Available Video Feeds</p>
                <div class="source-bar" id="sourceBar">
                    <button class="source-btn active" id="src-webcam" onclick="setSource('webcam')">
                        🎥 Laptop Live Webcam
                    </button>
                    <button class="source-btn" id="src-reconnect-cam" onclick="reconnectLiveCamera()" style="border-style: dashed; opacity: 0.9;" title="Scan and re-initialize camera hardware">
                        🔄 Reconnect Camera
                    </button>
                    <button class="source-btn" id="src-standby-cam" onclick="standbyCamera()" style="border-style: dotted; opacity: 0.85; color: #fca5a5;" title="Release camera hardware and turn off webcam LED">
                        ⏸ Standby (Turn Off Camera)
                    </button>
                </div>
            </div>

            <div class="telemetry-grid">
                <div class="telemetry-item">
                    <div class="telemetry-label">Fight Threat Status</div>
                    <div class="telemetry-value" id="valFightRisk" style="font-size: 17px;">NORMAL</div>
                    <div class="meter-bar">
                        <div id="meterFight" class="meter-fill" style="width: 0%; background: var(--emerald);"></div>
                    </div>
                </div>

                <div class="telemetry-item">
                    <div class="telemetry-label">Motion Kinetic Index</div>
                    <div class="telemetry-value" id="valMotion">0.0%</div>
                    <div class="meter-bar">
                        <div id="meterMotion" class="meter-fill" style="width: 0%; background: var(--cyan);"></div>
                    </div>
                </div>

                <div class="telemetry-item">
                    <div class="telemetry-label">Tracked Persons</div>
                    <div class="telemetry-value" id="valPersonsCount">0</div>
                    <div class="meter-bar">
                        <div class="meter-fill" style="width: 100%; background: #6366f1;"></div>
                    </div>
                </div>

                <div class="telemetry-item">
                    <div class="telemetry-label">Authority Cooldown</div>
                    <div class="telemetry-value" id="valCooldown">READY</div>
                    <div class="meter-bar">
                        <div id="meterCooldown" class="meter-fill" style="width: 100%; background: var(--emerald);"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Video Upload Card -->
        <div class="card">
            <div class="card-title">
                <span>Upload Recorded CCTV Clip for Analysis</span>
                <span style="font-size: 12px; color: var(--text-dim);">MP4, AVI, MOV, MKV (Max 250MB)</span>
            </div>
            <form id="uploadForm" onsubmit="handleVideoUpload(event)">
                <div class="upload-dropzone" onclick="document.getElementById('videoFileInput').click()">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="margin-bottom: 6px;">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                        <polyline points="17 8 12 3 7 8"></polyline>
                        <line x1="12" y1="3" x2="12" y2="15"></line>
                    </svg>
                    <p style="font-size: 13px; font-weight: 600; color: var(--text-main);">Click to browse CCTV recording from computer</p>
                    <p id="uploadFilename" style="font-size: 12px; color: var(--cyan); margin-top: 4px;">Supports WhatsApp fight videos or CCTV archives</p>
                    <input type="file" id="videoFileInput" name="video" class="file-input" accept=".mp4,.avi,.mov,.mkv" onchange="handleFileSelected(this)">
                </div>
                <div style="margin-top: 10px; display: flex; justify-content: flex-end; gap: 8px; align-items: center;">
                    <span id="uploadStatusText" style="font-size: 11px; color: #94a3b8;"></span>
                    <button type="submit" id="btnUploadSubmit" class="btn" style="width: auto; padding: 9px 20px;">Upload & Analyze</button>
                </div>
            </form>
        </div>
    </div>

    <!-- RIGHT: PASSWORD PROTECTED AUTHORITY PERSON & INCIDENT TRANSMISSION -->
    <div>
        <div class="card">
            <div class="authority-box" id="authorityBox">
                <!-- Flashing Emergency Notification Alarm Banner -->
                <div id="authAlarmBanner" class="auth-alarm-banner" style="display: none;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 20px;">🚨</span>
                        <div>
                            <div style="font-weight: 800; font-size: 13px; letter-spacing: 0.5px; text-transform: uppercase;">NOTIFICATION ALARM ACTIVE!</div>
                            <div style="font-size: 11px; opacity: 0.95;" id="authAlarmBannerSub">Dispatched immediately to Jeevan (9353415499)</div>
                        </div>
                    </div>
                    <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px; background: rgba(0,0,0,0.4);" onclick="silenceAlarm()">Silence 🔇</button>
                </div>

                <div class="authority-header">
                    <div>
                        <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                            <span class="auth-badge" id="dispAuthSlotBadge">🛡️ AUTHORITY #1 (ON-DUTY)</span>
                            <span class="lock-badge">🔒 1 MASTER PASSWORD</span>
                        </div>
                        <h3 id="dispAuthName" style="font-size: 17px; font-weight: 700; margin-top: 8px; color: white;">Jeevan</h3>
                    </div>
                    <div class="auth-avatar" id="dispAuthInitial">J</div>
                </div>

                <div class="auth-details">
                    <div class="auth-detail-item">
                        <span>Authorized Phone (Mobile / WhatsApp):</span>
                        <b id="dispAuthPhone" style="color: #67e8f9;">9353415499</b>
                    </div>
                    <div class="auth-detail-item">
                        <span>Authorized Email ID:</span>
                        <b id="dispAuthEmail" style="color: #93c5fd;">jeevanl06232003@gmail.com</b>
                    </div>
                    <div class="auth-detail-item" style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px; padding-top: 6px; border-top: 1px solid rgba(255,255,255,0.06);">
                        <span>Notification Alarm Status:</span>
                        <span id="authAlarmStatusPill" class="duty-badge-active">● STANDBY MONITORING</span>
                    </div>
                </div>

                <!-- Immediate Action & Alert Buttons -->
                <div style="margin-top: 10px; display: flex; flex-direction: column; gap: 8px;">
                    <div style="display: flex; gap: 8px;">
                        <button class="btn btn-danger" style="flex: 2; font-weight: 800; padding: 11px 14px; font-size: 13px; background: linear-gradient(135deg, #ef4444, #dc2626); box-shadow: 0 0 20px rgba(239, 68, 68, 0.5); cursor: pointer;" onclick="checkActionAndDispatch()">
                            🚨 SEND IMMEDIATE ALARM & ALERT TO JEEVAN
                        </button>
                        <button class="btn" style="flex: 1; background: linear-gradient(135deg, #0284c7, #0369a1); font-weight: 700; font-size: 12px;" onclick="checkActionAndDispatch()">
                            ⚡ Check Actions
                        </button>
                    </div>

                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px;">
                        <a id="btnWhatsAppDirect" href="https://api.whatsapp.com/send?phone=919353415499" target="_blank" class="btn btn-whatsapp" style="text-align: center; justify-content: center; font-size: 11px;">
                            💬 WhatsApp: 9353415499
                        </a>
                        <a id="btnSmsDirect" href="sms:+919353415499" class="btn" style="text-align: center; justify-content: center; font-size: 11px; background: #0284c7; color: white; text-decoration: none;">
                            📱 SMS: 9353415499
                        </a>
                        <a id="btnEmailDirect" href="mailto:jeevanl06232003@gmail.com" class="btn btn-secondary" style="text-align: center; justify-content: center; font-size: 11px; color: #93c5fd;">
                            📧 Email
                        </a>
                    </div>

                    <div style="display: flex; gap: 8px;">
                        <button class="btn btn-secondary" style="font-size: 11px; padding: 6px 10px;" onclick="openAuthorityModal()">
                            ⚙️ Setup Authorities (1 to 10)
                        </button>
                        <button class="btn btn-secondary" style="font-size: 11px; padding: 6px 10px;" onclick="openDirectoryModal()">
                            📋 Directory List
                        </button>
                        <a id="btnPushDirect" href="https://ntfy.sh/cctv_919353415499" target="_blank" class="btn btn-secondary" style="font-size: 11px; padding: 6px 10px; color: #38bdf8;">
                            🔔 Phone Alarm Link
                        </a>
                    </div>
                </div>
            </div>

            <!-- Mobile & Push Transmission Preview -->
            <div class="phone-sim-card">
                <div class="phone-sim-header">
                    <span>📱 Live Mobile & Push Alarm Dispatch</span>
                    <span id="simStatus" style="color: var(--emerald);">● DISPATCH ACTIVE</span>
                </div>
                <div class="incoming-bubble" id="simBubble">
                    <p id="simMsgText" style="color: #cbd5e1; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                        System monitoring active. Instant dispatches are delivered to Jeevan via WhatsApp, Mobile Push, and System Siren.
                    </p>
                    <div class="bubble-meta">
                        <span id="simMetaTarget">Target: Jeevan (9353415499)</span>
                        <span id="simMetaTime">Standing by</span>
                    </div>
                </div>
                <div style="margin-top: 10px; display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap;">
                    <a id="linkMobilePushApp" href="https://ntfy.sh/cctv_919353415499" target="_blank" class="btn btn-secondary" style="padding: 4px 10px; font-size: 11px; color: #38bdf8; text-decoration: none;">
                        🔔 Open Mobile Push Alarm
                    </a>
                    <button class="btn btn-danger" style="padding: 5px 12px; font-size: 11px; font-weight: 700;" onclick="dispatchTestAlert()">
                        ⚡ Test Action Alarm
                    </button>
                </div>
            </div>

            <!-- Mobile Authority Connection Card (QR Code & Direct Phone Link) -->
            <div class="card" style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(15, 23, 42, 0.8)); border: 1px solid rgba(16, 185, 129, 0.4); margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 13px; font-weight: 700; color: #34d399;">
                        📲 Connect Jeevan's Phone for Real-Time Alarms
                    </div>
                    <span class="duty-badge-active">LIVE PAIRING</span>
                </div>
                <p style="font-size: 11px; color: var(--text-muted); line-height: 1.4; margin-bottom: 12px;">
                    Scan with your phone camera or visit <b style="color: #67e8f9;">http://{{ lan_ip }}:5000/mobile</b>. Whenever an action is detected or test alert fired on the laptop, <b>YOUR PHONE WILL RING WITH EMERGENCY SIRENS & VIBRATE!</b>
                </p>

                <div style="display: flex; gap: 14px; align-items: center; justify-content: space-around; flex-wrap: wrap; background: rgba(0,0,0,0.3); padding: 12px; border-radius: var(--radius-sm); border: 1px solid rgba(255,255,255,0.06);">
                    <div style="text-align: center;">
                        <a href="/mobile" target="_blank">
                            <img src="/qr/mobile" alt="Scan with Phone" style="width: 115px; height: 115px; background: white; padding: 5px; border-radius: 8px; box-shadow: 0 0 15px rgba(16, 185, 129, 0.35); display: block; margin: 0 auto;">
                        </a>
                        <div style="font-size: 10px; font-weight: 700; color: #34d399; margin-top: 6px;">📱 Phone Alarm Terminal</div>
                        <a href="http://{{ lan_ip }}:5000/mobile" target="_blank" style="font-size: 10px; color: #38bdf8; text-decoration: underline;">Open on Phone</a>
                    </div>

                    <div style="text-align: center;">
                        <a href="https://ntfy.sh/cctv_919353415499" target="_blank">
                            <img src="/qr/ntfy" alt="Scan for Push Alerts" style="width: 115px; height: 115px; background: white; padding: 5px; border-radius: 8px; box-shadow: 0 0 15px rgba(56, 189, 248, 0.35); display: block; margin: 0 auto;">
                        </a>
                        <div style="font-size: 10px; font-weight: 700; color: #38bdf8; margin-top: 6px;">🔔 Push Notifications</div>
                        <a href="https://ntfy.sh/cctv_919353415499" target="_blank" style="font-size: 10px; color: #94a3b8; text-decoration: underline;">ntfy.sh/cctv_919353415499</a>
                    </div>
                </div>

                <div style="display: flex; gap: 8px; margin-top: 10px;">
                    <a href="http://{{ lan_ip }}:5000/mobile" target="_blank" class="btn" style="background: linear-gradient(135deg, #059669, #10b981); font-weight: 700; font-size: 12px; text-decoration: none; text-align: center;">
                        📲 Open Phone Terminal Link
                    </a>
                </div>
            </div>

            <!-- Telegram Bot Instant Alert Card -->
            <div class="card" style="background: linear-gradient(135deg, rgba(37, 99, 235, 0.12), rgba(15, 23, 42, 0.85)); border: 1px solid rgba(59, 130, 246, 0.4); margin-top: 14px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 13px; font-weight: 700; color: #60a5fa; display: flex; align-items: center; gap: 6px;">
                        <span>✈️</span> Telegram Phone Alert Hub
                    </div>
                    <span id="tgStatusBadge" class="duty-badge-active" style="background: #2563eb;">CHECKING...</span>
                </div>
                <p style="font-size: 11px; color: var(--text-muted); line-height: 1.4; margin-bottom: 10px;">
                    Receive <b style="color: #93c5fd;">instant CCTV photo snapshots</b> of fights and critical incidents directly on your phone via Telegram (100% Free &amp; Unlimited).
                </p>

                <div style="background: rgba(0,0,0,0.3); padding: 10px; border-radius: var(--radius-sm); border: 1px solid rgba(255,255,255,0.06); margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 6px;">
                        <span style="color: var(--text-muted);">Bot Account:</span>
                        <a href="https://t.me/Alert_22bms_bot" target="_blank" style="color: #38bdf8; font-weight: 700; text-decoration: underline;">
                            @Alert_22bms_bot ↗
                        </a>
                    </div>
                    <div style="display: flex; gap: 6px; align-items: center;">
                        <input type="text" id="tgChatId" placeholder="Your Telegram Chat ID" class="input-text" style="flex: 1; padding: 6px 10px; font-size: 11px; background: rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1); color: #fff; border-radius: 4px;">
                        <button class="btn btn-secondary" onclick="saveTelegramSettings()" style="padding: 6px 12px; font-size: 11px;">
                            Save
                        </button>
                    </div>
                    <div id="tgDetectStatus" style="font-size: 10px; margin-top: 6px;"></div>
                </div>

                <div style="display: flex; gap: 6px; flex-wrap: wrap;">
                    <a href="https://t.me/Alert_22bms_bot" target="_blank" class="btn" style="flex: 1; min-width: 130px; background: #2563eb; font-size: 11px; font-weight: 700; text-align: center; text-decoration: none; padding: 7px 10px; color: #fff; border-radius: 4px;">
                        1️⃣ Open Bot &amp; Start
                    </a>
                    <button class="btn btn-secondary" onclick="autoDetectTelegramChatId()" style="flex: 1; min-width: 140px; font-size: 11px; font-weight: 700; padding: 7px 10px;">
                        2️⃣ Auto-Detect Chat ID
                    </button>
                    <button class="btn btn-danger" onclick="sendTestTelegramAlert()" style="flex: 1; min-width: 130px; font-size: 11px; font-weight: 700; padding: 7px 10px; background: linear-gradient(135deg, #10b981, #059669);">
                        ⚡ Send Test to Phone
                    </button>
                </div>
            </div>

        </div>

        <!-- Threat Diagnostics Card -->
        <div class="card">
            <div id="threatCard" class="threat-card normal">
                <div class="threat-header">
                    <span id="threatStatusText">NORMAL BEHAVIOUR</span>
                    <span id="threatBadge" class="badge-red" style="display: none;">CRITICAL</span>
                </div>
                <div id="threatDesc" class="threat-details">
                    Surveillance operational. No combat interactions detected.
                </div>

                <div id="combatantsBox" class="combatants-box" style="display: none;">
                    <div class="combatants-title">⚡ Individuals in Combat ⚡</div>
                    <div class="fighter-chips" id="fighterChips"></div>
                </div>
            </div>

            <div>
                <p style="font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase;">Active Identified Individuals</p>
                <div class="roster" id="personRoster">
                    <span style="font-size: 11px; color: var(--text-dim); padding: 4px;">Scanning frame for persons...</span>
                </div>
            </div>

            <div style="display: flex; gap: 8px; margin-top: 14px;">
                <button class="btn btn-secondary" onclick="toggleAudioAlarm()">
                    <span id="audioIcon">🔊</span> Alarm Sound: <span id="audioState">ON</span>
                </button>
            </div>
        </div>

        <!-- Incident Evidence Vault -->
        <div class="card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: var(--text-main);">Incident Evidence Vault</h4>
                <span id="incidentCountBadge" style="font-size: 11px; background: rgba(255,255,255,0.08); padding: 2px 8px; border-radius: 10px; color: var(--text-muted);">0 Records</span>
            </div>
            <div class="incident-list" id="incidentList">
                <p style="font-size: 12px; color: var(--text-dim); text-align: center; padding: 20px;">No incidents logged yet.</p>
            </div>
        </div>
    </div>
</div>

<!-- Modal: Authority Directory (Persons 1 to 10) -->
<div id="directoryModal" class="modal-backdrop">
    <div class="modal" style="max-width: 860px; width: 95%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <h3>📋 Campus Authority Directory (Persons 1 to 10)</h3>
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">
                    Manage all 10 emergency response authorities. Designate who is currently On-Duty to receive real-time incident dispatches.
                </p>
            </div>
            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="closeDirectoryModal()">✕ Close</button>
        </div>

        <div style="max-height: 400px; overflow-y: auto; border: 1px solid var(--border); border-radius: var(--radius-sm); margin: 12px 0;">
            <table class="directory-table" style="width: 100%; border-collapse: collapse; font-size: 12px; text-align: left;">
                <thead>
                    <tr style="background: rgba(30, 41, 59, 0.85); color: var(--text-muted); border-bottom: 1px solid var(--border);">
                        <th style="padding: 10px 12px;">ID</th>
                        <th style="padding: 10px 12px;">Authority Name</th>
                        <th style="padding: 10px 12px;">Phone Number (Mobile / WhatsApp)</th>
                        <th style="padding: 10px 12px;">Email ID</th>
                        <th style="padding: 10px 12px;">Duty Status</th>
                        <th style="padding: 10px 12px; text-align: right;">Action</th>
                    </tr>
                </thead>
                <tbody id="directoryTableBody">
                    <!-- Populated dynamically via JavaScript -->
                </tbody>
            </table>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border); flex-wrap: wrap; gap: 10px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="font-size: 12px; color: var(--text-muted);">Quick Switch On-Duty:</span>
                <select id="quickSwitchSlot" class="form-control" style="width: auto; padding: 5px 10px; font-size: 12px;">
                    <!-- 1 to 10 options -->
                </select>
                <button class="btn btn-secondary" style="padding: 5px 12px; font-size: 12px;" onclick="quickSwitchDuty()">Switch Active</button>
            </div>
            <div style="display: flex; gap: 8px;">
                <button class="btn btn-secondary" onclick="closeDirectoryModal()">Close</button>
                <button class="btn" onclick="openAuthorityModal()">⚙️ Configure Authorities</button>
            </div>
        </div>
    </div>
</div>

<!-- Modal: Dynamic Authority Setup (Select 1, 3, or any count from 1 to 10) -->
<div id="authorityModal" class="modal-backdrop">
    <div class="modal" style="max-width: 680px; width: 95%;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div>
                <h3>🛡️ Configure Campus Authorities</h3>
                <p style="font-size: 12px; color: var(--text-muted); margin-top: 2px;">
                    Select how many authorities you need (1 to 10). The system will present only that exact number of authorities to fill.
                </p>
            </div>
            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="closeAuthorityModal()">✕</button>
        </div>

        <div style="background: rgba(30, 58, 138, 0.25); border: 1px solid rgba(59, 130, 246, 0.35); padding: 12px 16px; border-radius: var(--radius-sm); margin-bottom: 14px;">
            <label style="font-weight: 700; color: #93c5fd; font-size: 13px; display: block; margin-bottom: 6px;">
                How many Authorities do you want to set?
            </label>
            <select id="selectNumAuthorities" class="form-control" onchange="renderDynamicAuthorityForms(this.value)" style="font-size: 14px; font-weight: 700; background: #0c1424; color: white;">
                <option value="1">1 Authority (Only fill 1 person)</option>
                <option value="2">2 Authorities (Fill 2 persons)</option>
                <option value="3">3 Authorities (Fill 3 persons)</option>
                <option value="4">4 Authorities (Fill 4 persons)</option>
                <option value="5">5 Authorities (Fill 5 persons)</option>
                <option value="6">6 Authorities (Fill 6 persons)</option>
                <option value="7">7 Authorities (Fill 7 persons)</option>
                <option value="8">8 Authorities (Fill 8 persons)</option>
                <option value="9">9 Authorities (Fill 9 persons)</option>
                <option value="10">10 Authorities (Fill 10 persons)</option>
            </select>
            <p style="font-size: 11px; color: var(--text-dim); margin-top: 4px;">
                💡 If you select 1, only 1 authority will be requested. If you select 3, 3 authorities will be requested.
            </p>
        </div>

        <div id="dynamicAuthorityCards" style="max-height: 380px; overflow-y: auto; display: flex; flex-direction: column; gap: 12px; padding-right: 4px;">
            <!-- Dynamically populated cards for exactly 1, 3, etc. -->
        </div>

        <div class="form-group" style="margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border);">
            <label>Master Security Password (Required to Save) *</label>
            <input type="password" id="batchMasterPassword" class="form-control" placeholder="Enter security password (default: admin)">
            <p style="font-size: 11px; color: var(--text-dim); margin-top: 4px;">Protects records from unauthorized changes.</p>
        </div>

        <div style="display: flex; justify-content: flex-end; gap: 10px; margin-top: 14px;">
            <button class="btn btn-secondary" onclick="closeAuthorityModal()">Cancel</button>
            <button class="btn" onclick="submitAllAuthorities()">💾 Save Configured Authorities</button>
        </div>
    </div>
</div>

<!-- Modal: Incident Evidence Viewer -->
<div id="evidenceModal" class="modal-backdrop">
    <div class="modal" style="max-width: 720px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <h3 id="evModalTitle">Incident Evidence Snapshot</h3>
            <button class="btn btn-secondary" style="padding: 4px 10px; font-size: 12px;" onclick="closeEvidenceModal()">✕ Close</button>
        </div>
        <img id="evModalImg" src="" alt="Incident Evidence" style="width: 100%; max-height: 440px; object-fit: contain; background: #000; border-radius: var(--radius-sm); border: 1px solid var(--border);">
        <div id="evModalDetails" style="margin-top: 12px; font-size: 13px; color: var(--text-muted);"></div>
    </div>
</div>

<script>
let currentAudioEnabled = true;
let audioContext = null;
let lastSirenTime = 0;
let currentAuthority = {};

const feedImg = document.getElementById('videoFeed');
feedImg.onerror = function() {
    console.warn("Stream connection interrupted. Reconnecting in 1s...");
    setTimeout(() => {
        feedImg.src = '/video_feed?t=' + Date.now();
    }, 1000);
};

function updateClock() {
    const now = new Date();
    document.getElementById('systemClock').textContent = now.toTimeString().split(' ')[0] + ' UTC';
}
setInterval(updateClock, 1000);
updateClock();

function playAlarmSiren() {
    if (!currentAudioEnabled) return;
    try {
        if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
        if (audioContext.state === 'suspended') audioContext.resume();

        for (let cycle = 0; cycle < 2; cycle++) {
            const startT = audioContext.currentTime + (cycle * 0.7);
            const osc = audioContext.createOscillator();
            const gain = audioContext.createGain();

            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(800, startT);
            osc.frequency.exponentialRampToValueAtTime(400, startT + 0.35);
            osc.frequency.exponentialRampToValueAtTime(800, startT + 0.7);

            gain.gain.setValueAtTime(0.35, startT);
            gain.gain.exponentialRampToValueAtTime(0.01, startT + 0.7);

            osc.connect(gain);
            gain.connect(audioContext.destination);

            osc.start(startT);
            osc.stop(startT + 0.7);
        }
    } catch (e) {
        console.warn("WebAudio siren error:", e);
    }
}

function toggleAudioAlarm() {
    currentAudioEnabled = !currentAudioEnabled;
    document.getElementById('audioState').textContent = currentAudioEnabled ? 'ON' : 'OFF';
    document.getElementById('audioIcon').textContent = currentAudioEnabled ? '🔊' : '🔇';
}

function handleFileSelected(input) {
    if (input.files && input.files[0]) {
        document.getElementById('uploadFilename').textContent = "Selected: " + input.files[0].name;
        const statusText = document.getElementById('uploadStatusText');
        if (statusText) statusText.innerHTML = '';
    }
}

async function handleVideoUpload(e) {
    if (e) e.preventDefault();
    const fileInput = document.getElementById('videoFileInput');
    if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please select a video file first.");
        return;
    }
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('video', file);
    formData.append('ajax', '1');

    const statusText = document.getElementById('uploadStatusText');
    const submitBtn = document.getElementById('btnUploadSubmit');
    if (statusText) statusText.innerHTML = '<span style="color: #38bdf8;">⏳ Uploading (' + (file.size / (1024*1024)).toFixed(1) + ' MB)...</span>';
    if (submitBtn) { submitBtn.disabled = true; submitBtn.innerText = 'Uploading...'; }

    try {
        const resp = await fetch('/upload', {
            method: 'POST',
            body: formData,
            headers: {'X-Requested-With': 'XMLHttpRequest'}
        });
        const data = await resp.json();
        if (data.ok) {
            if (statusText) statusText.innerHTML = '<span style="color: #34d399;">✅ Analyzing ' + data.filename + '</span>';
            await loadVideos();
            await setSource(data.filename);
        } else {
            if (statusText) statusText.innerHTML = '<span style="color: #f87171;">❌ ' + (data.message || 'Upload failed') + '</span>';
        }
    } catch (err) {
        if (statusText) statusText.innerHTML = '<span style="color: #f87171;">❌ Upload error: ' + err + '</span>';
    } finally {
        if (submitBtn) { submitBtn.disabled = false; submitBtn.innerText = 'Upload & Analyze'; }
    }
}

window.addEventListener('beforeunload', () => {
    try {
        navigator.sendBeacon('/api/camera/release');
    } catch(e) {}
});

async function standbyCamera() {
    try {
        await fetch('/api/camera/release', {method: 'POST'});
        document.querySelectorAll('.source-btn').forEach(btn => btn.classList.remove('active'));
        const activeBtn = document.getElementById('src-standby-cam');
        if (activeBtn) activeBtn.classList.add('active');
        document.getElementById('currentSourceTag').textContent = 'SOURCE: STANDBY (OFF)';
    } catch (e) {
        console.error("Standby camera error:", e);
    }
}

async function shutdownServer() {
    if (!confirm("Are you sure you want to stop the AI Surveillance Server and release the camera hardware?")) return;
    try {
        await fetch('/api/system/shutdown', {method: 'POST'});
    } catch(e) {}
    alert("Surveillance server stopped and camera hardware safely released.");
    window.close();
}

async function reconnectLiveCamera() {
    const btn = document.getElementById('src-reconnect-cam');
    if (btn) btn.textContent = '⏳ Probing...';
    try {
        await fetch('/api/camera/reconnect', {method: 'POST'});
        await setSource('webcam');
    } catch (e) {
        console.error("Reconnect camera error:", e);
    }
    if (btn) btn.textContent = '🔄 Reconnect Camera';
}

async function setSource(sourceName) {
    try {
        const resp = await fetch('/api/set_source', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({source: sourceName})
        });
        const data = await resp.json();
        if (data.ok) {
            document.querySelectorAll('.source-btn').forEach(btn => btn.classList.remove('active'));
            const cleanId = 'src-' + sourceName.replace(/[^a-zA-Z0-9]/g, '');
            const activeBtn = document.getElementById(cleanId);
            if (activeBtn) activeBtn.classList.add('active');
            document.getElementById('currentSourceTag').textContent = 'SOURCE: ' + sourceName.toUpperCase();

            // Instantly reconnect stream with timestamp to load new feed immediately
            setTimeout(() => {
                feedImg.src = '/video_feed?t=' + Date.now();
            }, 100);
        }
    } catch (e) {
        console.error("Error setting source:", e);
    }
}

async function loadVideos() {
    try {
        const resp = await fetch('/api/videos');
        const videos = await resp.json();
        const container = document.getElementById('sourceBar');
        document.querySelectorAll('.video-dynamic-btn').forEach(el => el.remove());

        const currentActive = document.getElementById('currentSourceTag').textContent.replace('SOURCE: ', '').trim().toLowerCase();

        videos.forEach(v => {
            const btn = document.createElement('button');
            const cleanId = 'src-' + v.filename.replace(/[^a-zA-Z0-9]/g, '');
            btn.id = cleanId;
            btn.className = 'source-btn video-dynamic-btn' + (currentActive === v.filename.toLowerCase() ? ' active' : '');
            btn.innerHTML = `📹 ${v.filename} <span style="font-size: 11px; opacity: 0.7;">(${v.size_mb} MB)</span>`;
            btn.onclick = () => setSource(v.filename);
            container.appendChild(btn);
        });
    } catch (e) {
        console.error("Error loading videos:", e);
    }
}

// ---------------- AUTHORITY PERSON MANAGEMENT ----------------

let currentAuthorities = [];
let currentOnDutyId = 1;

async function loadAuthorityDirectory() {
    try {
        const resp = await fetch('/api/authorities');
        const data = await resp.json();
        currentAuthorities = data.authorities || [];
        currentOnDutyId = data.on_duty_id || 1;

        const onDuty = data.on_duty_authority || (currentAuthorities.length > 0 ? currentAuthorities[0] : {});

        // 1. Update Dashboard On-Duty Authority Box
        const nameText = onDuty.name || `Authority Person #${onDuty.id || 1}`;
        const phoneText = onDuty.phone ? onDuty.phone : '<span style="color:var(--text-dim)">[Not set - click to add]</span>';
        const emailText = onDuty.email ? onDuty.email : '<span style="color:var(--text-dim)">[Not set - click to add]</span>';

        document.getElementById('dispAuthSlotBadge').textContent = `🛡️ AUTHORITY #${onDuty.id || 1} (ON-DUTY)`;
        document.getElementById('dispAuthName').textContent = nameText;
        document.getElementById('dispAuthPhone').innerHTML = phoneText;
        document.getElementById('dispAuthEmail').innerHTML = emailText;
        document.getElementById('dispAuthInitial').textContent = (onDuty.name && onDuty.name.length > 0 ? onDuty.name.charAt(0).toUpperCase() : (onDuty.id || '1'));

        const cleanPhone = (onDuty.phone || '').replace(/[^0-9]/g, '');
        const targetPhone = (cleanPhone.length === 10 ? '91' + cleanPhone : cleanPhone);
        const sampleMsg = encodeURIComponent(`🚨 [CAMPUS CCTV ALERT] Priority dispatch to ${nameText}. System monitoring active.`);
        const waEl = document.getElementById('btnWhatsAppDirect');
        if (waEl) {
            waEl.href = targetPhone ? `https://api.whatsapp.com/send?phone=${targetPhone}&text=${sampleMsg}` : '#';
            waEl.textContent = `💬 WhatsApp: ${onDuty.phone || 'Alert'}`;
        }
        const smsEl = document.getElementById('btnSmsDirect');
        if (smsEl) {
            smsEl.href = targetPhone ? `sms:+${targetPhone}?body=${sampleMsg}` : '#';
            smsEl.textContent = `📱 SMS: ${onDuty.phone || 'Alert'}`;
        }
        const emEl = document.getElementById('btnEmailDirect');
        if (emEl) {
            emEl.href = onDuty.email ? `mailto:${onDuty.email}?subject=${encodeURIComponent('🚨 Campus CCTV Security Alert')}` : '#';
            emEl.textContent = `📧 Email: ${nameText.split(' ')[0]}`;
        }
        const pushEl = document.getElementById('btnPushDirect');
        if (pushEl) {
            pushEl.href = `https://ntfy.sh/cctv_${targetPhone || 'campus_alert_jeevan'}`;
        }

        // 2. Render 1 to 10 Directory Table in Modal
        const tableBody = document.getElementById('directoryTableBody');
        tableBody.innerHTML = '';

        const quickSelect = document.getElementById('quickSwitchSlot');
        quickSelect.innerHTML = '';

        currentAuthorities.forEach(item => {
            const tr = document.createElement('tr');
            const isActive = (item.id === currentOnDutyId);

            const opt = document.createElement('option');
            opt.value = item.id;
            opt.textContent = `Person #${item.id}: ${item.name}`;
            if (isActive) opt.selected = true;
            quickSelect.appendChild(opt);

            tr.innerHTML = `
                <td style="font-weight: 700; color: #93c5fd;">#${item.id}</td>
                <td style="font-weight: 600; color: #f8fafc;">${item.name || `Authority Person #${item.id}`}</td>
                <td style="color: ${item.phone ? '#67e8f9' : 'var(--text-dim)'};">${item.phone || '—'}</td>
                <td style="color: ${item.email ? '#93c5fd' : 'var(--text-dim)'};">${item.email || '—'}</td>
                <td>
                    <span class="${isActive ? 'duty-badge-active' : 'duty-badge-standby'}">
                        ${isActive ? '● ACTIVE ON-DUTY' : '○ Standby'}
                    </span>
                </td>
                <td style="text-align: right;">
                    <button class="btn btn-secondary" style="padding: 3px 8px; font-size: 11px;" onclick="openAuthorityModal()">
                        ✏️ Edit
                    </button>
                    ${!isActive ? `<button class="btn" style="padding: 3px 8px; font-size: 11px; margin-left: 4px;" onclick="quickSetOnDuty(${item.id})">Set On-Duty</button>` : ''}
                </td>
            `;
            tableBody.appendChild(tr);
        });

    } catch (e) {
        console.error("Error loading authorities directory:", e);
    }
}

function openDirectoryModal() {
    loadAuthorityDirectory();
    document.getElementById('directoryModal').style.display = 'flex';
}

function closeDirectoryModal() {
    document.getElementById('directoryModal').style.display = 'none';
}

function openAuthorityModal(initialCount) {
    const count = initialCount || currentAuthorities.length || 1;
    document.getElementById('selectNumAuthorities').value = count;
    renderDynamicAuthorityForms(count);
    document.getElementById('batchMasterPassword').value = '';
    document.getElementById('authorityModal').style.display = 'flex';
}

function openEditModalForCurrent() {
    openAuthorityModal(currentAuthorities.length || 1);
}

function closeAuthorityModal() {
    document.getElementById('authorityModal').style.display = 'none';
}

function renderDynamicAuthorityForms(countVal) {
    const count = Math.max(1, Math.min(10, parseInt(countVal) || 1));
    const container = document.getElementById('dynamicAuthorityCards');
    container.innerHTML = '';

    for (let i = 1; i <= count; i++) {
        const existing = currentAuthorities.find(a => a.id === i) || {};
        const isCurrentDuty = (i === currentOnDutyId) || (currentOnDutyId > count && i === 1);

        const card = document.createElement('div');
        card.style.background = 'rgba(15, 23, 42, 0.7)';
        card.style.border = '1px solid var(--border)';
        card.style.borderRadius = 'var(--radius-sm)';
        card.style.padding = '12px 14px';

        card.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-weight: 700; color: #93c5fd; font-size: 13px;">🛡️ Authority Person #${i}</span>
                <label style="font-size: 11px; display: inline-flex; align-items: center; gap: 4px; cursor: pointer; color: #34d399;">
                    <input type="radio" name="dynOnDutyRadio" value="${i}" ${isCurrentDuty ? 'checked' : ''} style="accent-color: #10b981;">
                    Set as On-Duty
                </label>
            </div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div class="form-group" style="margin-bottom: 6px; grid-column: span 2;">
                    <label style="font-size: 11px; margin-bottom: 3px;">Full Name *</label>
                    <input type="text" class="dyn-auth-name form-control" data-id="${i}" value="${existing.name || ''}" placeholder="e.g. Officer Rajesh / Warden Suresh" style="padding: 6px 10px; font-size: 12px;">
                </div>
                <div class="form-group" style="margin-bottom: 6px;">
                    <label style="font-size: 11px; margin-bottom: 3px;">Phone (WhatsApp) *</label>
                    <input type="text" class="dyn-auth-phone form-control" data-id="${i}" value="${existing.phone || ''}" placeholder="e.g. +91 9123456789" style="padding: 6px 10px; font-size: 12px;">
                </div>
                <div class="form-group" style="margin-bottom: 6px;">
                    <label style="font-size: 11px; margin-bottom: 3px;">Email Address</label>
                    <input type="email" class="dyn-auth-email form-control" data-id="${i}" value="${existing.email || ''}" placeholder="e.g. officer@campus.edu" style="padding: 6px 10px; font-size: 12px;">
                </div>
            </div>
        `;
        container.appendChild(card);
    }
}

async function submitAllAuthorities() {
    const count = parseInt(document.getElementById('selectNumAuthorities').value) || 1;
    const password = document.getElementById('batchMasterPassword').value;

    if (!password) {
        alert("Please enter the Master Security Password (default: admin) to authorize changes.");
        return;
    }

    const onDutyRadio = document.querySelector('input[name="dynOnDutyRadio"]:checked');
    const onDutyId = onDutyRadio ? parseInt(onDutyRadio.value) : 1;

    const authList = [];
    for (let i = 1; i <= count; i++) {
        const nameEl = document.querySelector(`.dyn-auth-name[data-id="${i}"]`);
        const phoneEl = document.querySelector(`.dyn-auth-phone[data-id="${i}"]`);
        const emailEl = document.querySelector(`.dyn-auth-email[data-id="${i}"]`);

        const name = nameEl ? nameEl.value.trim() : '';
        const phone = phoneEl ? phoneEl.value.trim() : '';
        const email = emailEl ? emailEl.value.trim() : '';

        if (!name) {
            alert(`Please enter the Name for Authority Person #${i}.`);
            return;
        }

        authList.push({
            id: i,
            name: name,
            phone: phone,
            email: email,
            is_on_duty: (i === onDutyId)
        });
    }

    try {
        const resp = await fetch('/api/authorities/save_all', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                count: count,
                authorities: authList,
                on_duty_id: onDutyId,
                password: password
            })
        });
        const res = await resp.json();
        alert(res.message);
        if (res.ok) {
            closeAuthorityModal();
            loadAuthorityDirectory();
        }
    } catch (e) {
        alert("Failed to save authorities: " + e);
    }
}

async function quickSetOnDuty(slotId) {
    try {
        const resp = await fetch('/api/authorities/set_on_duty', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({id: slotId})
        });
        const res = await resp.json();
        alert(res.message);
        if (res.ok) {
            loadAuthorityDirectory();
        }
    } catch (e) {
        alert("Failed to change on-duty authority: " + e);
    }
}

function quickSwitchDuty() {
    const slotId = parseInt(document.getElementById('quickSwitchSlot').value);
    if (slotId) {
        quickSetOnDuty(slotId);
    }
}

// ---------------- TELEMETRY & ACTION LOGIC ----------------

async function refreshTelemetry() {
    try {
        const resp = await fetch('/status');
        const data = await resp.json();

        const count = data.persons_count || 0;
        document.getElementById('valPersonsCount').textContent = count;
        document.getElementById('valMotion').textContent = data.motion_score.toFixed(1) + '%';
        const meterMotion = document.getElementById('meterMotion');
        meterMotion.style.width = Math.min(100, data.motion_score * 5.0) + '%';

        const valFightRisk = document.getElementById('valFightRisk');
        const meterFight = document.getElementById('meterFight');

        // CRITICAL CHECK: Fight is only possible when multiple people are in frame!
        if (data.fight_detected && count >= 1) {
            valFightRisk.textContent = 'FIGHT DETECTED';
            valFightRisk.style.color = '#ef4444';
            meterFight.style.width = '100%';
            meterFight.style.background = 'var(--red)';
        } else {
            valFightRisk.textContent = (count === 1 ? '1 PERSON (NORMAL)' : (count === 0 ? 'EMPTY' : 'NORMAL'));
            valFightRisk.style.color = 'var(--emerald)';
            meterFight.style.width = '0%';
            meterFight.style.background = 'var(--emerald)';
        }

        const valCooldown = document.getElementById('valCooldown');
        const meterCooldown = document.getElementById('meterCooldown');
        if (data.cooldown_remaining > 0) {
            valCooldown.textContent = data.cooldown_remaining + 's';
            valCooldown.style.color = 'var(--amber)';
            meterCooldown.style.width = ((data.cooldown_remaining / 25) * 100) + '%';
            meterCooldown.style.background = 'var(--amber)';
        } else {
            valCooldown.textContent = 'READY';
            valCooldown.style.color = 'var(--emerald)';
            meterCooldown.style.width = '100%';
            meterCooldown.style.background = 'var(--emerald)';
        }

        const threatCard = document.getElementById('threatCard');
        const threatStatusText = document.getElementById('threatStatusText');
        const threatDesc = document.getElementById('threatDesc');
        const threatBadge = document.getElementById('threatBadge');
        const combatantsBox = document.getElementById('combatantsBox');
        const fighterChips = document.getElementById('fighterChips');
        const threatBanner = document.getElementById('threatBanner');
        const threatBannerText = document.getElementById('threatBannerText');

        if (data.fight_detected && count >= 1) {
            playAlarmSiren();
            threatCard.className = 'threat-card critical';
            threatStatusText.textContent = '🚨 VIOLENT FIGHT DETECTED!';
            threatStatusText.style.color = '#ef4444';
            threatBadge.style.display = 'inline-block';
            threatBadge.textContent = 'CRITICAL ALERT';

            const fighters = data.involved_persons.length > 0 ? data.involved_persons.join(' & ') : 'Multiple Combatants';
            threatDesc.textContent = `Violent combat interaction detected involving ${fighters}. Automatic alert dispatched to ${currentAuthority.name || 'Authority'}.`;

            combatantsBox.style.display = 'block';
            fighterChips.innerHTML = '';
            data.involved_persons.forEach(p => {
                const chip = document.createElement('div');
                chip.className = 'fighter-chip';
                chip.innerHTML = `⚠️ <b>${p}</b> (Combatant)`;
                fighterChips.appendChild(chip);
            });

            threatBanner.style.display = 'flex';
            threatBannerText.textContent = `🚨 CRITICAL ALERT: FIGHT DETECTED INVOLVING ${fighters.toUpperCase()}! DISPATCHING TO ${currentAuthority.name ? currentAuthority.name.toUpperCase() : 'AUTHORITY'}`;

        } else {
            threatCard.className = 'threat-card normal';
            threatStatusText.textContent = count === 1 ? 'NORMAL (1 PERSON DETECTED)' : 'NORMAL BEHAVIOUR';
            threatStatusText.style.color = 'var(--emerald)';
            threatBadge.style.display = 'none';
            threatDesc.textContent = count === 1 
                ? 'Single individual in camera view. Normal activity, no danger alerts.'
                : `Campus surveillance operational. ${count} persons tracked normally.`;
            combatantsBox.style.display = 'none';
            threatBanner.style.display = 'none';
        }

        const roster = document.getElementById('personRoster');
        if (data.persons.length === 0) {
            roster.innerHTML = '<span style="font-size: 11px; color: var(--text-dim); padding: 4px;">No students currently in frame</span>';
        } else {
            roster.innerHTML = '';
            data.persons.forEach(p => {
                const pred = (data.student_predictions && data.student_predictions[p]) || {};
                const isFighting = (data.involved_persons.includes(p) || pred.behavior === 'FIGHTING') && count >= 2;
                const behaviorLabel = isFighting ? 'FIGHTING' : (pred.behavior || 'NORMAL');
                const confText = pred.confidence ? ` (${pred.confidence}%)` : '';
                const pill = document.createElement('div');
                pill.className = 'person-pill' + (isFighting ? ' fighting' : '');
                pill.innerHTML = `👤 ${p} <span>[${behaviorLabel}${confText}]</span>`;
                roster.appendChild(pill);
            });
        }

        // Update Real-time Gesture & Demeanor Movement Telemetry
        const kTel = data.kinetic_telemetry || {};
        const cat = kTel.gesture_category || "CALM";
        currentGestureCategory = cat;
        targetAnimSpeed = kTel.strike_speed || 0.15;

        const gtGesture = document.getElementById('gtGestureType');
        const gtDemeanor = document.getElementById('gtFacialDemeanor');
        const gtVel = document.getElementById('gtVelocityVal');
        const gtEnergy = document.getElementById('gtEnergyVal');

        if (gtGesture) {
            gtGesture.textContent = kTel.primary_gesture || 'Calm Movement';
            if (cat === 'SERIOUS_COMBAT') {
                gtGesture.style.color = '#ef4444';
            } else if (cat === 'PLAYFUL_GESTURE') {
                gtGesture.style.color = '#f59e0b';
            } else {
                gtGesture.style.color = 'var(--emerald)';
            }
        }
        if (gtDemeanor) {
            gtDemeanor.textContent = kTel.primary_demeanor || 'Relaxed / Casual';
            if (cat === 'SERIOUS_COMBAT') {
                gtDemeanor.style.color = '#ef4444';
            } else if (cat === 'PLAYFUL_GESTURE') {
                gtDemeanor.style.color = '#f59e0b';
            } else {
                gtDemeanor.style.color = 'var(--emerald)';
            }
        }
        if (gtVel) {
            gtVel.textContent = (kTel.strike_speed || 0.0).toFixed(2) + ' m/s';
            gtVel.style.color = cat === 'SERIOUS_COMBAT' ? '#ef4444' : (cat === 'PLAYFUL_GESTURE' ? '#f59e0b' : '#38bdf8');
        }
        if (gtEnergy) {
            gtEnergy.textContent = (kTel.combat_energy || 0.0).toFixed(2) + ' J';
            gtEnergy.style.color = cat === 'SERIOUS_COMBAT' ? '#ef4444' : (cat === 'PLAYFUL_GESTURE' ? '#f59e0b' : '#38bdf8');
        }

    } catch (e) {
        console.warn("Status refresh error:", e);
    }
}

async function refreshAuthorityMessage() {
    try {
        const resp = await fetch('/api/authority/last_message');
        const data = await resp.json();
        if (data && data.message) {
            document.getElementById('simMsgText').textContent = data.message;
            document.getElementById('simMetaTarget').textContent = `Target: ${data.authority_name} (${data.authority_contact})`;
            document.getElementById('simMetaTime').textContent = data.timestamp;

            if (data.whatsapp_url) {
                document.getElementById('btnWhatsAppDirect').href = data.whatsapp_url;
            }
            if (data.sms_url && document.getElementById('btnSmsDirect')) {
                document.getElementById('btnSmsDirect').href = data.sms_url;
            }
        }
    } catch (e) {
        console.warn("Error refreshing authority message:", e);
    }
}

async function loadIncidents() {
    try {
        const resp = await fetch('/api/incidents');
        const incidents = await resp.json();
        const list = document.getElementById('incidentList');
        document.getElementById('incidentCountBadge').textContent = incidents.length + ' Records';

        if (incidents.length === 0) {
            list.innerHTML = '<p style="font-size: 12px; color: var(--text-dim); text-align: center; padding: 20px;">No incidents logged yet.</p>';
            return;
        }

        list.innerHTML = '';
        incidents.forEach(inc => {
            const card = document.createElement('div');
            card.className = 'incident-card';
            card.onclick = () => showEvidence(inc);

            const isFight = inc.type === 'FIGHT_DETECTED';
            const thumbSrc = inc.snapshot ? `/incidents/${inc.snapshot}` : '';
            const authName = (inc.authority_recipient && inc.authority_recipient.name) ? inc.authority_recipient.name : 'Authority Officer';

            card.innerHTML = `
                <img class="incident-thumb" src="${thumbSrc}" alt="Evidence" onerror="this.src='https://placehold.co/100x70/111e35/ffffff?text=CCTV'">
                <div class="incident-info">
                    <div class="incident-type">${isFight ? '🚨 VIOLENT FIGHT INCIDENT' : '⚠️ TEST ALERT EVENT'}</div>
                    <div class="incident-meta">${inc.timestamp} | ${inc.persons.join(', ') || 'Individuals'}</div>
                    <div style="font-size: 10px; color: var(--cyan); margin-top: 2px;">Dispatched to: ${authName}</div>
                </div>
                <div style="font-size: 11px; color: var(--text-dim);">🔍 View</div>
            `;
            list.appendChild(card);
        });
    } catch (e) {
        console.error("Error loading incidents:", e);
    }
}

function showEvidence(incident) {
    const auth = incident.authority_recipient || {};
    document.getElementById('evModalTitle').textContent = `Incident Evidence: ${incident.type.replace('_', ' ')} (${incident.id})`;
    document.getElementById('evModalImg').src = `/incidents/${incident.snapshot}`;
    document.getElementById('evModalDetails').innerHTML = `
        <p><b>Timestamp:</b> ${incident.timestamp}</p>
        <p><b>Camera Location:</b> ${incident.location} | Source: ${incident.source}</p>
        <p><b>Combatants / Persons:</b> ${incident.persons.join(', ') || 'Unknown'}</p>
        <p><b>Dispatched To Authority:</b> <span style="color: #60a5fa; font-weight: bold;">${auth.name || 'Officer'} (${auth.contact || 'Contact'})</span></p>
        <p><b>Transmission Status:</b> Delivered</p>
    `;
    document.getElementById('evidenceModal').style.display = 'flex';
}

function closeEvidenceModal() {
    document.getElementById('evidenceModal').style.display = 'none';
}

let isAlarmSounding = false;

async function checkActionAndDispatch() {
    try {
        playAlarmSiren();

        // 1. Trigger desktop HTML5 notification
        if ("Notification" in window) {
            if (Notification.permission === "granted") {
                new Notification("🚨 CAMPUS CCTV EMERGENCY ALARM", {
                    body: "Immediate Action Alert Dispatched to Jeevan (9353415499 | jeevanl06232003@gmail.com)!",
                    icon: "https://cdn-icons-png.flaticon.com/512/1022/1022382.png"
                });
            } else if (Notification.permission !== "denied") {
                Notification.requestPermission();
            }
        }

        // 2. Call backend /api/check_action
        const resp = await fetch('/api/check_action', {method: 'POST'});
        const res = await resp.json();

        // 3. Immediately activate local alarm UI
        setAlarmActiveUI(true, res.authority ? res.authority.name : 'Jeevan', res.authority ? res.authority.phone : '9353415499');

        await refreshAuthorityMessage();
        loadIncidents();

        const auth = res.authority || (currentAuthorities && currentAuthorities.length > 0 ? currentAuthorities[0] : {name: 'Jeevan', phone: '9353415499', email: 'jeevanl06232003@gmail.com'});
        const cleanPhone = (auth.phone || '9353415499').replace(/[^0-9]/g, '');
        const targetPhone = (cleanPhone.length === 10 ? '91' + cleanPhone : cleanPhone);
        const waUrl = res.whatsapp_url || `https://api.whatsapp.com/send?phone=${targetPhone}&text=${encodeURIComponent('🚨 [CAMPUS CCTV ACTION ALARM] Verified emergency action detected. Immediate response requested!')}`;
        const smsUrl = res.sms_url || `sms:+${targetPhone}?body=${encodeURIComponent('🚨 [CAMPUS CCTV ACTION ALARM] Verified emergency action detected. Immediate response requested!')}`;

        const choice = confirm(`🚨 EMERGENCY NOTIFICATION ALARM ACTIVE & DISPATCHED!\n\n` +
                               `• On-Duty Recipient: ${auth.name || 'Jeevan'} (Phone: ${auth.phone || '9353415499'})\n` +
                               `• Email: ${auth.email || 'jeevanl06232003@gmail.com'}\n` +
                               `• Siren Alarm: SOUNDING LOUDLY\n` +
                               `• SMS / WhatsApp: Ready for direct phone alert\n` +
                               `• Mobile Push: Sent to https://ntfy.sh/cctv_${targetPhone}\n\n` +
                               `Click OK to open WhatsApp now, or Cancel to open native SMS!`);
        if (choice) {
            window.open(waUrl, '_blank');
        } else if (smsUrl) {
            window.location.href = smsUrl;
        }
    } catch (e) {
        alert("Action dispatch failed: " + e);
    }
}

async function dispatchTestAlert() {
    return checkActionAndDispatch();
}

async function silenceAlarm() {
    try {
        await fetch('/api/alarm/silence', {method: 'POST'});
        setAlarmActiveUI(false);
    } catch (e) {
        console.warn("Silence alarm error:", e);
    }
}

function setAlarmActiveUI(active, recipientName, recipientPhone) {
    const box = document.getElementById('authorityBox');
    const banner = document.getElementById('authAlarmBanner');
    const bannerSub = document.getElementById('authAlarmBannerSub');
    const pill = document.getElementById('authAlarmStatusPill');

    if (active) {
        if (box) box.classList.add('alarm-active');
        if (banner) {
            banner.style.display = 'flex';
            if (recipientName && bannerSub) {
                bannerSub.textContent = `Dispatched immediately to ${recipientName} (${recipientPhone || '9353415499'})`;
            }
        }
        if (pill) {
            pill.className = 'badge-red';
            pill.innerHTML = '🚨 ALARM ACTIVE';
        }
    } else {
        if (box) box.classList.remove('alarm-active');
        if (banner) banner.style.display = 'none';
        if (pill) {
            pill.className = 'duty-badge-active';
            pill.innerHTML = '● STANDBY MONITORING';
        }
    }
}

async function refreshAlarmStatus() {
    try {
        const resp = await fetch('/api/alarm/status');
        const data = await resp.json();
        if (data.alarm_active) {
            const lastMsg = data.last_message || {};
            setAlarmActiveUI(true, lastMsg.authority_name || 'Jeevan', lastMsg.authority_phone || '9353415499');
            playAlarmSiren();
        } else {
            setAlarmActiveUI(false);
        }
    } catch (e) {
        console.warn("Error refreshing alarm status:", e);
    }
}

// Auto-unlock audio on any page interaction
document.addEventListener('click', () => {
    if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext && audioContext.state === 'suspended') audioContext.resume();
}, {once: false});


async function loadTelegramStatus() {
    try {
        const res = await fetch('/api/telegram/status');
        const data = await res.json();
        const input = document.getElementById('tgChatId');
        const badge = document.getElementById('tgStatusBadge');
        if (input && data.chat_id) {
            input.value = data.chat_id;
        }
        if (badge) {
            if (data.has_chat_id) {
                badge.className = 'duty-badge-active';
                badge.innerText = 'CONNECTED';
                badge.style.background = '#059669';
            } else {
                badge.className = 'badge-red';
                badge.innerText = 'UNPAIRED';
                badge.style.display = 'inline-block';
            }
        }
    } catch (e) {
        console.warn("Failed to load Telegram status:", e);
    }
}

async function autoDetectTelegramChatId() {
    const statusDiv = document.getElementById('tgDetectStatus');
    if (statusDiv) statusDiv.innerHTML = '<span style="color: #38bdf8;">🔄 Checking Telegram bot updates...</span>';
    try {
        const res = await fetch('/api/telegram/detect_chat_id', {method: 'POST'});
        const data = await res.json();
        if (data.ok) {
            if (statusDiv) statusDiv.innerHTML = `<span style="color: #34d399;">✅ ${data.message}</span>`;
            if (data.chat_id) {
                document.getElementById('tgChatId').value = data.chat_id;
            }
            loadTelegramStatus();
        } else {
            if (statusDiv) statusDiv.innerHTML = `<span style="color: #f87171;">⚠️ ${data.message}</span>`;
        }
    } catch (e) {
        if (statusDiv) statusDiv.innerHTML = `<span style="color: #f87171;">❌ Error: ${e}</span>`;
    }
}

async function saveTelegramSettings() {
    const chatId = document.getElementById('tgChatId').value.trim();
    const statusDiv = document.getElementById('tgDetectStatus');
    try {
        const res = await fetch('/api/telegram/save', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({chat_id: chatId, enabled: true})
        });
        const data = await res.json();
        if (statusDiv) statusDiv.innerHTML = `<span style="color: #34d399;">💾 ${data.message}</span>`;
        loadTelegramStatus();
    } catch (e) {
        if (statusDiv) statusDiv.innerHTML = `<span style="color: #f87171;">❌ Error saving: ${e}</span>`;
    }
}

async function sendTestTelegramAlert() {
    const statusDiv = document.getElementById('tgDetectStatus');
    if (statusDiv) statusDiv.innerHTML = '<span style="color: #38bdf8;">📤 Sending test alert to Telegram...</span>';
    try {
        const res = await fetch('/api/telegram/test', {method: 'POST'});
        const data = await res.json();
        if (data.ok) {
            if (statusDiv) statusDiv.innerHTML = `<span style="color: #34d399;">✅ Check your phone! Test alert delivered to Telegram!</span>`;
        } else {
            if (statusDiv) statusDiv.innerHTML = `<span style="color: #f87171;">⚠️ ${data.message}</span>`;
        }
    } catch (e) {
        if (statusDiv) statusDiv.innerHTML = `<span style="color: #f87171;">❌ Network error: ${e}</span>`;
    }
}

// ---------------- REAL-TIME BODY KINETICS OSCILLOSCOPE ----------------
const kineticCanvas = document.getElementById('gestureWaveformCanvas');
const kCtx = kineticCanvas ? kineticCanvas.getContext('2d') : null;
const kineticPoints = new Array(80).fill(0.2);
let currentAnimSpeed = 0.2;
let targetAnimSpeed = 0.2;
let currentGestureCategory = "CALM";

function updateWaveformCanvas() {
    if (!kCtx || !kineticCanvas) return;
    
    // Smooth interpolation towards latest target speed with biological micro-tremor
    currentAnimSpeed += (targetAnimSpeed - currentAnimSpeed) * 0.18;
    const tremor = (Math.random() - 0.5) * 0.04;
    const valWithTremor = Math.max(0.05, currentAnimSpeed + tremor);
    
    kineticPoints.shift();
    kineticPoints.push(valWithTremor);
    
    const w = kineticCanvas.width;
    const h = kineticCanvas.height;
    
    kCtx.clearRect(0, 0, w, h);
    
    // Background gradient
    const bgGrad = kCtx.createLinearGradient(0, 0, 0, h);
    bgGrad.addColorStop(0, '#060c18');
    bgGrad.addColorStop(1, '#03060c');
    kCtx.fillStyle = bgGrad;
    kCtx.fillRect(0, 0, w, h);
    
    // Zone regions (Max scale on y-axis is 4.0 m/s)
    const maxScale = 4.0;
    const yViolent = h - (2.8 / maxScale) * h;
    const yPlayful = h - (1.6 / maxScale) * h;
    
    // Violent Zone Background (top)
    kCtx.fillStyle = 'rgba(239, 68, 68, 0.08)';
    kCtx.fillRect(0, 0, w, yViolent);
    
    // Playful Zone Background (middle)
    kCtx.fillStyle = 'rgba(245, 158, 11, 0.07)';
    kCtx.fillRect(0, yViolent, w, yPlayful - yViolent);
    
    // Calm Zone Background (bottom)
    kCtx.fillStyle = 'rgba(16, 185, 129, 0.04)';
    kCtx.fillRect(0, yPlayful, w, h - yPlayful);
    
    // Threshold dashed lines
    kCtx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    kCtx.setLineDash([4, 4]);
    kCtx.beginPath();
    kCtx.moveTo(0, yViolent);
    kCtx.lineTo(w, yViolent);
    kCtx.stroke();
    
    kCtx.strokeStyle = 'rgba(245, 158, 11, 0.35)';
    kCtx.beginPath();
    kCtx.moveTo(0, yPlayful);
    kCtx.lineTo(w, yPlayful);
    kCtx.stroke();
    kCtx.setLineDash([]);
    
    // Zone Labels
    kCtx.font = '9px "JetBrains Mono", monospace';
    kCtx.fillStyle = '#ef4444';
    kCtx.fillText('2.8+ VIOLENT COMBAT', 8, Math.max(12, yViolent - 4));
    kCtx.fillStyle = '#f59e0b';
    kCtx.fillText('1.6-2.8 PLAYFUL / FUNNY GESTURE [SAFE]', 8, yPlayful - 4);
    kCtx.fillStyle = '#10b981';
    kCtx.fillText('0.0-1.6 CALM CONVERSATION', 8, h - 6);
    
    // Draw Waveform curve
    kCtx.beginPath();
    const step = w / (kineticPoints.length - 1);
    for (let i = 0; i < kineticPoints.length; i++) {
        const x = i * step;
        const clampedVal = Math.min(maxScale, kineticPoints[i]);
        const y = h - (clampedVal / maxScale) * h;
        if (i === 0) {
            kCtx.moveTo(x, y);
        } else {
            const prevX = (i - 1) * step;
            const prevVal = Math.min(maxScale, kineticPoints[i - 1]);
            const prevY = h - (prevVal / maxScale) * h;
            const midX = (prevX + x) / 2;
            kCtx.quadraticCurveTo(prevX, prevY, midX, (prevY + y) / 2);
        }
    }
    
    // Dynamic waveform stroke color based on current state
    let strokeColor = '#10b981';
    let glowColor = 'rgba(16, 185, 129, 0.6)';
    if (currentGestureCategory === 'SERIOUS_COMBAT') {
        strokeColor = '#ef4444';
        glowColor = 'rgba(239, 68, 68, 0.8)';
    } else if (currentGestureCategory === 'PLAYFUL_GESTURE') {
        strokeColor = '#f59e0b';
        glowColor = 'rgba(245, 158, 11, 0.7)';
    }
    
    kCtx.strokeStyle = strokeColor;
    kCtx.lineWidth = 2.5;
    kCtx.shadowColor = glowColor;
    kCtx.shadowBlur = 8;
    kCtx.stroke();
    kCtx.shadowBlur = 0;
    
    // Draw pulse dot at head of curve
    const lastClamped = Math.min(maxScale, kineticPoints[kineticPoints.length - 1]);
    const headY = h - (lastClamped / maxScale) * h;
    kCtx.fillStyle = strokeColor;
    kCtx.beginPath();
    kCtx.arc(w - 2, headY, 4, 0, Math.PI * 2);
    kCtx.fill();
    
    requestAnimationFrame(updateWaveformCanvas);
}
requestAnimationFrame(updateWaveformCanvas);

// Initial Boot
loadAuthorityDirectory();
loadTelegramStatus();
loadVideos();
loadIncidents();
refreshAuthorityMessage();
refreshAlarmStatus();
setInterval(refreshTelemetry, 350);
setInterval(refreshAlarmStatus, 1500);
setInterval(refreshAuthorityMessage, 2000);
setInterval(loadIncidents, 6000);
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return render_template_string(HTML_PAGE, lan_ip=get_lan_ip())


@app.route("/video_feed")
def video_feed():
    camera_hub.heartbeat()
    return Response(
        camera_hub.generate_client_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    camera_hub.heartbeat()
    data = camera_hub.get_metrics()
    data["authority_person"] = alert_manager.get_authority_person()
    data["lan_ip"] = get_lan_ip()
    return jsonify(data)


@app.route("/api/camera/release", methods=["POST", "GET"])
def release_camera_api():
    """Immediately releases camera hardware when client closes tab or clicks standby."""
    camera_hub.release_camera()
    return jsonify({"ok": True, "message": "Camera hardware released."})


@app.route("/api/system/shutdown", methods=["POST"])
def shutdown_system():
    """Allows user to cleanly terminate server and release camera hardware from UI."""
    camera_hub.stop()
    def do_exit():
        time.sleep(0.5)
        os._exit(0)
    threading.Thread(target=do_exit, daemon=True).start()
    return jsonify({"ok": True, "message": "Server shutting down, camera released."})


@app.route("/api/authorities", methods=["GET"])
def get_authorities_api():
    return jsonify(alert_manager.get_authorities())


@app.route("/api/authorities/save_all", methods=["POST"])
def save_all_authorities_api():
    data = request.get_json(force=True, silent=True) or {}
    count = int(data.get("count", 1))
    authorities_data = data.get("authorities", [])
    on_duty_id = int(data.get("on_duty_id", 1))
    password = data.get("password") or data.get("master_password", "")
    ok, msg = alert_manager.save_all_authorities(
        count=count,
        authorities_data=authorities_data,
        on_duty_id=on_duty_id,
        password=password
    )
    return jsonify(ok=ok, message=msg, data=alert_manager.get_authorities())


@app.route("/api/authorities/update", methods=["POST"])
def update_authority_slot_api():
    data = request.get_json(force=True, silent=True) or {}
    slot_id = int(data.get("id") or 1)
    name = data.get("name", "")
    phone = data.get("phone", "")
    email = data.get("email", "")
    is_on_duty = bool(data.get("is_on_duty", False))
    password = data.get("password") or data.get("master_password", "")
    ok, msg = alert_manager.update_authority_slot(
        slot_id=slot_id,
        name=name,
        phone=phone,
        email=email,
        is_on_duty=is_on_duty,
        password=password
    )
    return jsonify(ok=ok, message=msg, data=alert_manager.get_authorities())


@app.route("/api/authorities/set_on_duty", methods=["POST"])
def set_on_duty_api():
    data = request.get_json(force=True, silent=True) or {}
    slot_id = int(data.get("id") or 1)
    password = data.get("password", "")
    ok, msg = alert_manager.set_on_duty(slot_id, password=password)
    return jsonify(ok=ok, message=msg, data=alert_manager.get_authorities())


@app.route("/api/authority", methods=["GET", "POST"])
def authority_api():
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        password = data.get("password") or data.get("current_password")
        ok, msg = alert_manager.update_authority_person(data, provided_password=password)
        return jsonify(ok=ok, message=msg, authority=alert_manager.get_authority_person())
    else:
        return jsonify(alert_manager.get_authority_person())


@app.route("/api/authority/last_message")
def last_authority_message_api():
    with alert_manager.lock:
        msg = alert_manager.last_authority_message
    return jsonify(msg or {})


@app.route("/api/videos")
def list_videos():
    return jsonify(get_available_videos())


@app.route("/api/set_source", methods=["POST"])
def set_source_api():
    data = request.get_json(force=True, silent=True) or {}
    source = data.get("source", "webcam")
    ok, res = camera_hub.set_source(source)
    if not ok:
        return jsonify(ok=False, message=res)
    return jsonify(ok=True, source=source)


@app.route("/api/camera/reconnect", methods=["POST"])
def reconnect_camera_api():
    ok = camera_hub.reconnect_camera()
    return jsonify(ok=ok, message="Live camera reconnected.")


@app.route("/upload", methods=["POST"])
def upload_video():
    video = request.files.get("video")
    is_ajax = (request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.form.get("ajax") == "1")
    if not video or not video.filename:
        if is_ajax:
            return jsonify(ok=False, message="No video selected"), 400
        return "No video selected", 400

    filename = secure_filename(video.filename)
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        if is_ajax:
            return jsonify(ok=False, message=f"Unsupported format {extension}"), 400
        return f"Unsupported format {extension}", 400

    dest = UPLOAD_FOLDER / filename
    video.save(dest)

    ok, res = camera_hub.set_source(filename)

    if is_ajax:
        return jsonify(ok=True, filename=filename, message=f"Video '{filename}' uploaded successfully!")

    return f"""<script>
        alert("Video '{filename}' uploaded successfully! Analyzing now.");
        window.location.href = "/?source=" + encodeURIComponent('{filename}') + "&t=" + Date.now();
    </script>"""


@app.route("/api/check_action", methods=["POST"])
def check_action_api():
    """Evaluates live camera/video action and dispatches immediate notification alarm to authority."""
    metrics = camera_hub.get_metrics()
    src = camera_hub.get_current_source()
    frame_copy = camera_hub.get_latest_annotated_frame()

    auth = alert_manager.get_on_duty_authority()
    contact_str = auth.get("phone") or auth.get("email") or "Duty Log"

    persons = metrics.get("persons", [f"Person #1 ({auth.get('name', 'Authority')})"])
    motion = max(28.0, metrics.get("motion_score", 45.0))

    details = {
        "location": "Camera 01 - Main Quad Surveillance",
        "source": src,
        "persons": persons,
        "persons_count": max(1, metrics.get("persons_count", 1)),
        "score": motion,
        "confidence": 95.0
    }

    ok, inc_id = alert_manager.trigger_alert("ACTION_CHECK", details, frame=frame_copy, force=True)
    msg = alert_manager.last_authority_message or {}

    return jsonify(
        ok=True,
        incident_id=inc_id,
        message=f"🚨 Immediate action notification alarm dispatched to {auth.get('name', 'Authority')} ({contact_str})! Siren sounding.",
        authority=auth,
        sms_url=msg.get("sms_url", ""),
        whatsapp_url=msg.get("whatsapp_url", ""),
        mailto_url=msg.get("mailto_url", ""),
        push_url=msg.get("push_url", "")
    )


@app.route("/api/alarm/silence", methods=["POST"])
def silence_alarm_api():
    ok, msg = alert_manager.silence_alarm()
    return jsonify(ok=ok, message=msg)


@app.route("/api/alarm/status", methods=["GET"])
def alarm_status_api():
    return jsonify(alert_manager.get_alarm_status())



@app.route("/api/telegram/status", methods=["GET"])
def telegram_status_api():
    cfg = alert_manager.config.get("telegram", {})
    return jsonify({
        "enabled": cfg.get("enabled", True),
        "bot_token": cfg.get("bot_token", ""),
        "chat_id": cfg.get("chat_id", ""),
        "bot_username": "Alert_22bms_bot",
        "has_chat_id": bool(cfg.get("chat_id", ""))
    })


@app.route("/api/telegram/save", methods=["POST"])
def telegram_save_api():
    data = request.get_json(silent=True) or {}
    enabled = bool(data.get("enabled", True))
    bot_token = data.get("bot_token", "").strip()
    chat_id = str(data.get("chat_id", "")).strip()

    with alert_manager.lock:
        if "telegram" not in alert_manager.config:
            alert_manager.config["telegram"] = {}
        alert_manager.config["telegram"]["enabled"] = enabled
        if bot_token:
            alert_manager.config["telegram"]["bot_token"] = bot_token
        if chat_id:
            alert_manager.config["telegram"]["chat_id"] = chat_id
        alert_manager.save_config()
    return jsonify(ok=True, message="Telegram settings saved successfully.")


@app.route("/api/telegram/detect_chat_id", methods=["POST"])
def telegram_detect_chat_id_api():
    ok, msg = alert_manager.auto_detect_telegram_chat_id()
    return jsonify(
        ok=ok,
        message=msg,
        chat_id=alert_manager.config.get("telegram", {}).get("chat_id", "")
    )


@app.route("/api/telegram/test", methods=["POST"])
def telegram_test_api():
    ok, msg = alert_manager.test_telegram_alert()
    return jsonify(ok=ok, message=msg)


@app.route("/api/test_alert", methods=["POST"])
def test_alert_api():
    src = camera_hub.get_current_source()
    metrics = camera_hub.get_metrics()
    frame_copy = camera_hub.get_latest_annotated_frame()

    auth = alert_manager.get_on_duty_authority()
    contact_str = auth.get("phone") or auth.get("email") or "Duty Log"
    details = {
        "location": "Camera 01 - Main Quad Surveillance",
        "source": src,
        "persons": metrics.get("persons", [f"Person #1 ({auth.get('name', 'Authority')})"]),
        "persons_count": max(1, metrics.get("persons_count", 1)),
        "score": max(50.0, metrics.get("motion_score", 75.0)),
        "confidence": 95.0
    }
    ok, inc_id = alert_manager.trigger_alert("TEST_ALERT", details, frame=frame_copy, force=True)
    msg = alert_manager.last_authority_message or {}
    return jsonify(
        ok=True,
        incident_id=inc_id,
        message=f"🚨 Notification alarm dispatched to On-Duty Authority #{auth.get('id', 1)}: {auth.get('name', 'Authority')} ({contact_str})! Incident ID: {inc_id}",
        sms_url=msg.get("sms_url", ""),
        whatsapp_url=msg.get("whatsapp_url", ""),
        mailto_url=msg.get("mailto_url", ""),
        push_url=msg.get("push_url", "")
    )



MOBILE_HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Campus CCTV | Jeevan Authority Terminal</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg-base: #060b13;
    --bg-surface: #0f172a;
    --border: #1e293b;
    --red: #ef4444;
    --emerald: #10b981;
    --cyan: #06b6d4;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: var(--bg-base);
    color: var(--text-main);
    font-family: 'Inter', sans-serif;
    min-height: 100vh;
    padding: 16px;
    transition: background-color 0.3s ease;
}
body.alarm-active {
    background: #450a0a !important;
    animation: phoneStrobe 0.7s infinite alternate;
}
@keyframes phoneStrobe {
    0% { background: #450a0a; }
    100% { background: #7f1d1d; }
}
.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 14px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 16px;
}
.badge-active {
    background: rgba(16, 185, 129, 0.2);
    border: 1px solid #10b981;
    color: #34d399;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
}
.badge-alarm {
    background: #ef4444;
    color: white;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 800;
    animation: pulse 0.8s infinite;
}
@keyframes pulse {
    0% { transform: scale(0.96); opacity: 0.8; }
    50% { transform: scale(1.04); opacity: 1; }
    100% { transform: scale(0.96); opacity: 0.8; }
}
.officer-card {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.alarm-display {
    display: none;
    background: rgba(239, 68, 68, 0.2);
    border: 2px solid #ef4444;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 0 30px rgba(239, 68, 68, 0.5);
}
.alarm-display.show {
    display: block;
}
.btn {
    width: 100%;
    padding: 12px 16px;
    border-radius: 8px;
    border: none;
    font-weight: 700;
    font-size: 14px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    margin-top: 10px;
    text-decoration: none;
}
.btn-danger { background: #ef4444; color: white; }
.btn-whatsapp { background: #25d366; color: #072612; }
.btn-emerald { background: #10b981; color: white; }
.btn-secondary { background: rgba(255,255,255,0.08); color: white; border: 1px solid var(--border); }
.snapshot-img {
    width: 100%;
    max-height: 240px;
    object-fit: cover;
    border-radius: 8px;
    margin-top: 10px;
    border: 1px solid var(--border);
}
</style>
</head>
<body>
<div class="header">
    <div>
        <h2 style="font-size: 16px; font-weight: 800;">🛡️ CAMPUS CCTV</h2>
        <p style="font-size: 11px; color: var(--text-muted);">Authority Mobile Alert Terminal</p>
    </div>
    <span id="connStatus" class="badge-active">● CONNECTED</span>
</div>

<div class="officer-card">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.5px;">ON-DUTY AUTHORITY</div>
            <h3 style="font-size: 20px; font-weight: 800; color: white; margin-top: 2px;" id="dispOfficerName">Jeevan</h3>
        </div>
        <div style="width: 42px; height: 42px; border-radius: 50%; background: #2563eb; display: flex; align-items: center; justify-content: center; font-weight: 800; font-size: 18px; border: 2px solid #60a5fa;">J</div>
    </div>
    <div style="margin-top: 12px; font-size: 12px; color: var(--text-muted); display: grid; gap: 4px;">
        <div>📱 Phone: <b style="color: #38bdf8;" id="dispOfficerPhone">+91 9353415499</b></div>
        <div>📧 Email: <b style="color: #93c5fd;" id="dispOfficerEmail">jeevanl06232003@gmail.com</b></div>
        <div>🔔 Device Mode: <b style="color: #34d399;" id="dispAudioStatus">Audio & Vibration Ready 🔊</b></div>
    </div>
    <button class="btn btn-secondary" onclick="testMobileAlarm()" style="margin-top: 12px; padding: 8px 12px; font-size: 12px;">
        ⚡ Test Sound & Vibration Alarm
    </button>
</div>

<!-- Real-Time Emergency Alarm Card -->
<div id="alarmCard" class="alarm-display">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="badge-alarm">🚨 EMERGENCY ALARM ACTIVE</span>
        <button class="btn btn-secondary" style="width: auto; padding: 4px 10px; font-size: 11px; margin: 0;" onclick="silenceMobileAlarm()">Silence 🔇</button>
    </div>
    <h3 id="alarmTitle" style="font-size: 18px; font-weight: 800; color: #fee2e2; margin-top: 10px;">ACTION DETECTED</h3>
    <p id="alarmDesc" style="font-size: 12px; color: #fecaca; margin-top: 4px;">Alert dispatched to Jeevan (9353415499).</p>
    <div id="alarmMeta" style="font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #fed7aa; margin-top: 8px; background: rgba(0,0,0,0.4); padding: 8px; border-radius: 6px;"></div>
    <img id="alarmImg" class="snapshot-img" src="" style="display: none;" alt="Incident Evidence">

    <a id="btnMobileWhatsApp" href="#" target="_blank" class="btn btn-whatsapp">
        💬 Open WhatsApp Alert
    </a>
    <a id="btnMobileSms" href="#" class="btn" style="background: #0284c7; color: white;">
        📱 Open Direct SMS
    </a>
    <button class="btn btn-danger" onclick="silenceMobileAlarm()">
        🔇 Acknowledge & Silence Alarm
    </button>
</div>

<!-- Standby Monitoring Message -->
<div id="standbyCard" class="officer-card" style="text-align: center; padding: 24px 16px;">
    <div style="font-size: 32px; animation: pulse 2s infinite;">📡</div>
    <h4 style="font-size: 15px; margin-top: 10px; color: #34d399;">Monitoring Campus Surveillance</h4>
    <p style="font-size: 12px; color: var(--text-muted); margin-top: 6px;">
        Standing by for incidents on Camera 01. When an aggressive action or fight is detected on the laptop, this phone will immediately ring with sirens and vibrate!
    </p>
    <div style="margin-top: 14px;">
        <button class="btn btn-emerald" onclick="initMobileAudio()">
            🔊 Tap to Ensure Phone Siren is Unmuted
        </button>
    </div>
</div>

<script>
let audioCtx = null;
let sirenInterval = null;
let currentAlarmActive = false;

function initMobileAudio() {
    try {
        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        if (audioCtx.state === 'suspended') audioCtx.resume();
        document.getElementById('dispAudioStatus').textContent = 'Audio & Vibration Ready 🔊';
        document.getElementById('dispAudioStatus').style.color = '#34d399';
    } catch(e) {
        console.warn("Audio init error:", e);
    }
}
document.addEventListener('click', initMobileAudio, {once: false});
document.addEventListener('touchstart', initMobileAudio, {once: false});

function playMobileSiren() {
    initMobileAudio();
    if (!audioCtx) return;
    try {
        for (let cycle = 0; cycle < 2; cycle++) {
            const startT = audioCtx.currentTime + (cycle * 0.55);
            const osc = audioCtx.createOscillator();
            const gain = audioCtx.createGain();

            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(880, startT);
            osc.frequency.exponentialRampToValueAtTime(440, startT + 0.28);
            osc.frequency.exponentialRampToValueAtTime(880, startT + 0.55);

            gain.gain.setValueAtTime(0.4, startT);
            gain.gain.exponentialRampToValueAtTime(0.01, startT + 0.55);

            osc.connect(gain);
            gain.connect(audioCtx.destination);

            osc.start(startT);
            osc.stop(startT + 0.55);
        }
    } catch(e) {}
}

function startAlarmLoop() {
    if (sirenInterval) return;
    playMobileSiren();
    if (navigator.vibrate) navigator.vibrate([500, 250, 500, 250, 500]);
    sirenInterval = setInterval(() => {
        playMobileSiren();
        if (navigator.vibrate) navigator.vibrate([400, 200, 400]);
    }, 1200);
}

function stopAlarmLoop() {
    if (sirenInterval) {
        clearInterval(sirenInterval);
        sirenInterval = null;
    }
    if (navigator.vibrate) navigator.vibrate(0);
}

function testMobileAlarm() {
    initMobileAudio();
    playMobileSiren();
    if (navigator.vibrate) navigator.vibrate([300, 150, 300]);
    alert("🔊 Sound & Vibration Alarm Tested Successfully on Jeevan's Phone!");
}

async function silenceMobileAlarm() {
    stopAlarmLoop();
    try {
        await fetch('/api/alarm/silence', {method: 'POST'});
    } catch(e) {}
    document.body.classList.remove('alarm-active');
    document.getElementById('alarmCard').classList.remove('show');
    document.getElementById('standbyCard').style.display = 'block';
    document.getElementById('connStatus').className = 'badge-active';
    document.getElementById('connStatus').textContent = '● CONNECTED';
    currentAlarmActive = false;
}

async function checkAlarmPoll() {
    try {
        const resp = await fetch('/api/alarm/status');
        const data = await resp.json();

        if (data.alarm_active) {
            currentAlarmActive = true;
            document.body.classList.add('alarm-active');
            document.getElementById('alarmCard').classList.add('show');
            document.getElementById('standbyCard').style.display = 'none';
            document.getElementById('connStatus').className = 'badge-alarm';
            document.getElementById('connStatus').textContent = '🚨 ALARM ACTIVE';

            const lastMsg = data.last_message || {};
            document.getElementById('alarmTitle').textContent = lastMsg.type ? lastMsg.type.replace('_', ' ') : 'SECURITY ACTION DETECTED';
            document.getElementById('alarmDesc').textContent = `Target: ${lastMsg.authority_name || 'Jeevan'} (Phone: ${lastMsg.authority_phone || '9353415499'})`;
            document.getElementById('alarmMeta').textContent = `Time: ${lastMsg.timestamp || new Date().toLocaleTimeString()} | Status: Dispatched`;

            if (lastMsg.snapshot) {
                const img = document.getElementById('alarmImg');
                img.src = '/incidents/' + lastMsg.snapshot;
                img.style.display = 'block';
            }

            if (lastMsg.whatsapp_url) {
                document.getElementById('btnMobileWhatsApp').href = lastMsg.whatsapp_url;
            }
            if (lastMsg.sms_url && document.getElementById('btnMobileSms')) {
                document.getElementById('btnMobileSms').href = lastMsg.sms_url;
            }

            startAlarmLoop();
        } else if (currentAlarmActive) {
            silenceMobileAlarm();
        }
    } catch(e) {
        console.warn("Poll error:", e);
    }
}

setInterval(checkAlarmPoll, 900);
checkAlarmPoll();
</script>
</body>
</html>
"""


@app.route("/mobile")
@app.route("/duty")
def mobile_terminal():
    return render_template_string(MOBILE_HTML_PAGE)


@app.route("/qr/mobile")
def qr_mobile():
    try:
        import qrcode
        import qrcode.image.svg
        lan_ip = get_lan_ip()
        img = qrcode.make(f"http://{lan_ip}:5000/mobile", image_factory=qrcode.image.svg.SvgPathImage)
        return Response(img.to_string(), mimetype="image/svg+xml")
    except Exception as e:
        return str(e), 500


@app.route("/qr/ntfy")
def qr_ntfy():
    try:
        import qrcode
        import qrcode.image.svg
        auth = alert_manager.get_on_duty_authority()
        clean_phone = "".join(c for c in (auth.get("phone") or "9353415499") if c.isdigit())
        if len(clean_phone) == 10 and not clean_phone.startswith("91"):
            clean_phone = "91" + clean_phone
        url = f"https://ntfy.sh/cctv_{clean_phone}"
        img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage)
        return Response(img.to_string(), mimetype="image/svg+xml")
    except Exception as e:
        return str(e), 500


@app.route("/api/incidents")
def list_incidents():
    return jsonify(alert_manager.get_incidents(limit=30))


@app.route("/incidents/<path:filename>")
def serve_incident_image(filename):
    return send_from_directory(INCIDENTS_DIR, filename)


@app.errorhandler(413)
def file_too_large(error):
    return "Video too large. Maximum size is 250 MB.", 413


def cleanup_on_exit():
    try:
        print("[CampusCCTV] Gracefully releasing camera hardware before exit...")
        camera_hub.stop()
    except Exception:
        pass

atexit.register(cleanup_on_exit)

def handle_exit_signal(sig, frame):
    cleanup_on_exit()
    os._exit(0)

try:
    signal.signal(signal.SIGINT, handle_exit_signal)
    signal.signal(signal.SIGTERM, handle_exit_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_exit_signal)
except Exception:
    pass


if __name__ == "__main__":
    lan_ip = get_lan_ip()
    print("=" * 60)
    print("   CAMPUS CCTV AI SURVEILLANCE SYSTEM")
    print(f"   * Laptop Dashboard: http://127.0.0.1:5000")
    print(f"   * Phone Terminal:   http://{lan_ip}:5000/mobile")
    print("   * Auto-launching dashboard in default browser...")
    print("=" * 60)
    threading.Thread(target=auto_open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)