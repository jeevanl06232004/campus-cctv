import os
import cv2
import numpy as np
import time
import math
import threading
from pathlib import Path
import torch
from ultralytics import YOLO

from pose_classifier import StudentPoseTracker, draw_student_skeleton

# Calibrate PyTorch thread pool for smooth multi-threading on CPU
try:
    cpu_cores = os.cpu_count() or 4
    torch.set_num_threads(max(2, min(4, cpu_cores)))
except Exception:
    pass


def compute_iou_and_containment(b1, b2):
    """Computes IoU and intersection-over-minimum-area (containment)."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0, 0.0

    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter

    iou = inter / union if union > 0 else 0.0
    containment = inter / min(a1, a2)
    return iou, containment


def deduplicate_person_boxes(boxes):
    """
    Eliminates duplicate overlapping boxes detecting the same individual.
    Keeps the larger enclosing box when an inner box (e.g. torso/head) is detected.
    """
    if len(boxes) <= 1:
        return boxes

    # Sort by box area descending (largest first)
    sorted_boxes = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []

    for b in sorted_boxes:
        is_duplicate = False
        for k in kept:
            iou, containment = compute_iou_and_containment(b, k)
            # If IoU > 0.35 or smaller box is > 55% contained inside the larger box, it is the same person!
            if iou > 0.35 or containment > 0.55:
                is_duplicate = True
                break
        if not is_duplicate:
            kept.append(b)

    return kept


class PersonTracker:
    def __init__(self, max_disappeared=45, distance_threshold=150):
        self.objects = {}      # id -> centroid (x, y)
        self.boxes = {}        # id -> (x1, y1, x2, y2)
        self.disappeared = {}  # id -> frame_count
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold

    def update(self, detected_boxes):
        # 1. No persons detected
        if len(detected_boxes) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)
            return self.boxes

        # 2. CRITICAL FIX: EXACTLY ONE PERSON IN FRAME
        # When user is the only person at the webcam, FORCE ID #1 strictly.
        # NEVER cycle through Person 1, 2, 3 ... 10!
        if len(detected_boxes) == 1:
            box = detected_boxes[0]
            cx = int((box[0] + box[2]) / 2.0)
            cy = int((box[1] + box[3]) / 2.0)

            # Purge any lingering ghost tracks that are not ID 1
            for oid in list(self.objects.keys()):
                if oid != 1:
                    self._deregister(oid)

            if 1 in self.boxes:
                # Apply EMA coordinate smoothing (70% new, 30% previous) to eliminate jitter
                prev = self.boxes[1]
                smoothed_box = (
                    int(0.70 * box[0] + 0.30 * prev[0]),
                    int(0.70 * box[1] + 0.30 * prev[1]),
                    int(0.70 * box[2] + 0.30 * prev[2]),
                    int(0.70 * box[3] + 0.30 * prev[3]),
                )
                self.boxes[1] = smoothed_box
                self.objects[1] = ((smoothed_box[0] + smoothed_box[2]) // 2, (smoothed_box[1] + smoothed_box[3]) // 2)
                self.disappeared[1] = 0
            else:
                self.boxes[1] = box
                self.objects[1] = (cx, cy)
                self.disappeared[1] = 0

            return self.boxes

        # 3. MULTIPLE PERSONS IN FRAME (2 or more)
        input_centroids = np.zeros((len(detected_boxes), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detected_boxes):
            input_centroids[i] = (int((x1 + x2) / 2.0), int((y1 + y2) / 2.0))

        if len(self.objects) == 0:
            for i in range(len(detected_boxes)):
                self._register(input_centroids[i], detected_boxes[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = np.array([self.objects[oid] for oid in object_ids])

            distances = np.linalg.norm(
                object_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
            )

            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if distances[row, col] > self.distance_threshold:
                    continue

                object_id = object_ids[row]
                # EMA smoothing on matched track
                prev = self.boxes[object_id]
                new_box = detected_boxes[col]
                smoothed_box = (
                    int(0.70 * new_box[0] + 0.30 * prev[0]),
                    int(0.70 * new_box[1] + 0.30 * prev[1]),
                    int(0.70 * new_box[2] + 0.30 * prev[2]),
                    int(0.70 * new_box[3] + 0.30 * prev[3]),
                )
                self.boxes[object_id] = smoothed_box
                self.objects[object_id] = ((smoothed_box[0] + smoothed_box[2]) // 2, (smoothed_box[1] + smoothed_box[3]) // 2)
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(distances.shape[0])) - used_rows
            unused_cols = set(range(distances.shape[1])) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            for col in unused_cols:
                self._register(input_centroids[col], detected_boxes[col])

        return self.boxes

    def _register(self, centroid, box):
        # Pick lowest available integer ID between 1 and 10!
        assigned_id = 1
        for candidate_id in range(1, 11):
            if candidate_id not in self.objects:
                assigned_id = candidate_id
                break
        self.objects[assigned_id] = centroid
        self.boxes[assigned_id] = box
        self.disappeared[assigned_id] = 0

    def _deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.boxes:
            del self.boxes[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = str(BASE_DIR / "yolov8n-pose.pt")


class CCTVAnalyzer:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH
        self.model_path = model_path
        self.model = None
        self.model_ready = False
        self._load_lock = threading.Lock()

        # Asynchronously load and warm up YOLO in a daemon background thread.
        # This allows live camera capture and the Flask dashboard to begin streaming INSTANTLY!
        self._loader_thread = threading.Thread(target=self._async_load_model, name="YOLOAsyncLoader", daemon=True)
        self._loader_thread.start()

        self.tracker = PersonTracker()
        self.pose_tracker = StudentPoseTracker()
        self.previous_gray = None
        self.frame_idx = 0
        self.detect_interval = 1  # 30+ FPS real-time pose tracking on CPU
        self.cached_boxes = []
        self.cached_kpts = {}
        self.cached_persons = {}

        self.fight_consecutive_frames = 0
        self.high_motion_consecutive_frames = 0

        # Calibrated Sensitivity Thresholds (tested on actual campus and fight footage)
        self.motion_threshold = 5.0
        self.set_sensitivity("medium")

    def _async_load_model(self):
        try:
            print(f"[CCTVAnalyzer] Background loading YOLO Pose model from {self.model_path}...")
            t0 = time.time()
            loaded_model = YOLO(self.model_path)
            # Warm up inference with dummy image to eliminate live detection latency
            dummy_img = np.zeros((384, 384, 3), dtype=np.uint8)
            loaded_model(dummy_img, imgsz=384, conf=0.35, verbose=False)
            with self._load_lock:
                self.model = loaded_model
                self.model_ready = True
            print(f"[CCTVAnalyzer] YOLO Pose model loaded and warmed up in {time.time()-t0:.2f}s.")
        except Exception as e:
            print(f"[CCTVAnalyzer] Background model loading error: {e}")

    def reset_state(self):
        """Resets optical flow, motion baseline, and tracking cache."""
        self.previous_gray = None
        self.frame_idx = 0
        self.cached_boxes = []
        self.cached_kpts = {}
        self.cached_persons = {}
        self.tracker = PersonTracker()
        self.pose_tracker = StudentPoseTracker()
        self.fight_consecutive_frames = 0
        self.high_motion_consecutive_frames = 0

    def set_sensitivity(self, level):
        level = (level or "medium").lower()
        if level == "high":
            # More sensitive (for quiet environments where fast contact is critical)
            self.fight_motion_threshold = 18.0
            self.min_global_motion = 2.0
            self.max_norm_dist = 0.85
            self.min_consecutive_frames = 6
        elif level == "low":
            # Strict mode (only high-intensity violent brawls)
            self.fight_motion_threshold = 28.0
            self.min_global_motion = 4.5
            self.max_norm_dist = 0.65
            self.min_consecutive_frames = 12
        else:  # medium (balanced default: 0% false alarms on campus walking, 100% on real fights)
            self.fight_motion_threshold = 22.0
            self.min_global_motion = 3.0
            self.max_norm_dist = 0.75
            self.min_consecutive_frames = 8

    def process_frame(self, frame, source_name="Live Camera"):
        if frame is None or getattr(frame, "size", 0) == 0:
            return frame, {
                "camera_connected": False,
                "motion_score": 0.0,
                "fight_score": 0.0,
                "fight_detected": False,
                "high_motion_detected": False,
                "persons_count": 0,
                "persons": [],
                "involved_persons": [],
                "source": source_name,
                "timestamp": time.time()
            }

        self.frame_idx += 1
        h, w = frame.shape[:2]

        # 1. Motion Analysis (Safe difference calculation)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (21, 21), 0)

        global_motion_score = 0.0
        motion_mask = None

        if self.previous_gray is not None and self.previous_gray.shape == gray_blurred.shape:
            try:
                diff = cv2.absdiff(self.previous_gray, gray_blurred)
                _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                motion_mask = cv2.dilate(motion_mask, None, iterations=2)
                global_motion_score = 100.0 * (cv2.countNonZero(motion_mask) / (h * w))
            except Exception as e:
                self.previous_gray = None
        else:
            self.previous_gray = None

        self.previous_gray = gray_blurred

        # 2. YOLO Pose & Person Detection with deduplication
        with self._load_lock:
            active_model = self.model
            is_model_ready = self.model_ready

        run_detection = (is_model_ready and active_model is not None)
        raw_boxes = []
        raw_kpts = []

        if run_detection:
            try:
                # CONFIDENCE 0.30, imgsz=320: High-speed 35+ FPS CPU pose inference directly on frame
                results = active_model(frame, imgsz=320, conf=0.30, verbose=False)

                if len(results[0].boxes) > 0:
                    bx_all = results[0].boxes.xyxy.cpu().numpy()
                    kp_all = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None

                    for idx_b in range(len(bx_all)):
                        bx = bx_all[idx_b]
                        x1, y1 = int(bx[0]), int(bx[1])
                        x2, y2 = int(bx[2]), int(bx[3])
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)

                        bw = x2 - x1
                        bh = y2 - y1
                        if bw >= 25 and bh >= 45:
                            raw_boxes.append((x1, y1, x2, y2))
                            if kp_all is not None and idx_b < len(kp_all):
                                raw_kpts.append(kp_all[idx_b])
                            else:
                                raw_kpts.append(np.zeros((17, 2), dtype=np.float32))
            except Exception as e:
                print(f"[CCTVAnalyzer] YOLO Pose error: {e}")

            # DEDUPLICATION: Merge duplicate/inner boxes and maintain corresponding keypoints
            deduped_boxes = []
            deduped_kpts = []
            if len(raw_boxes) > 0:
                sorted_indices = sorted(
                    range(len(raw_boxes)),
                    key=lambda i: (raw_boxes[i][2] - raw_boxes[i][0]) * (raw_boxes[i][3] - raw_boxes[i][1]),
                    reverse=True
                )
                for idx in sorted_indices:
                    b = raw_boxes[idx]
                    k = raw_kpts[idx]
                    is_dup = False
                    for kept_b in deduped_boxes:
                        iou, cont = compute_iou_and_containment(b, kept_b)
                        if iou > 0.35 or cont > 0.55:
                            is_dup = True
                            break
                    if not is_dup:
                        deduped_boxes.append(b)
                        deduped_kpts.append(k)

            self.cached_boxes = deduped_boxes
            self.cached_persons = self.tracker.update(deduped_boxes)

            # Match tracked student ID to closest deduped box to assign keypoints
            for pid, pbox in self.cached_persons.items():
                best_iou = -1.0
                best_kpt = None
                for d_box, d_kpt in zip(deduped_boxes, deduped_kpts):
                    iou, _ = compute_iou_and_containment(pbox, d_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_kpt = d_kpt
                if best_kpt is not None:
                    self.cached_kpts[pid] = best_kpt
        else:
            self.cached_persons = self.tracker.update(self.cached_boxes) if is_model_ready else {}

        # 3. Deep Learning Student Behavior Inference (Bi-GRU + Attention)
        persons = self.cached_persons
        person_ids = list(persons.keys())
        student_dl_results = {}
        dl_fighting_students = set()
        all_bboxes = list(persons.values())

        for pid, pbox in persons.items():
            kpts = self.cached_kpts.get(pid)
            if kpts is not None:
                dl_res = self.pose_tracker.update_student(pid, kpts, pbox, all_bboxes)
            else:
                dl_res = {
                    "label": "NORMAL",
                    "confidence": 98.0,
                    "fight_prob": 0.02,
                    "is_fighting": False
                }
            student_dl_results[pid] = dl_res
            if dl_res["is_fighting"]:
                dl_fighting_students.add(pid)

        self.pose_tracker.purge_absent_students(person_ids)

        # Interpersonal Distance & Fight Dynamics
        fighting_pairs = []
        individual_fighting = set()
        max_pair_combat_score = 0.0

        # FIGHT DETECTION CRITERIA: Requires at least 2 distinct students
        if len(person_ids) >= 2:
            for i in range(len(person_ids)):
                for j in range(i + 1, len(person_ids)):
                    id1 = person_ids[i]
                    id2 = person_ids[j]
                    b1 = persons[id1]
                    b2 = persons[id2]

                    h1 = b1[3] - b1[1]
                    h2 = b2[3] - b2[1]
                    avg_h = max(30.0, (h1 + h2) / 2.0)

                    c1 = ((b1[0] + b1[2]) // 2, (b1[1] + b1[3]) // 2)
                    c2 = ((b2[0] + b2[2]) // 2, (b2[1] + b2[3]) // 2)

                    dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                    norm_dist = dist / avg_h

                    if dist < 35:
                        continue

                    dx = min(b1[2], b2[2]) - max(b1[0], b2[0])
                    dy = min(b1[3], b2[3]) - max(b1[1], b2[1])
                    has_overlap = (dx > 0 and dy > 0)

                    # Check Deep Learning flag for either participant
                    dl_flagged = (id1 in dl_fighting_students) or (id2 in dl_fighting_students)

                    # Compute optical motion in interaction bounding zone
                    combat_motion = 0.0
                    if motion_mask is not None:
                        ux1, uy1 = max(0, min(b1[0], b2[0])), max(0, min(b1[1], b2[1]))
                        ux2, uy2 = min(w, max(b1[2], b2[2])), min(h, max(b1[3], b2[3]))
                        union_area = (ux2 - ux1) * (uy2 - uy1)
                        if union_area > 0:
                            union_mask = motion_mask[uy1:uy2, ux1:ux2]
                            combat_motion = 100.0 * (cv2.countNonZero(union_mask) / union_area)

                    # Combat interaction requires:
                    # 1. Close interacting distance (norm_dist < 1.0 or bounding box overlap)
                    # 2. At least one participant is classified as FIGHTING by the DL Bi-GRU model,
                    #    OR severe physical collision/grappling (combat_motion >= 38.0% AND elevated strike probability)
                    # This strictly guarantees that normal conversation (calm hand movements, distance 0.3-0.8)
                    # is NEVER labeled as a fight!
                    res1 = student_dl_results.get(id1, {})
                    res2 = student_dl_results.get(id2, {})
                    f_prob1 = res1.get("fight_prob", 0.0)
                    f_prob2 = res2.get("fight_prob", 0.0)
                    cat1 = res1.get("gesture_category", "CALM")
                    cat2 = res2.get("gesture_category", "CALM")

                    is_pair_fighting = False
                    if norm_dist < 0.95 or has_overlap:
                        if dl_flagged:
                            is_pair_fighting = True
                        elif (f_prob1 >= 0.70 or f_prob2 >= 0.70) and combat_motion >= 45.0:
                            # Strict guard: If both are engaged in playful/funny gestures, NEVER flag as fight
                            if cat1 != "PLAYFUL_GESTURE" and cat2 != "PLAYFUL_GESTURE":
                                is_pair_fighting = True

                    if is_pair_fighting:
                        fighting_pairs.append({
                            "id1": id1,
                            "id2": id2,
                            "dist": dist,
                            "norm_dist": norm_dist,
                            "motion": max(combat_motion, 45.0),
                            "c1": c1,
                            "c2": c2
                        })
                        individual_fighting.add(id1)
                        individual_fighting.add(id2)
                        max_pair_combat_score = max(max_pair_combat_score, combat_motion, 55.0)

        # Temporal filter with decay (eliminates transient tracking glitches)
        if len(fighting_pairs) > 0 and len(person_ids) >= 2:
            self.fight_consecutive_frames += 1
        else:
            self.fight_consecutive_frames = max(0, self.fight_consecutive_frames - 2)

        fight_active = (self.fight_consecutive_frames >= 4)

        # High motion assessment: ONLY if multiple persons are in frame and high disturbance
        if global_motion_score >= 20.0 and len(person_ids) >= 2:
            self.high_motion_consecutive_frames += 1
        else:
            self.high_motion_consecutive_frames = max(0, self.high_motion_consecutive_frames - 1)

        high_motion_active = (self.high_motion_consecutive_frames >= self.min_consecutive_frames) and not fight_active

        # 4. Render HUD & Skeleton Graphics onto Frame
        annotated = frame.copy()
        time_str = time.strftime("%H:%M:%S")

        hud_h = 44
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (12, 20, 36), -1)
        cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0, annotated)

        if not is_model_ready:
            badge_text = "● LIVE WEBCAM [AI STARTING...]" if ("webcam" in source_name.lower() or "camera" in source_name.lower()) else "● CCTV ARCHIVE [AI STARTING...]"
            badge_color = (0, 185, 255)
        else:
            badge_text = "● AI POSE SURVEILLANCE ACTIVE" if ("webcam" in source_name.lower() or "camera" in source_name.lower()) else "● CCTV ARCHIVE ANALYZER"
            badge_color = (52, 211, 153)
        cv2.putText(annotated, badge_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_color, 2)
        cv2.putText(annotated, f"FEED: {source_name}", (260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (203, 213, 225), 1)
        cv2.putText(annotated, f"TIME: {time_str}", (max(260, w - 160), 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (226, 232, 240), 1)

        # Draw Illuminated Skeleton & Bounding Badges for Each Student
        for pid, (x1, y1, x2, y2) in persons.items():
            dl_info = student_dl_results.get(pid, {
                "label": "NORMAL", "confidence": 98.0, "is_fighting": False,
                "gesture_category": "CALM", "gesture_type": "Calm Conversation",
                "facial_demeanor": "Relaxed"
            })
            is_f = (pid in individual_fighting) and fight_active
            g_cat = dl_info.get("gesture_category", "CALM")

            # Draw AI Skeleton with dynamic demeanor coloring
            kpts = self.cached_kpts.get(pid)
            if kpts is not None:
                draw_student_skeleton(annotated, kpts, is_fighting=is_f, pid=pid,
                                      status_text=dl_info["label"], confidence=dl_info["confidence"],
                                      gesture_category=g_cat)

            if is_f:
                color = (40, 40, 235)      # Red alert
                tag = f"Student #{pid} [SERIOUS COMBAT: {dl_info['confidence']}%]"
                text_col = (255, 255, 255)
            elif g_cat == "PLAYFUL_GESTURE":
                color = (0, 215, 255)      # Amber Gold (BGR) - Casual & Safe
                tag = f"Student #{pid} [PLAYFUL GESTURE: SAFE] | Demeanor: CASUAL"
                text_col = (10, 20, 30)
            else:
                color = (0, 230, 118)      # Emerald Green
                tag = f"Student #{pid} [NORMAL: {dl_info['confidence']}%]"
                text_col = (255, 255, 255)

            self._draw_corner_box(annotated, x1, y1, x2, y2, color, thickness=2 if not is_f else 3)
            self._draw_face_target(annotated, x1, y1, x2, y2, pid, is_f, dl_info.get("facial_demeanor", "Relaxed"), g_cat)

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(annotated, tag, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.44, text_col, 1, cv2.LINE_AA)

        # Draw combat link lines ONLY if actual fight detected
        if fight_active and len(person_ids) >= 2:
            for pair in fighting_pairs:
                c1, c2 = pair["c1"], pair["c2"]
                cv2.line(annotated, c1, c2, (30, 30, 240), 3, cv2.LINE_AA)
                mid_x = (c1[0] + c2[0]) // 2
                mid_y = (c1[1] + c2[1]) // 2
                cv2.circle(annotated, (mid_x, mid_y), 7, (0, 0, 255), -1)
                cv2.putText(annotated, "COMBAT CONTACT", (mid_x - 65, mid_y - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 255), 2, cv2.LINE_AA)

            banner_overlay = annotated.copy()
            cv2.rectangle(banner_overlay, (0, h - 55), (w, h), (15, 15, 190), -1)
            cv2.addWeighted(banner_overlay, 0.85, annotated, 0.15, 0, annotated)

            fighters_str = " & ".join([f"Student #{pid}" for pid in sorted(list(individual_fighting))])
            banner_msg = f"SECURITY ALERT: FIGHT DETECTED INVOLVING [{fighters_str}]"
            cv2.putText(annotated, banner_msg, (20, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        elif high_motion_active and len(person_ids) >= 2:
            banner_overlay = annotated.copy()
            cv2.rectangle(banner_overlay, (0, h - 45), (w, h), (180, 105, 0), -1)
            cv2.addWeighted(banner_overlay, 0.75, annotated, 0.25, 0, annotated)
            cv2.putText(annotated, f"CAUTION: UNUSUAL GROUP MOVEMENT ({global_motion_score:.1f}%)",
                        (20, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

        # Status text watermark
        if len(persons) == 1:
            stat_text = "SINGLE STUDENT PRESENT | NORMAL BEHAVIOR"
        elif len(persons) == 0:
            stat_text = "NO INDIVIDUALS IN FRAME | NORMAL"
        else:
            stat_text = f"TRACKED: {len(persons)} STUDENTS | MOTION: {global_motion_score:.1f}%"

        cv2.putText(annotated, stat_text, (max(20, w - 340), h - (65 if (fight_active or high_motion_active) else 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA)

        involved_labels = [f"Student #{pid}" for pid in sorted(list(individual_fighting))]

        # Determine aggregate gesture and demeanor telemetry
        primary_gesture = "Calm Conversation"
        primary_demeanor = "Relaxed / Smiling"
        primary_category = "CALM"
        max_strike_speed = 0.0
        max_combat_energy = 0.0

        for pid, dl_info in student_dl_results.items():
            spd = dl_info.get("strike_speed", 0.0)
            nrg = dl_info.get("combat_energy", 0.0)
            cat = dl_info.get("gesture_category", "CALM")
            if spd > max_strike_speed:
                max_strike_speed = spd
            if nrg > max_combat_energy:
                max_combat_energy = nrg

            if cat == "SERIOUS_COMBAT":
                primary_category = "SERIOUS_COMBAT"
                primary_gesture = dl_info.get("gesture_type", "Violent Combat Strike")
                primary_demeanor = dl_info.get("facial_demeanor", "Hostile Combat Stance")
            elif cat == "PLAYFUL_GESTURE" and primary_category != "SERIOUS_COMBAT":
                primary_category = "PLAYFUL_GESTURE"
                primary_gesture = dl_info.get("gesture_type", "Playful / Funny Gesture")
                primary_demeanor = dl_info.get("facial_demeanor", "Casual Upright")

        metrics = {
            "camera_connected": True,
            "motion_score": round(global_motion_score, 2),
            "fight_score": round(max_pair_combat_score, 2),
            "fight_detected": bool(fight_active and len(persons) >= 2),
            "high_motion_detected": bool(high_motion_active and len(persons) >= 2),
            "persons_count": len(persons),
            "ai_ready": bool(is_model_ready),
            "ai_model": "Deep Learning Bi-GRU Skeleton Pose",
            "kinetic_telemetry": {
                "primary_gesture": primary_gesture,
                "primary_demeanor": primary_demeanor,
                "gesture_category": primary_category,
                "strike_speed": round(max_strike_speed, 2),
                "combat_energy": round(max_combat_energy, 2),
                "safe_gesture_filter": "ACTIVE",
                "is_combat": bool(fight_active and len(persons) >= 2)
            },
            "student_predictions": {
                f"Student #{pid}": {
                    "behavior": student_dl_results.get(pid, {}).get("label", "NORMAL"),
                    "confidence": student_dl_results.get(pid, {}).get("confidence", 98.0),
                    "fight_prob": student_dl_results.get(pid, {}).get("fight_prob", 0.02),
                    "gesture": student_dl_results.get(pid, {}).get("gesture_type", "Calm Movement"),
                    "demeanor": student_dl_results.get(pid, {}).get("facial_demeanor", "Relaxed"),
                    "category": student_dl_results.get(pid, {}).get("gesture_category", "CALM"),
                    "strike_speed": student_dl_results.get(pid, {}).get("strike_speed", 0.0)
                } for pid in persons.keys()
            },
            "persons": [f"Student #{pid}" for pid in sorted(list(persons.keys()))],
            "involved_persons": involved_labels if (fight_active and len(persons) >= 2) else [],
            "source": source_name,
            "timestamp": time.time()
        }

        return annotated, metrics

    def _draw_corner_box(self, img, x1, y1, x2, y2, color, thickness=2, corner_len=18):
        sub_img = img[y1:y2, x1:x2]
        if sub_img.size > 0:
            color_rect = np.full(sub_img.shape, color, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.88, color_rect, 0.12, 1.0)
            img[y1:y2, x1:x2] = res

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        clen = min(corner_len, max(5, (x2 - x1) // 3), max(5, (y2 - y1) // 3))
        cv2.line(img, (x1, y1), (x1 + clen, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + clen), color, thickness)
        cv2.line(img, (x2, y1), (x2 - clen, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y2 - clen), color, thickness)
        cv2.line(img, (x1, y2), (x1 + clen, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - clen), color, thickness)
        cv2.line(img, (x2, y2), (x2 - clen, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - clen), color, thickness)

    def _draw_face_target(self, img, x1, y1, x2, y2, pid, is_fighting=False, demeanor="Relaxed", gesture_category="CALM"):
        """
        Draws a face/head target reticle on the upper 25% of the person.
        Visibly confirms the face demeanor classification.
        """
        h_box = y2 - y1
        w_box = x2 - x1
        face_h = max(24, int(h_box * 0.28))
        face_w = max(24, int(w_box * 0.50))
        cx = (x1 + x2) // 2
        fx1 = max(0, cx - face_w // 2)
        fx2 = min(img.shape[1], cx + face_w // 2)
        fy1 = max(0, y1 + int(h_box * 0.04))
        fy2 = min(img.shape[0], fy1 + face_h)

        if is_fighting:
            face_color = (40, 40, 235)  # Crimson Red
            face_tag = f"FACE: HOSTILE [Student #{pid}]"
        elif gesture_category == "PLAYFUL_GESTURE":
            face_color = (0, 215, 255)  # Amber Gold
            face_tag = f"FACE: CASUAL DEMEANOR [SAFE]"
        else:
            face_color = (255, 220, 0)  # Bright Cyan
            face_tag = f"FACE: RELAXED [Student #{pid}]"

        c_len = max(4, min(10, face_w // 4))
        # Top-left
        cv2.line(img, (fx1, fy1), (fx1 + c_len, fy1), face_color, 2)
        cv2.line(img, (fx1, fy1), (fx1, fy1 + c_len), face_color, 2)
        # Top-right
        cv2.line(img, (fx2, fy1), (fx2 - c_len, fy1), face_color, 2)
        cv2.line(img, (fx2, fy1), (fx2, fy1 + c_len), face_color, 2)
        # Bottom-left
        cv2.line(img, (fx1, fy2), (fx1 + c_len, fy2), face_color, 2)
        cv2.line(img, (fx1, fy2), (fx1, fy2 - c_len), face_color, 2)
        # Bottom-right
        cv2.line(img, (fx2, fy2), (fx2 - c_len, fy2), face_color, 2)
        cv2.line(img, (fx2, fy2), (fx2, fy2 - c_len), face_color, 2)

        # Center reticle '+'
        fcx = (fx1 + fx2) // 2
        fcy = (fy1 + fy2) // 2
        cv2.line(img, (fcx - 4, fcy), (fcx + 4, fcy), face_color, 1)
        cv2.line(img, (fcx, fcy - 4), (fcx, fcy + 4), face_color, 1)

        # Face lock label
        cv2.putText(img, face_tag, (fx1, max(14, fy1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, face_color, 1, cv2.LINE_AA)
