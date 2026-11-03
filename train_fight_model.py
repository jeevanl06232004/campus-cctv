"""
Campus CCTV AI Surveillance System
Model Training & Fine-Tuning Pipeline: Student Fight vs Normal Deep Learning Model
"""

import os
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

from pose_classifier import (
    StudentFightClassifier,
    extract_frame_pose_features,
    MODEL_WEIGHTS_PATH
)

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"


class PoseSequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def extract_sequences_from_video(video_path, pose_model, seq_len=16, label=0, sample_stride=2):
    """
    Extracts sliding 16-frame 78-dim feature sequences from a video clip using YOLOv8-pose.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open {video_path}")
        return [], []

    sequences = []
    labels = []
    frame_features = []
    prev_kpts = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % sample_stride != 0:
            continue

        h, w = frame.shape[:2]
        results = pose_model(frame, imgsz=384, conf=0.35, verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0 and results[0].keypoints is not None:
            # Take the most prominent person or iterate over all
            kpts_all = results[0].keypoints.xy.cpu().numpy()
            boxes_all = boxes.xyxy.cpu().numpy()
            for i in range(len(boxes_all)):
                b = boxes_all[i]
                k = kpts_all[i]
                other_boxes = [boxes_all[j] for j in range(len(boxes_all)) if j != i]
                feat = extract_frame_pose_features(k, prev_kpts, b, other_boxes)
                prev_kpts = k.copy()
                frame_features.append(feat)

                if len(frame_features) >= seq_len:
                    sequences.append(frame_features[-seq_len:])
                    labels.append(label)
                    # Keep rolling
                    if len(frame_features) > seq_len * 2:
                        frame_features = frame_features[-seq_len:]

    cap.release()
    return sequences, labels


def generate_synthetic_training_data(num_samples=1200, seq_len=16):
    """
    Generates realistic biomechanical sequences of:
      Class 0 (NORMAL): walking, standing, typing, conversational arm movements
      Class 1 (FIGHTING): sudden punches, lunges, defensive guard, rapid joint accelerations
    """
    print(f"[Training] Synthesizing {num_samples} kinetic sequence vectors...")
    half = num_samples // 2

    # Normal Class
    x_normal = np.random.randn(half, seq_len, 78).astype(np.float32) * 0.12
    # Rhythmic walking arm swing
    t = np.linspace(0, 2 * np.pi, seq_len)
    for i in range(half):
        swing = np.sin(t + np.random.rand() * np.pi) * 0.20
        x_normal[i, :, 68] = np.abs(swing) + np.random.rand() * 0.1 # left wrist speed
        x_normal[i, :, 69] = np.abs(-swing) + np.random.rand() * 0.1 # right wrist speed
        x_normal[i, :, 76] = np.random.rand() * 1.5 + 1.2           # student separation distance
        x_normal[i, :, 77] = x_normal[i, :, 68] / (x_normal[i, :, 76] + 0.3)
    y_normal = np.zeros(half, dtype=np.int64)

    # Fighting Class
    x_fight = np.random.randn(half, seq_len, 78).astype(np.float32) * 0.25
    for i in range(half):
        burst_start = np.random.randint(2, seq_len - 4)
        burst_duration = np.random.randint(3, seq_len - burst_start)
        # Violent wrist acceleration & punch strikes
        x_fight[i, burst_start:burst_start+burst_duration, 68] = np.random.uniform(2.5, 6.0, burst_duration)
        x_fight[i, burst_start:burst_start+burst_duration, 69] = np.random.uniform(2.5, 6.0, burst_duration)
        x_fight[i, :, 72] = np.random.uniform(0.7, 1.0, seq_len)   # guard stance
        x_fight[i, :, 75] = np.random.uniform(1.2, 3.5, seq_len)   # lunging speed
        x_fight[i, :, 76] = np.random.uniform(0.15, 0.65, seq_len) # close contact (<0.7)
        x_fight[i, :, 77] = (x_fight[i, :, 68] + x_fight[i, :, 69]) / (x_fight[i, :, 76] + 0.1)
    y_fight = np.ones(half, dtype=np.int64)

    x = np.concatenate([x_normal, x_fight], axis=0)
    y = np.concatenate([y_normal, y_fight], axis=0)

    # Shuffle
    idx = np.random.permutation(len(y))
    return x[idx], y[idx]


def train_model(epochs=35, batch_size=32, lr=0.002, use_videos=False):
    device = torch.device("cpu")
    print(f"=== Deep Learning Training: Student Fight vs Normal Classifier ===")
    print(f"Device: {device} | Target Epochs: {epochs} | Batch Size: {batch_size}")

    if use_videos and DATASET_DIR.exists():
        print(f"Searching for video dataset in {DATASET_DIR}...")
        pose_model = YOLO("yolov8n-pose.pt")
        all_seqs = []
        all_labels = []

        fight_dir = DATASET_DIR / "fights"
        normal_dir = DATASET_DIR / "normal"

        if fight_dir.exists():
            for vid in fight_dir.glob("*.mp4"):
                print(f"Processing fight video: {vid.name}")
                s, l = extract_sequences_from_video(vid, pose_model, label=1)
                all_seqs.extend(s)
                all_labels.extend(l)

        if normal_dir.exists():
            for vid in normal_dir.glob("*.mp4"):
                print(f"Processing normal video: {vid.name}")
                s, l = extract_sequences_from_video(vid, pose_model, label=0)
                all_seqs.extend(s)
                all_labels.extend(l)

        if len(all_seqs) > 100:
            x_data = np.array(all_seqs, dtype=np.float32)
            y_data = np.array(all_labels, dtype=np.int64)
        else:
            print("Video dataset has insufficient samples, combining with synthetic data.")
            x_synth, y_synth = generate_synthetic_training_data(1200)
            x_data = x_synth
            y_data = y_synth
    else:
        x_data, y_data = generate_synthetic_training_data(1600)

    # Train / Val Split (80 / 20)
    split = int(0.80 * len(x_data))
    train_x, val_x = x_data[:split], x_data[split:]
    train_y, val_y = y_data[:split], y_data[split:]

    train_loader = DataLoader(PoseSequenceDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(PoseSequenceDataset(val_x, val_y), batch_size=batch_size, shuffle=False)

    model = StudentFightClassifier(input_dim=78, hidden_dim=64, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _ = model(seqs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += seqs.size(0)

        scheduler.step()
        train_acc = (correct / total) * 100.0
        avg_loss = train_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                logits, _ = model(seqs)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += seqs.size(0)

        val_acc = (val_correct / val_total) * 100.0

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(MODEL_WEIGHTS_PATH))

    print("=" * 60)
    print(f"Training Complete! Peak Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Weights saved to: {MODEL_WEIGHTS_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    use_vids = "--use-videos" in sys.argv
    train_model(epochs=30, batch_size=32, use_videos=use_vids)

