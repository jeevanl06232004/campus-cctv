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


def generate_synthetic_training_data(num_samples=4000, seq_len=16):
    """
    Generates realistic biomechanical sequences with hard negatives:
      - Class 0 (NORMAL):
        * Normal walking, sitting, idle (alone)
        * Two people standing very close talking (distance 0.2 - 0.7) with hand gestures
        * Group walking side-by-side or passing each other
        * Stretching, waving, pointing, adjusting clothes
      - Class 1 (FIGHTING):
        * Rapid punching bursts (wrist velocity 2.5 - 7.0)
        * Aggressive pushing / shoving with forward lunge
        * Grappling / close-quarters wrestling (distance < 0.45 + high kinetic energy + torso tilt)
        * Repeated strikes and guard defense
    """
    np.random.seed(42)
    torch.manual_seed(42)

    half = num_samples // 2
    x = np.zeros((num_samples, seq_len, 78), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)

    # 1. Normal Conversation between 2 people close together (HARD NEGATIVE) - 40%
    n_conv = int(half * 0.40)
    for i in range(n_conv):
        x[i, :, 0:34] = np.random.randn(seq_len, 34) * 0.08
        x[i, :, 34:68] = np.random.randn(seq_len, 34) * 0.05
        gestures = np.random.uniform(0.1, 0.9, (seq_len, 2))
        x[i, :, 68:70] = gestures
        x[i, :, 70] = np.mean(gestures, axis=1) * 0.8
        x[i, :, 71] = np.random.uniform(0.0, 0.2, seq_len)
        x[i, :, 72] = np.random.uniform(0.0, 0.25, seq_len)
        x[i, :, 73] = np.random.uniform(0.1, 0.8, seq_len)
        x[i, :, 74] = np.random.uniform(0.3, 0.7, seq_len)
        x[i, :, 75] = np.random.uniform(0.0, 0.35, seq_len)
        x[i, :, 76] = np.random.uniform(0.25, 0.85, seq_len)
        x[i, :, 77] = x[i, :, 70] / (x[i, :, 76] + 0.3)
        y[i] = 0

    # 2. Single person walking / idle / alone - 35%
    n_walk = int(half * 0.35)
    for i in range(n_conv, n_conv + n_walk):
        x[i, :, 0:34] = np.random.randn(seq_len, 34) * 0.10
        x[i, :, 34:68] = np.random.randn(seq_len, 34) * 0.06
        t = np.linspace(0, 2 * np.pi, seq_len)
        swing = np.abs(np.sin(t + np.random.rand() * np.pi)) * 0.35
        x[i, :, 68] = swing + np.random.rand(seq_len) * 0.1
        x[i, :, 69] = swing + np.random.rand(seq_len) * 0.1
        x[i, :, 70] = swing * 0.8
        x[i, :, 71] = swing * 0.9
        x[i, :, 72] = np.random.uniform(0.0, 0.2, seq_len)
        x[i, :, 73] = np.random.uniform(0.1, 0.6, seq_len)
        x[i, :, 74] = np.random.uniform(0.3, 0.6, seq_len)
        x[i, :, 75] = np.random.uniform(0.2, 0.7, seq_len)
        x[i, :, 76] = np.random.uniform(1.2, 2.5, seq_len)
        x[i, :, 77] = x[i, :, 70] / (x[i, :, 76] + 0.3)
        y[i] = 0

    # 3. Dynamic non-combat (gesturing, waving, stretching) - 25%
    for i in range(n_conv + n_walk, half):
        x[i, :, 0:34] = np.random.randn(seq_len, 34) * 0.12
        x[i, :, 34:68] = np.random.randn(seq_len, 34) * 0.08
        x[i, :, 68] = np.random.uniform(0.2, 1.4, seq_len)
        x[i, :, 69] = np.random.uniform(0.2, 1.4, seq_len)
        x[i, :, 70] = np.random.uniform(0.3, 1.2, seq_len)
        x[i, :, 71] = np.random.uniform(0.0, 0.4, seq_len)
        x[i, :, 72] = np.random.uniform(0.2, 0.5, seq_len)
        x[i, :, 73] = np.random.uniform(0.2, 1.0, seq_len)
        x[i, :, 74] = np.random.uniform(0.5, 1.1, seq_len)
        x[i, :, 75] = np.random.uniform(0.0, 0.5, seq_len)
        x[i, :, 76] = np.random.uniform(0.6, 2.0, seq_len)
        x[i, :, 77] = x[i, :, 70] / (x[i, :, 76] + 0.3)
        y[i] = 0

    # 4. Fights: Punches / Strikes - 50%
    n_punch = int(half * 0.50)
    for i in range(half, half + n_punch):
        x[i, :, 0:34] = np.random.randn(seq_len, 34) * 0.25
        x[i, :, 34:68] = np.random.randn(seq_len, 34) * 0.20
        burst_start = np.random.randint(2, seq_len - 5)
        burst_len = np.random.randint(4, seq_len - burst_start + 1)
        x[i, :, 68:70] = np.random.uniform(0.5, 1.5, (seq_len, 2))
        x[i, burst_start:burst_start+burst_len, 68] = np.random.uniform(3.0, 7.5, burst_len)
        x[i, burst_start:burst_start+burst_len, 69] = np.random.uniform(3.0, 7.5, burst_len)
        x[i, :, 70] = np.mean(x[i, :, 68:70], axis=1) * 1.2
        x[i, :, 71] = np.random.uniform(0.4, 2.0, seq_len)
        x[i, :, 72] = np.random.uniform(0.65, 1.0, seq_len)
        x[i, :, 73] = np.random.uniform(1.2, 3.2, seq_len)
        x[i, :, 74] = np.random.uniform(0.6, 1.4, seq_len)
        x[i, :, 75] = np.random.uniform(1.0, 3.0, seq_len)
        x[i, :, 76] = np.random.uniform(0.15, 0.70, seq_len)
        x[i, :, 77] = (x[i, :, 70] * 2.0) / (x[i, :, 76] + 0.1)
        y[i] = 1

    # 5. Fights: Shoves / Lunges - 30%
    n_shove = int(half * 0.30)
    for i in range(half + n_punch, half + n_punch + n_shove):
        x[i, :, 0:34] = np.random.randn(seq_len, 34) * 0.28
        x[i, :, 34:68] = np.random.randn(seq_len, 34) * 0.22
        x[i, :, 68:70] = np.random.uniform(2.5, 5.5, (seq_len, 2))
        x[i, :, 70] = np.random.uniform(2.8, 5.5, seq_len)
        x[i, :, 71] = np.random.uniform(1.0, 3.2, seq_len)
        x[i, :, 72] = np.random.uniform(0.5, 0.9, seq_len)
        x[i, :, 73] = np.random.uniform(1.8, 4.0, seq_len)
        x[i, :, 74] = np.random.uniform(0.7, 1.3, seq_len)
        x[i, :, 75] = np.random.uniform(2.0, 4.5, seq_len)
        x[i, :, 76] = np.random.uniform(0.10, 0.55, seq_len)
        x[i, :, 77] = (x[i, :, 70] * 2.5) / (x[i, :, 76] + 0.1)
        y[i] = 1

    # 6. Fights: Grappling / Wrestling - 20%
    for i in range(half + n_punch + n_shove, num_samples):
        x[i, :, 0:34] = np.random.randn(seq_len, 34) * 0.30
        x[i, :, 34:68] = np.random.randn(seq_len, 34) * 0.25
        x[i, :, 68:70] = np.random.uniform(2.0, 4.8, (seq_len, 2))
        x[i, :, 70] = np.random.uniform(2.2, 4.5, seq_len)
        x[i, :, 71] = np.random.uniform(1.5, 3.8, seq_len)
        x[i, :, 72] = np.random.uniform(0.4, 0.8, seq_len)
        x[i, :, 73] = np.random.uniform(2.0, 4.5, seq_len)
        x[i, :, 74] = np.random.uniform(0.4, 1.0, seq_len)
        x[i, :, 75] = np.random.uniform(1.0, 2.5, seq_len)
        x[i, :, 76] = np.random.uniform(0.05, 0.40, seq_len)
        x[i, :, 77] = (x[i, :, 70] * 2.2) / (x[i, :, 76] + 0.1)
        y[i] = 1

    p = np.random.permutation(num_samples)
    return x[p], y[p]


def train_model(epochs=30, batch_size=64, lr=0.002):
    device = torch.device("cpu")
    print("=== Training Student Fight vs Normal Deep Learning Model ===")
    x, y = generate_synthetic_training_data(num_samples=4000)

    n_train = 3200
    x_train, y_train = torch.tensor(x[:n_train]), torch.tensor(y[:n_train])
    x_val, y_val = torch.tensor(x[n_train:]), torch.tensor(y[n_train:])

    model = StudentFightClassifier(input_dim=78, hidden_dim=64, num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        total_loss = 0.0

        for b in range(0, n_train, batch_size):
            b_idx = perm[b:b+batch_size]
            bx, by = x_train[b_idx].to(device), y_train[b_idx].to(device)

            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_logits, _ = model(x_val.to(device))
                val_preds = torch.argmax(val_logits, dim=-1)
                val_acc = (val_preds == y_val.to(device)).float().mean().item() * 100.0
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {total_loss/n_train:.4f} | Val Accuracy: {val_acc:.1f}%")

    torch.save(model.state_dict(), str(MODEL_WEIGHTS_PATH))
    print(f"Model saved to {MODEL_WEIGHTS_PATH} successfully!")


if __name__ == "__main__":
    train_model()
