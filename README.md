# 🛡️ Campus CCTV AI Surveillance System

An intelligent, real-time AI surveillance and fight detection platform powered by Deep Learning, Computer Vision, and Automated Authority Alert Dispatch.

---

## 🌟 Key Features

* **Pose-Based Deep Learning Action Recognition:**
  * Uses **YOLOv8-Pose** for 17-keypoint skeletal tracking per student.
  * Deep Spatio-Temporal **Bi-Directional GRU with Self-Attention** (`StudentFightClassifier`) classifies sequential human movements as `NORMAL` or `FIGHTING` with confidence scores.
* **Per-Student Behavior Tracking:**
  * Displays individual status (e.g. `Student #1 [NORMAL 98%]`, `Student #2 [FIGHTING 94%]`).
  * Real-time illuminated AI skeleton overlays on video (Green = Normal, Red = Fighting).
* **Zero False Alarms for Solitary Individuals:**
  * Automatically isolates single individuals stretching, walking, or gesturing—fights require multi-person physical combat contact.
* **Smart Camera Hardware Management:**
  * Automatic hardware release (turns webcam LED OFF) when browser tabs are closed.
  * Graceful shutdown hooks prevent hardware lockouts in Windows.
* **Automated Security Notifications:**
  * Instant multi-channel alert dispatch to on-duty authorities (WhatsApp, SMS, Email, Web Siren, ntfy.sh push notifications).
  * Evidence snapshots logged automatically to `incidents/`.
* **Cross-Device Mobile Terminal:**
  * Mobile-responsive interface accessible from any phone or tablet on the local campus network via dynamic QR code.

---

## 🚀 Quick Start

### 1. Requirements
* Windows 10/11
* Python 3.10+
* Webcam or CCTV video archive

### 2. Setup & Installation
```bash
# Clone the repository
git clone https://github.com/jeevanl06232004/campus-cctv.git
cd campus-cctv

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Application
Double-click `run.bat` or run:
```bash
python main.py
```
Open **`http://127.0.0.1:5000`** in your browser.

---

## 🧠 Model Training & Fine-Tuning

To train or fine-tune the student fight classifier on your own recorded videos:
```bash
python train_fight_model.py --use-videos
```

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Ultralytics YOLOv8-Pose
* **Computer Vision:** OpenCV, NumPy
* **Backend:** Python Flask
* **Frontend:** HTML5, CSS3, JavaScript, WebAudio API

