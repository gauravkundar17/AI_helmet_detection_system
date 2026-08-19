# 🛡️ AI Helmet Detection & Bike Ignition Control System

An end-to-end, real-time Computer Vision and IoT safety monitoring system powered by **YOLOv8**, **OpenCV**, and **PySerial**. The system continuously detects whether individuals (riders, industrial workers, site visitors) are wearing helmets and can interface directly with hardware microcontrollers (e.g., ESP32, Arduino) to enforce safety rules—such as triggering alarms or controlling vehicle ignitions.

---

## 📋 Table of Contents
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Directory Structure](#-directory-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [How to Run](#-how-to-run)
- [Hardware Integration (ESP32 / Arduino)](#-hardware-integration-esp32--arduino)
- [Model & Dataset Training](#-model--dataset-training)
- [Troubleshooting](#-troubleshooting)
- [License & Acknowledgments](#-license--acknowledgments)

---

## ✨ Features

- **⚡ Real-Time YOLOv8 Detection**: High-accuracy object detection trained to identify bare heads (`head`) and helmets (`helmet`) at HD resolutions (1280x720).
- **⏱️ Temporal Verification Logic**: Prevents false positive alerts or visual flicker by verifying detection state continuously over a configurable window (default: `2.0 seconds`).
- **🟢 Dynamic HUD Status Overlay**:
  - `SAFE` (Green): Helmet detected and verified for $\ge 2$ seconds.
  - `NO HELMET` (Red): Unprotected head detected for $\ge 2$ seconds.
  - `CHECKING...` (Yellow): State transition in progress.
  - `NO PERSON` (White): No target detected in frame.
- **🔌 Hardware / Microcontroller Interface**: Integrated Serial communication via `pyserial` to send commands (`SAFE` / `NO HELMET`) to ESP32 or Arduino microcontrollers.
- **📁 Modular & Trainable**: Easily retrain or fine-tune models with updated datasets using Ultralytics YOLOv8.

---

## 🏗️ System Architecture

```
+------------------+      +--------------------------+      +------------------------------+
|   Webcam Input   | ---> | YOLOv8 Neural Network    | ---> |  Temporal State Controller   |
| (HD 1280x720 Stream)|   | (helmet vs. bare head)   |      | (2.0 sec stability filter)   |
+------------------+      +--------------------------+      +------------------------------+
                                                                            |
                                                                            v
+------------------+      +--------------------------+      +------------------------------+
| Hardware Control | <--- |  Serial Communication    | <--- | Real-Time OpenCV Visual HUD |
| (ESP32/Arduino)  |      |  (COM Port @ 115200)     |      | (Status Bounding Boxes)      |
+------------------+      +--------------------------+      +------------------------------+
```

---

## 📁 Directory Structure

```directory
AI_helmet_detection_system/
├── dataset/                    # Training and validation dataset files
│   └── helmet_detection/
│       ├── data.yaml           # Dataset configuration & class definitions
│       ├── train/              # Training images and YOLO annotations
│       ├── valid/              # Validation set
│       └── test/               # Test set
├── docs/                       # Project documentation & reference assets
├── esp/                        # ESP32 / Arduino microcontroller firmware source code
├── model/                      # Custom model storage directory
├── output/                     # Exported outputs, captures, and logs
├── runs/                       # Training logs and fine-tuned YOLOv8 weights
│   └── detect/
│       └── train-3/
│           └── weights/        # Trained model checkpoints (best.pt, last.pt)
├── helmet_detection.py         # Main application script for real-time inference
├── requirements.txt            # Python dependencies
├── yolov8n.pt                  # Pre-trained YOLOv8 nano base model
└── README.md                   # Project documentation
```

---

## 🔧 Prerequisites

- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python**: Version `3.8` to `3.11` recommended.
- **Hardware**:
  - Webcam / USB Camera (Standard 720p or 1080p).
  - *(Optional)* Microcontroller (ESP32 / Arduino Uno/Nano) for hardware actions.
- **NVIDIA GPU** *(Optional for higher FPS)*: CUDA-enabled GPU for faster deep learning inference.

---

## ⚙️ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gauravkundar17/AI_helmet_detection_system.git
   cd AI_helmet_detection_system
   ```

2. **Set Up Virtual Environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎛️ Configuration

You can customize parameters inside `helmet_detection.py`:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `CONFIDENCE_THRESHOLD` | `0.80` (80%) | Minimum confidence required to accept a detection. |
| `DETECTION_TIME` | `2.0` (seconds) | Time required for consistent detection before switching status. |
| `model` | `"runs/detect/train-3/weights/best.pt"` | Path to the trained YOLOv8 model weights. |
| `cap = cv2.VideoCapture(0)` | `0` | Camera device index (`0` for built-in webcam, `1` for external USB camera). |
| `CONF_PROP_FRAME_WIDTH/HEIGHT` | `1280x720` | Resolution for the webcam capture stream. |

---

## 🚀 How to Run

1. Make sure your webcam is plugged in and accessible.
2. Run the main Python detection script:
   ```bash
   python helmet_detection.py
   ```
3. A live OpenCV window titled **"AI Helmet Detection"** will open displaying:
   - Green bounding boxes around detected **helmets**.
   - Red bounding boxes around bare **heads**.
   - Live status indicator text (**SAFE**, **NO HELMET**, **CHECKING...**, **NO PERSON**).
4. **To Exit**: Press the `q` key on your keyboard while the video window is focused.

---

## 🔌 Hardware Integration (ESP32 / Arduino)

To control real-world devices (such as turning on a relay, sounding a buzzer, or enabling a bike ignition when `SAFE` is declared):

1. **Connect Hardware**: Plug your ESP32 or Arduino via USB. Note the assigned COM port (e.g., `COM3` on Windows or `/dev/ttyUSB0` on Linux).
2. **Flash Firmware**: Upload your microcontroller code (located in the `esp/` directory).
3. **Enable Serial Port in Python**:
   Uncomment lines 3, 10-11, 162-172, and 182 in `helmet_detection.py`:
   ```python
   import serial

   # Initialize Serial port (Update "COM3" to your port)
   signal = serial.Serial("COM3", 115200)
   time.sleep(2)

   # Inside detection loop:
   if status != previous_status:
       if status == "SAFE":
           signal.write(b"SAFE\n")
       elif status == "NO HELMET":
           signal.write(b"NO HELMET\n")
   ```

---

## 📊 Model & Dataset Training

The model is trained on a safety helmet detection dataset containing class targets such as `helmet` and `head`.

### Retraining the Model
If you want to re-train YOLOv8 on custom data:
1. Ensure your dataset is formatted in YOLO format inside `dataset/helmet_detection/`.
2. Run the Ultralytics training command:
   ```bash
   yolo detect train data=dataset/helmet_detection/data.yaml model=yolov8n.pt epochs=50 imgsz=640
   ```
3. Updated weights will be saved under `runs/detect/train/weights/best.pt`. Update the path in `helmet_detection.py` to use your new model.

---

## ❓ Troubleshooting

- **Webcam cannot be opened**:
  - Ensure no other application (Zoom, Teams, Camera app) is using the webcam.
  - Change `cv2.VideoCapture(0)` to `1` or `2` if using an external USB camera.
- **PySerial `SerialException`**:
  - Verify that the COM port matching your microcontroller is correct in `helmet_detection.py`.
  - Ensure the serial monitor in Arduino IDE is closed before running the Python script.
- **Low FPS / Lag**:
  - Ensure PyTorch with CUDA support is installed if using an NVIDIA GPU.
  - Lower the camera resolution in `helmet_detection.py` (e.g., `640x480`).

---

## 📜 License & Acknowledgments

- **Dataset**: Safety Helmet Dataset hosted on [Roboflow Universe](https://universe.roboflow.com/object-detection-tn0y3/safety-helmet-rtki3/dataset/3) under CC BY 4.0 License.
- **YOLOv8 Architecture**: Developed by [Ultralytics](https://github.com/ultralytics/ultralytics).
- **OpenCV & PySerial**: Open-source libraries powering real-time vision and hardware telemetry.
