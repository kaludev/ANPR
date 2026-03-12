# 🚘 Edge AI ANPR: Autonomous Garage Access System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/Model-YOLO%20v11n-orange)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-green)
![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red)
![Status](https://img.shields.io/badge/Status-Active%20Development-success)

An end-to-end Edge AI computer vision pipeline designed to automate garage door access. The system detects vehicles in real-time, extracts license plate numbers using OCR, verifies them against a local SQLite database, and triggers a hardware mechanism. 

*Note: The project is currently active, with hardware optimizations and a full-stack web dashboard in the development roadmap.*

<p align="center">
  <!-- OBAVEZNO: Zameni ovaj link ispod sa pravim GIF-om ili slikom tvog sistema u akciji -->
  <img src="UBACI_LINK_SLIKE_OVDE_ILI_GIF" alt="ANPR System Demo" width="700"/>
</p>

## ✨ Current Features

- **Real-Time Object Detection**: Utilizes a lightweight, custom-trained **YOLO v11n** model to detect license plates.
- **Optical Character Recognition (OCR)**: Integrates robust OCR techniques alongside **OpenCV** image preprocessing to accurately read alphanumeric characters from cropped plate frames.
- **Edge Deployment**: Runs locally on a **Raspberry Pi 5**, eliminating the need for cloud processing and ensuring privacy.
- **Automated Access Control**: Cross-references extracted plates with an **SQLite** database and triggers GPIO pins to open the garage door for whitelisted vehicles.

## 🏗️ System Architecture (Core)

1. **Camera Feed**: Captures live video stream at the garage entrance.
2. **Inference (YOLO + OpenCV)**: Identifies the region of interest (license plate) and preprocesses the image (grayscale, thresholding).
3. **Extraction (OCR)**: Reads the string from the processed image.
4. **Validation (SQLite)**: Checks if the string exists in the local `whitelist` table.
5. **Hardware Trigger**: If valid, a signal is sent via Raspberry Pi GPIO to the door relay.

## ⚠️ Current Limitations & Bottlenecks

- **Hardware Inference FPS**: It has been observed that the Raspberry Pi 5 currently struggles to maintain the desired high FPS when running the full pipeline (simultaneous YOLO object detection + OCR extraction). The CPU/NPU bottleneck causes frame drops during peak processing. *Mitigation strategies (e.g., model quantization, TensorRT, or NCNN) are being researched.*

## 🗺️ Roadmap & Planned Features

- [ ] **Next.js Web Dashboard**: Develop a full-stack Next.js 14 administrative panel to remotely monitor access logs, add/remove whitelisted plates, and view system health. *(Currently in planning phase)*
- [ ] **Inference Optimization**: Export the YOLO model to ONNX/TensorRT or use a Google Coral Edge TPU to bypass the current Raspberry Pi FPS bottleneck.
- [ ] **Tracking Algorithm**: Implement a deep-sort tracking algorithm to prevent redundant OCR readings on the same stationary vehicle.
- [ ] **Database Migration**: Migrate SQLite to PostgreSQL for better concurrency handling if scaling to multiple garage doors.

## 🛠️ Tech Stack (Current)

- **Computer Vision & AI**: Python, OpenCV, YOLO (v11n), PyTorch/Keras, EasyOCR / Tesseract
- **Hardware**: Raspberry Pi 5, Camera Module, Relay Switch
- **Database**: SQLite

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Raspberry Pi 5 (if deploying to hardware)

### 1. Clone the repository
```bash
git clone https://github.com/Kaludev/ANPR.git
cd ANPR
```

### 2. Setup the Environment
```bash
cd cv-pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Edge AI Pipeline
*Ensure your camera is connected and the weights file (`yolov11n_custom.pt`) is placed in the `models/` directory.*
```bash
python main.py --source 0
```

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/Kaludev/ANPR/issues).

## 👨‍💻 Author
**Luka Marković**
- LinkedIn: [@kaludev](https://linkedin.com/in/kaludev)
- GitHub: [@Kaludev](https://github.com/Kaludev)
