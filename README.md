# Edge AI ANPR: Autonomous Garage Access System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLO](https://img.shields.io/badge/Model-YOLO%20v11n-orange)
![OpenCV](https://img.shields.io/badge/Library-OpenCV-green)
![Hardware](https://img.shields.io/badge/Hardware-Raspberry%20Pi%205-red)
![Next.js](https://img.shields.io/badge/Dashboard-Next.js%2014-black)

An end-to-end Edge AI computer vision pipeline designed to completely automate garage door access. The system detects vehicles in real-time, extracts license plate numbers using OCR, verifies them against a whitelisted database, and triggers the hardware mechanism—all monitored via a modern web dashboard.


## Features

- **Real-Time Object Detection**: Utilizes a lightweight, custom-trained **YOLO v11n** model optimized for edge devices to detect license plates with minimal latency.
- **Optical Character Recognition (OCR)**: Integrates robust OCR techniques alongside **OpenCV** image preprocessing to accurately read alphanumeric characters from cropped plate frames.
- **Edge Deployment**: Fully optimized to run locally on a **Raspberry Pi 5**, eliminating the need for constant cloud processing and ensuring privacy.
- **Automated Access Control**: Cross-references extracted plates with an **SQLite** database and triggers GPIO pins to open the garage door for whitelisted vehicles.
- **Full-Stack Web Dashboard**: Includes a **Next.js 14** administrative panel to monitor access logs, add/remove whitelisted plates, and view system health.

## System Architecture

The project bridges machine learning, embedded hardware, and web development:

1. **Camera Feed**: Captures live video stream at the garage entrance.
2. **Inference (YOLO + OpenCV)**: Identifies the region of interest (license plate) and preprocesses the image (grayscale, thresholding).
3. **Extraction (OCR)**: Reads the string from the processed image.
4. **Validation (SQLite)**: Checks if the string exists in the `whitelist` table.
5. **Hardware Trigger**: If valid, a signal is sent via Raspberry Pi GPIO to the door relay.
6. **Logging (Next.js)**: Event (timestamp, plate string, success/fail) is logged and pushed to the web dashboard.

## Tech Stack

- **Computer Vision & AI**: Python, OpenCV, YOLO (v11n), PyTorch/Keras, EasyOCR / Tesseract
- **Hardware**: Raspberry Pi 5, Camera Module, Relay Switch
- **Database**: SQLite
- **Web Frontend**: Next.js 14, React, Tailwind CSS

## Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js & npm (for the Next.js dashboard)
- Raspberry Pi 5 (if deploying to hardware)

### 1. Clone the repository
```bash
git clone https://github.com/Kaludev/ANPR.git
cd ANPR
```
