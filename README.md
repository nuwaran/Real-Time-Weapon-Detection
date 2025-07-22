# 🎯 Real-Time Weapon Detection System using YOLOv8 and CCTV Surveillance

This project demonstrates a real-time AI-powered weapon detection system designed for public and institutional safety. The system integrates **YOLOv8** object detection with **CCTV camera feeds**, and runs on a **Raspberry Pi-based custom PCB** for efficient edge deployment.

🗓️ **Project Duration**: Feb 2025 – Apr 2025  
👨‍💻 **Type**: Team Project

---

## 🚀 Features

- 🔫 **Weapon Detection**: Detects firearms, knives, and other weapon-related objects with high accuracy in real-time.
- 🧠 **YOLOv8-based Model**: Trained on a curated dataset for weapon recognition across various environments.
- 🎥 **CCTV Integration**: Captures live video stream via USB/IP cameras for monitoring.
- 🧾 **Real-Time Alerts**: Automatically triggers alerts (email/SMS/audio) when a weapon is detected.
- 📦 **Edge Deployment**: Runs on a compact Raspberry Pi with custom PCB, optimized for low-latency inference.
- ⚡ **Optional TensorRT Acceleration** for faster inference.

---

## 🛠️ Technologies Used

| Tool/Platform      | Purpose                          |
|--------------------|----------------------------------|
| YOLOv8 (Ultralytics)| Object Detection Model           |
| Python             | Core application logic           |
| OpenCV             | Video streaming and frame ops    |
| Raspberry Pi 4     | Deployment on edge               |
| Custom PCB         | Integrates Pi + I/O + camera     |
| TensorRT (optional)| Optimized model acceleration     |
| SMTP/Twilio        | Alert mechanism                  |

---

## 🔍 System Architecture

```mermaid
graph TD
    A[User/CCTV Camera] --> B[YOLOv8 Model]
    B --> C[Weapon Detected?]
    C -- Yes --> D[Trigger Alert]
    C -- No --> E[Continue Stream]
    D --> F[Security Team / Dashboard]
