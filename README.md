# 👁️ Eye Controlled Mouse

<div align="center">

### AI-Powered Cursor Control Using Eye Tracking & Facial Landmarks

Control your computer mouse using only eye movements and blink gestures. Built with Computer Vision, MediaPipe Face Landmarker, OpenCV, and Python to provide an accessible, hands-free Human-Computer Interaction (HCI) solution.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Landmarker-orange)
![AI](https://img.shields.io/badge/AI-Eye%20Tracking-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

</div>

---

# 📌 Table of Contents

* Project Overview
* Problem Statement
* Key Features
* System Workflow
* Architecture Diagram
* Technology Stack
* Face Landmarker Model
* Technical Components
* Eye Tracking Pipeline
* Project Structure
* Installation
* Usage
* Applications
* Advantages
* Limitations
* Future Enhancements
* Research Domains
* Author

---

# 🎯 Project Overview

Eye Controlled Mouse is a Computer Vision-based Human Computer Interaction system that allows users to control the mouse cursor using eye movements and facial gestures.

The application processes live webcam input, detects facial landmarks using MediaPipe Face Landmarker, tracks eye movements, and translates them into mouse cursor actions on the operating system.

The project demonstrates the practical implementation of Artificial Intelligence, Accessibility Computing, and Real-Time Vision Systems.

---

# ❓ Problem Statement

Traditional computer systems depend heavily on physical devices such as:

* Mouse
* Keyboard
* Touchpad

Individuals with motor impairments, physical disabilities, injuries, or temporary movement limitations may find these devices difficult to use.

This project solves that challenge by creating a contactless computer interaction system where users can control the cursor through eye movement and blinking.

---

# 🚀 Key Features

| Feature                 | Description                             |
| ----------------------- | --------------------------------------- |
| Eye Tracking            | Tracks eye movement in real-time        |
| Cursor Navigation       | Moves cursor based on gaze direction    |
| Blink Detection         | Performs mouse click actions            |
| Face Landmark Detection | Detects detailed facial mesh points     |
| Real-Time Processing    | Instant response with webcam feed       |
| Hands-Free Interaction  | No physical mouse required              |
| Accessibility Support   | Assists users with physical limitations |
| AI-Based Tracking       | Uses MediaPipe Face Landmarker          |
| Smooth Cursor Control   | Reduces cursor jitter                   |
| Low Hardware Cost       | Works with standard webcam              |

---

# 🔄 System Workflow

```text
User Face
    │
    ▼
Webcam Capture
    │
    ▼
OpenCV Frame Processing
    │
    ▼
MediaPipe Face Landmarker
    │
    ▼
Face Landmark Detection
    │
    ▼
Eye Landmark Extraction
    │
    ▼
Eye Movement Analysis
    │
    ▼
Cursor Coordinate Mapping
    │
    ▼
Blink Detection
    │
    ▼
PyAutoGUI Controller
    │
    ▼
Operating System Cursor
```

---

# 🏗️ Architecture Diagram

```text
┌──────────────────────┐
│      Webcam Feed      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  OpenCV Processing   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Face Landmarker AI   │
│ (face_landmarker)    │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Facial Landmarks     │
│ Detection Module     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Eye Tracking Engine  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Cursor Mapping Unit  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ PyAutoGUI Controller │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Mouse Movement &     │
│ Click Operations     │
└──────────────────────┘
```

---

# 🧠 Technology Stack

## Programming Language

* Python

## Computer Vision

* OpenCV

## AI / Machine Learning

* MediaPipe Tasks Vision

## Facial Tracking Model

* face_landmarker.task

## Mouse Automation

* PyAutoGUI

## Numerical Processing

* NumPy

## GUI Components

* Tkinter (Optional)

---

# 🤖 Face Landmarker Model

## Model Used

```text
face_landmarker.task
```

The project uses MediaPipe's Face Landmarker model to detect and track facial landmarks in real time.

### Capabilities

* 468+ facial landmarks
* Eye landmark localization
* Face mesh estimation
* Head orientation tracking
* Real-time facial analysis
* High-speed processing

---

# 🔬 Why Face Landmarker?

| Traditional Methods   | Face Landmarker           |
| --------------------- | ------------------------- |
| Limited landmarks     | 468+ landmarks            |
| Less accurate         | Highly accurate           |
| Poor tracking         | Robust tracking           |
| Sensitive to lighting | More reliable             |
| Basic eye detection   | Advanced eye localization |

---

# ⚙️ Technical Components

## 1. Webcam Acquisition Module

Responsible for:

* Capturing live video stream
* Frame acquisition
* Resolution handling
* Camera initialization

---

## 2. Frame Processing Module

Responsible for:

* Image preprocessing
* RGB conversion
* Frame optimization
* Data preparation

---

## 3. Face Detection Module

Responsible for:

* Face localization
* Face tracking
* Multi-frame consistency

---

## 4. Landmark Detection Module

Responsible for:

* Running face_landmarker.task
* Extracting facial landmarks
* Generating facial mesh points

---

## 5. Eye Tracking Module

Responsible for:

* Eye region extraction
* Eye center estimation
* Eye movement analysis
* Gaze tracking

---

## 6. Cursor Mapping Module

Responsible for:

* Coordinate transformation
* Screen scaling
* Cursor positioning
* Motion smoothing

---

## 7. Blink Detection Module

Responsible for:

* Blink recognition
* Click gesture detection
* False-click prevention

---

## 8. Mouse Controller Module

Responsible for:

* Cursor movement
* Left-click execution
* Operating system interaction

---

# 👁️ Eye Tracking Pipeline

```text
Face Detection
       │
       ▼
Eye Landmark Detection
       │
       ▼
Eye Coordinate Extraction
       │
       ▼
Eye Direction Analysis
       │
       ▼
Screen Mapping
       │
       ▼
Cursor Movement
       │
       ▼
Blink Detection
       │
       ▼
Mouse Click Event
```

---

# 📂 Project Structure

```text
Eye-Controlled-Mouse/
│
├── assets/
│   └── face_landmarker.task
│
├── src/
│   ├── eye_tracker.py
│   ├── mouse_controller.py
│   ├── landmark_detector.py
│   ├── click_detector.py
│   └── utils.py
│
├── main.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# ⚡ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/eye-controlled-mouse.git
cd eye-controlled-mouse
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python main.py
```

---

# ▶️ Usage

1. Connect webcam.
2. Run application.
3. Position face within camera view.
4. Allow system to detect landmarks.
5. Move eyes to control cursor.
6. Blink to perform click actions.

---

# 🌍 Applications

* Assistive Technology
* Accessibility Systems
* Smart Workstations
* Medical Rehabilitation
* Human Computer Interaction Research
* Contactless Interfaces
* AI-Based Accessibility Solutions
* Educational Demonstrations
* Computer Vision Learning

---

# ✅ Advantages

* Hands-free interaction
* Real-time performance
* Affordable implementation
* AI-powered tracking
* Easy deployment
* Accessible computing solution
* No specialized hardware required

---

# ⚠️ Limitations

* Sensitive to poor lighting
* Webcam quality affects accuracy
* Eye fatigue during prolonged use
* Calibration may be required
* Performance varies with head movement

---

# 🔮 Future Enhancements

* AI Gaze Estimation Models
* Multi-Monitor Support
* Right Click Detection
* Scroll Control
* Voice Command Integration
* User Calibration Profiles
* Adaptive Cursor Speed
* Mobile Device Compatibility
* Deep Learning-Based Eye Tracking

---

# 📚 Research Domains

* Artificial Intelligence
* Computer Vision
* Machine Learning
* Human Computer Interaction (HCI)
* Accessibility Computing
* Assistive Technology
* Eye Tracking Systems
* Real-Time Vision Applications

---

# 👨‍💻 Author

**Hassan**

BS Artificial Intelligence

Computer Vision & AI Developer

University Project – Eye Controlled Mouse
