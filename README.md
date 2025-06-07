# Capsule Defect Detection System

A real-time quality control system that detects defective capsules using computer vision and TensorFlow Lite, deployed via Streamlit.

## Features

- **Dual Input Modes**:
  - Real-time webcam inspection
  - Image upload functionality
- **Accurate Detection**:
  - Binary classification (Good/Defective)
  - Confidence percentage display
- **User-Friendly Interface**:
  - Visual feedback with bounding boxes
  - Adjustable confidence threshold
- **Optimized Performance**:
  - TensorFlow Lite for fast inference
  - Cached model loading

## Prerequisites

- Python 3.9
- TensorFlow 2.10
- NumPy 1.24.4
- Streamlit
- OpenCV

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/capsule-defect-detection.git
   cd capsule-defect-detection
2. Create and activate environment from environment.yml
   ```bash
   conda env create -f environment.yml
   conda activate capsule-defect-env

## Usage

1. Place your TensorFlow Lite model (model.tflite) in the project root
2. Run the application:
```bash
streamlit run app.py
```
3.Access the web interface at 
```bash
http://localhost:8501
```
## 🧪 Dataset Used
# MvTec AD - Capsule
Includes variations like:
Contamination
Cracks
Pokes
Scratches
Squeezes
* Only Good and Defective categories used for training *

## ✍️ Author
Pranjul Shukla
B.E. Student at GEC Modasa

LinkedIn: https://www.linkedin.com/in/pranjul-shukla-877231282?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app | GitHub : https://github.com/Pranjul9947/Capsule_Detector
 
   
