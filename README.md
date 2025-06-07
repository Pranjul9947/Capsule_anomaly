# Capsule Anomaly Detection System

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
2. Create and activate conda environment 
   ```bash
   conda create -n Capsule_defect python=3.9.21
   conda activate Capsule_defect
3. Install Dependencies using requirements.txt
   ```bash
   pip install -r requirements.txt

## Usage

1. Place your TensorFlow Lite model (model_unquant.tflite) in the project root
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

LinkedIn: www.linkedin.com/in/pranjul-shukla-877231282 | GitHub : https://github.com/Pranjul9947/Capsule_anomaly
 
   
