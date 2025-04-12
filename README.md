---

# 🚗 Autonomous Driving System – Lane Detection, Object Detection & Steering Angle Prediction

## Source Code
[Click to Download from Google Drive](https://drive.google.com/file/d/1fckA5S0n_o5Tcm0i06gt9qZ0x5RFZKfD/view?usp=sharing)

---
A robust deep learning-based autonomous driving pipeline that simulates real-time road intelligence using advanced CNN and YOLO architectures. This system integrates:

- ✅ End-to-end **Steering Angle Prediction** (inspired by Nvidia's DAVE-2)
- ✅ **Lane Detection** using YOLOv11 semantic segmentation
- ✅ **Object Detection** using YOLOv11 Nano model with overlays

---

## 🎥 Demo Video

🎬 Watch a full demonstration of the system in action:
[Click to View on Google Drive](https://drive.google.com/file/d/1OWemP06_OfIpM84J_g1aTyuagzEDAfEt/view?usp=drive_link)

---

## 📌 Description

This project simulates an autonomous driving system with key components:

- **Real-time perception:** Lane and object detection overlays
- **Control prediction:** Predict steering angles directly from road images
- **Flexible modules:** Run components independently or together
- **Deep learning backbone:** Leverages CNN and YOLOv11 architectures

Inspired by Nvidia's DAVE-2 system ("End to End Learning for Self-Driving Cars"), this project replicates steering prediction from raw camera feeds while enhancing perception using segmentation and object detection.

## 🧠 Model Architectures

### 📀 Steering Angle Prediction (Nvidia CNN – DAVE-2 Inspired)
- **Input:** 66x200 RGB images
- **Architecture:**
  - Normalization layer
  - 5 convolutional layers (ReLU, strided, 5x5 & 3x3 kernels)
  - 3 fully connected layers (1164, 100, 50, 10)
  - **Output:** Predicted steering angle (regression)
- **Reference:** [RSNVIDIA3.pdf](./RSNVIDIA3.pdf)

### 🚣️ Lane Detection
- **Model:** YOLOv11
- **Trained Weights:** `best_yolo11_lane_segmentation.pt`
- **Function:** Semantic segmentation of lane markings

### 🧱 Object Detection
- **Model:** YOLOv11 Nano
- **Weights:** `yolo11s-seg.pt`
- **Function:** Detect vehicles, obstacles, and pedestrians in real-time

---

## 📁 Project Structure

```bash
.
├── data/
│   └── driving_dataset/                # Raw frames + steering angles
│   └── steering_wheel_image.jpg        # Steering wheel base image
├── saved_models/
│   ├── regression_model/               # Trained CNN model
│   ├── lane_segmentation_model/        # YOLOv11 segmentation model
│   └── object_detection_model/         # YOLOv11 Nano object detection model
├── source/
│   ├── inference/                      # Inference modules
│   └── utils/                          # (Optional) Utility functions
├── tests/                              # Unit tests (TBD)
├── training_lane_detection.ipynb       # YOLOv11 training notebook
├── driving_data.py                     # Data loader / processor
├── model.py                            # CNN architecture (Nvidia-style)
├── train.py                            # Train steering angle predictor
├── run_fsd_inference.py                # Full simulation runner
├── run_segmentation_obj_det.py         # Run lane + object segmentation
├── run_steering_angle_prediction.py    # Steering-only inference
├── requirements.txt                    # Dependencies
├── setup.py                            # Python project setup
└── README.md
```

---

## ⚙️ Installation & Setup

### 🔀 Clone the repository
```bash
git clone https://github.com/your-username/autonomous-driving.git
cd autonomous-driving
```

### 📦 Install Python dependencies
```bash
pip install -r requirements.txt
```

> Requires Python **3.8+**, TensorFlow **1.x**, and Ultralytics YOLO:
```bash
pip install ultralytics
```

### 🔧 Setup the project structure
Automatically creates folder structure:
```bash
python setup.py install
```

---

## 🚀 How to Use

### 🚗 Run Full Pipeline – Steering + Segmentation + Object Detection
```bash
python run_fsd_inference.py
```

- Live predictions with overlays
- OpenCV display of:
  - Raw camera view
  - Lane & object segmentation
  - Steering wheel animation

### 🎮 Run Only Steering Prediction
```bash
python run_steering_angle_prediction.py
```

### 🚣️ Visualize Lane and Object Segmentation
```bash
python run_segmentation_obj_det.py
```

> You can tweak frame display duration using the `display_time` variable.

---

## 🏋️️ Training Instructions

### 🧠 Train Steering Angle Model (Nvidia CNN)
```bash
python train.py
```
- Epochs: 30
- Batch Size: 100
- Optimizer: Adam
- Loss: MSE + L2 regularization

### 🛠 Train Lane Segmentation Model
Open and run the notebook:
```
training_lane_detection.ipynb
```

---

## 🗃️ Dataset Format

### 📷 Images
Saved in: `data/driving_dataset/`

### 📁 Steering Angle Labels
Text format: `data.txt`
```
frame0001.jpg  14.5
frame0002.jpg  0.0
```
- Input shape: Resized to **200x66**
- Normalized to feed CNN

---

## 🧱 Tech Stack
- Python 3.8+
- TensorFlow 1.x (for Nvidia-style steering model)
- YOLOv11 (for segmentation and object detection)
- OpenCV (visualization)
- Matplotlib / Numpy (training + plotting)

---

## 🔮 Roadmap
- [ ] Real-time camera-based inference
- [ ] Docker support
- [ ] ONNX / TFLite model export
- [ ] Optimize runtime GPU support
- [ ] Add simulated driving video recorder

---

## 🙌 Acknowledgements
- NVIDIA: [DAVE-2 End-to-End Self Driving Paper](https://arxiv.org/abs/1604.07316)
- Ultralytics: YOLOv11 & Nano
- TensorFlow, OpenCV
- Udacity SDCND inspiration



