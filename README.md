# 👤 AI Face Analyzer (Multi-Task Learning)

An end-to-end computer vision application that detects human faces in real-time and predicts **Age**, **Gender**, and **Ethnicity**. The system utilizes a customized **MobileNetV3** architecture and is deployed via a **Streamlit** web interface.

## 🚀 Features
* **Real-time Detection:** Uses OpenCV Haar Cascades to locate faces in a video stream or uploaded image.
* **Multi-Task Prediction:** A single model pass predicts three distinct attributes simultaneously, optimizing computational efficiency.
* **Dual-Mode UI:** 
    * 📸 **Live Camera:** Capture a selfie via webcam and get instant analysis.
    * 📤 **Image Upload:** Upload high-resolution photos (JPG, PNG) for deep analysis.

---

## 🏗️ System Architecture

The project follows a modular pipeline that transforms raw pixels into structured data.

### 1. The Model (MobileNetV3 Backbone)
The core of the system is a **Multi-Task CNN**. Instead of maintaining three separate models, we use a shared "feature extractor" (MobileNetV3) that branches out into three specialized "heads":
* **Age Head:** A regression layer (Linear) outputting a continuous numerical value.
* **Gender Head:** A classification layer for binary prediction (Male/Female).
* **Ethnicity Head:** A multi-class classification layer for 5 distinct categories.

### 2. The Inference Pipeline
1.  **Face Detection:** OpenCV scans the input image and identifies the bounding box coordinates (x, y, w, h).
2.  **Preprocessing:** The detected face is cropped, converted to grayscale, and resized to **48x48** pixels to match the training input.
3.  **Normalization:** Pixel values are scaled to a range of [0, 1].
4.  **Forward Pass:** The processed tensor is fed into the model to generate the three predictions.

---

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV
* **Frontend:** Streamlit
* **Backbone:** MobileNetV3 (Transfer Learning)
* **Data Handling:** NumPy, PIL, Matplotlib

---

## 📁 Project Structure
```text
├── app.py                # Streamlit Web Application
├── model.py              # FaceAnalysisModel Class Definition
├── face_cnn.pth          # Trained Model Weights (State Dict)
├── requirements.txt      # Project Dependencies
└── notebooks/
    ├── training.ipynb    # Model Training & Validation logic
    └── Face_Detection_System.ipynb  # OpenCV Pipeline Development
```
---

## 🚦 Getting Started

1. Clone the repository

```Bash
git clone [https://github.com/your-username/face-analyzer.git](https://github.com/your-username/face-analyzer.git)
cd face-analyzer
```

2. Install Dependencies

```Bash
uv add streamlit torch torchvision opencv-python pillow numpy
```

3. Run the Application

```Bash
streamlit run app.py
```

## 📈 Model Insights
The model was trained on the UTKFace Dataset, containing over 20,000 face images. By using a shared backbone, the model footprint is significantly smaller and faster than running three independent models, making it ideal for deployment on standard laptops and CPU-based servers.

💡 Future Improvements
[ ] Integrate MediaPipe or MTCNN for more robust face detection in challenging lighting or angles.

[ ] Add Confidence Scores (Softmax probabilities) to show how "sure" the model is.

[ ] Implement Face Tracking for smoother real-time video performance.