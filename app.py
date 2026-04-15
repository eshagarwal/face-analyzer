import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from model import FaceAnalysisModel  # Importing your saved class

# --- 1. SETTINGS & MODEL LOADING ---
st.set_page_config(page_title="AI Face Analyzer", layout="centered")

@st.cache_resource # This keeps the model in memory so it doesn't reload every click
def load_trained_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = FaceAnalysisModel(backbone_requires_grad=False).to(device)
    model.load_state_dict(torch.load("face_cnn.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_trained_model()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

genders = ["Male", "Female"]
ethnicities = ["White", "Black", "Asian", "Indian", "Others"]

# --- 2. PIPELINE LOGIC ---
def process_image(img_array):
    # Convert BGR (OpenCV) to Grayscale and RGB
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    display_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        # Crop & Transform
        face_roi = gray[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_roi)
        
        transform = T.Compose([T.Resize((48, 48)), T.ToTensor()])
        face_tensor = transform(face_pil).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            age_out, gen_out, eth_out = model(face_tensor)
        
        age = age_out.item()
        gender = genders[gen_out.argmax(1).item()]
        eth = ethnicities[eth_out.argmax(1).item()]
        
        # Draw on display image
        label = f"{int(age)}yrs | {gender} | {eth}"
        cv2.rectangle(display_img, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.putText(display_img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
    return display_img

# --- 3. FRONTEND TABS ---
st.title("👤 AI Face Analyzer")
tab1, tab2 = st.tabs(["📸 Live Camera", "📤 Upload Image"])

with tab1:
    st.header("Camera Capture")
    img_file = st.camera_input("Take a photo to analyze")
    if img_file:
        # Convert the file to an OpenCV-ready array
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        result_img = process_image(cv2_img)
        st.image(result_img, caption="Analysis Result", use_column_width=True)

with tab2:
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Choose a face photo...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        cv2_img = cv2.imdecode(file_bytes, 1)
        
        result_img = process_image(cv2_img)
        st.image(result_img, caption="Analysis Result", use_column_width=True)