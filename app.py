import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from model import HybridQuantumTransformer
IMG_SIZE = 32
NUM_QUBITS = 2
with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]
model = HybridQuantumTransformer(NUM_QUBITS, len(class_names))

model.load_state_dict(
    torch.load("hybrid_quantum_model.pth", map_location=torch.device("cpu"))
)

model.eval()
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
def predict(image):

    img = transform(image)

    img = img.unsqueeze(0)

    with torch.no_grad():

        outputs = model(img)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    return class_names[predicted.item()], confidence.item()
st.title("Quantum AI-Based Smart Video Surveillance System")

st.write("Upload an Image or Video")

uploaded_file = st.file_uploader(
    "Choose Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov","tif"]
)

if uploaded_file is not None:

    extension = uploaded_file.name.split(".")[-1].lower()
    if extension in ["jpg", "jpeg", "png"]:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image", use_container_width=True)

        label, confidence = predict(image)

        st.success(f"Prediction : {label}")

        st.write(f"Confidence : {confidence*100:.2f}%")
    else:

        temp_file = "temp_video.mp4"

        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())

        cap = cv2.VideoCapture(temp_file)

        success, frame = cap.read()

        cap.release()
        if success:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = Image.fromarray(frame)

            st.image(image, caption="First Frame", use_container_width=True)

            label, confidence = predict(image)

            st.success(f"Prediction : {label}")

            st.write(f"Confidence : {confidence*100:.2f}%")

        else:
            st.error("Unable to read video.")