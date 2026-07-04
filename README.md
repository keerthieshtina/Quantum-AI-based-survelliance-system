# Quantum AI-Based Smart Video Surveillance System

## Overview
The Quantum AI-Based Smart Video Surveillance System is an intelligent surveillance application that combines Deep Learning and Quantum Machine Learning for real-time activity detection. The system analyzes images and video frames to classify different surveillance events such as abuse, assault, burglary, explosion, fighting, robbery, shoplifting, vandalism, and normal activities.

This project explores the integration of Quantum Neural Networks (QNNs) with deep learning to improve video classification performance.

---

## Features

- Real-time surveillance activity detection
- Hybrid Quantum Transformer architecture
- Quantum Neural Network (QNN) using Qiskit
- Image and video frame classification
- Streamlit-based web interface
- Model trained using PyTorch
- Supports multiple surveillance event classes

---

## Technologies Used

- Python
- PyTorch
- OpenCV
- Qiskit
- Qiskit Machine Learning
- NumPy
- Streamlit
- Scikit-learn
- Matplotlib

---

## Dataset

The model was trained using the **UCF-crime Dataset** containing multiple surveillance activities.

Example Classes:

- Abuse
- Arrest
- Arson
- Assault
- Burglary
- Explosion
- Fighting
- Normal
- Robbery
- Road Accident
- Shooting
- Shoplifting
- Stealing
- Vandalism

---

## Project Structure

```
Quantum_AI_Surveillance/
│
├── train_model.py
├── hybrid_quantum_model.pth
├── class_names.txt
├── requirements.txt
├── README.md
```

---

## Installation
Clone the repository

```bash
git clone https://github.com/keerthieshtina/Quantum-AI-based-survelliance-system.git
```

Move to the project folder

```bash
cd Quantum_AI_Surveillance
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Training

```bash
python train_model.py
```

The trained model will be saved as:

```
hybrid_quantum_model.pth
```

---

## Run the Web Application

```bash
streamlit run app.py
```

---

## Model

The proposed model consists of:

- Convolutional Neural Network (CNN)
- Quantum Attention Layer
- Quantum Neural Network (QNN)
- Hybrid Quantum Transformer
- Fully Connected Classification Layer

---

## Results

- Successfully trained on surveillance image/video data
- Real-time activity classification
- Improved classification performance compared to a baseline CNN model
- Efficient feature extraction using a hybrid quantum-classical architecture

---

## Future Enhancements

- Live CCTV camera integration
- Multi-person activity detection
- Object detection with YOLO
- Alert notification system
- Cloud deployment
- Quantum hardware execution

---

## Author

**Keerthi Eshtina**

Aspiring Software Engineer | AI & Machine Learning Enthusiast