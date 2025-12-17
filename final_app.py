import streamlit as st
import cv2
import joblib
import numpy as np
from ultralytics import YOLO
import tempfile

# --- PAGE SETUP ---
st.set_page_config(page_title="Traffic Wait Predictor", layout="wide")

st.title("🚦 AI Traffic Wait Predictor")
st.write("This system uses Computer Vision to count cars and a Machine Learning model to predict wait times.")

# --- SIDEBAR ---
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("AI Confidence Threshold", 0.0, 1.0, 0.45)

# --- LOAD MODELS ---
# 1. Load the YOLO Object Detection Model
@st.cache_resource # This makes sure we only load it once (faster)
def load_yolo():
    return YOLO('yolov8n.pt')

# 2. Load Your Trained Data Science Model
@st.cache_resource
def load_prediction_model():
    return joblib.load('traffic_model.pkl')

yolo_model = load_yolo()
wait_time_model = load_prediction_model()

# --- APP LAYOUT ---
col1, col2 = st.columns([2, 1]) # Two columns: Video (Left) and Stats (Right)

with col1:
    st_frame = st.empty() # Placeholder for the video

with col2:
    st_stat_cars = st.empty()
    st_stat_wait = st.empty()
    st_chart = st.empty()

# --- VIDEO LOGIC ---
video_file = "traffic_video2.mp4" # We use your existing video
cap = cv2.VideoCapture(video_file)

# Lists to store data for the live chart
chart_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Video finished.")
        break

    # 1. Object Detection (The Eyes)
    results = yolo_model(frame, conf=confidence_threshold, verbose=False)
    
    # 2. Count Cars
    car_count = 0
    for box in results[0].boxes:
        if int(box.cls) in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
            car_count += 1
            # Draw box on frame (Optional visual)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 3. Predict Wait Time (The Brain)
    # We need to reshape the input because the model expects a list of lists [[car_count]]
    prediction_input = np.array([[car_count]])
    predicted_wait = wait_time_model.predict(prediction_input)[0]

    # --- UPDATE DISPLAY ---
    
    # Show Video
    # Convert BGR (OpenCV color) to RGB (Web color)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st_frame.image(frame, channels="RGB", use_container_width=True)

    # Show Stats
    st_stat_cars.metric(label="🚗 Cars Detected", value=car_count)
    
    # Color code the wait time (Green=Fast, Red=Slow)
    if predicted_wait < 30:
        color = "normal"
    elif predicted_wait < 60:
        color = "off" # Streamlit uses weird names for colors, 'off' is greyish
    else:
        color = "inverse" # 'inverse' is often red/dark in Streamlit

    st_stat_wait.metric(label="⏱️ Estimated Wait Time", value=f"{int(predicted_wait)} seconds")

    # Update Live Chart
    chart_data.append(predicted_wait)
    if len(chart_data) > 50: # Keep only last 50 frames to keep graph moving
        chart_data.pop(0)
    st_chart.line_chart(chart_data)

cap.release()