import streamlit as st
import cv2
import joblib
import numpy as np
from ultralytics import YOLO

# --- PAGE CONFIG ---
st.set_page_config(page_title="Smart Traffic AI", layout="wide")
st.title("🚦 AI Traffic Filter & Predictor")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("🔧 Settings")
confidence = st.sidebar.slider("AI Confidence", 0.0, 1.0, 0.45)
show_zone = st.sidebar.checkbox("Show Lane Zone", value=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    # Load YOLO Model
    yolo = YOLO('yolov8n.pt')
    
    # Load Wait Time Model (Handle error if file is missing)
    try:
        wait_model = joblib.load('traffic_model.pkl')
    except:
        wait_model = None 
    return yolo, wait_model

yolo_model, wait_time_model = load_models()

# --- 1. VIDEO SOURCE ---
video_file = "traffic_video2.mp4"  # Matches your renamed file
cap = cv2.VideoCapture(video_file)

# --- 2. DEFINE THE ZONE (THE FILTER) ---
# Your coordinates from the tool
lane_zone = np.array([[6, 470], [363, 118], [490, 129], [500, 478]], np.int32)

# --- LAYOUT SETUP (THE FIX) ---
col1, col2 = st.columns([2, 1])

# Create placeholders BEFORE the loop starts
# This tells Streamlit: "Reserve this spot on the screen, don't create new ones."
with col1:
    st_frame = st.empty()  # Placeholder for the video

with col2:
    st.markdown("### Live Statistics")
    kpi1, kpi2 = st.columns(2)
    
    st_kpi1 = kpi1.empty() # Placeholder for Car Count
    st_kpi2 = kpi2.empty() # Placeholder for Wait Time
    
    st.markdown("### Wait Time Trend")
    st_chart_place = st.empty() # Placeholder for the Graph

chart_data = []

# --- MAIN LOOP ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Video finished.")
        break
    
    # Standardize frame size to match your coordinates
    frame = cv2.resize(frame, (640, 480))

    # AI Detection
    results = yolo_model(frame, conf=confidence, verbose=False)
    
    filtered_car_count = 0

    # Process Detections
    for box in results[0].boxes:
        # Class IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        if int(box.cls) in [2, 3, 5, 7]: 
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Calculate Center
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            
            # Check if inside the Zone
            is_inside = cv2.pointPolygonTest(lane_zone, (cx, cy), False)
            
            if is_inside >= 0:
                filtered_car_count += 1
                # Green for Valid
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
            else:
                # Red for Ignored
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # Draw Zone Logic
    if show_zone:
        cv2.polylines(frame, [lane_zone], isClosed=True, color=(255, 0, 0), thickness=2)

    # Predict Wait Time
    wait_time = 0
    if wait_time_model:
        wait_time = wait_time_model.predict(np.array([[filtered_car_count]]))[0]

    # --- UPDATE DISPLAY (THE FIX) ---
    # Update Video
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st_frame.image(frame, channels="RGB", use_container_width=True)
    
    # Update Stats (Overwriting the placeholders)
    st_kpi1.metric("🚗 Cars in Lane", filtered_car_count)
    st_kpi2.metric("⏱️ Wait Time", f"{int(wait_time)}s")
    
    # Update Chart
    chart_data.append(wait_time)
    if len(chart_data) > 50: chart_data.pop(0)
    st_chart_place.line_chart(chart_data)

cap.release()