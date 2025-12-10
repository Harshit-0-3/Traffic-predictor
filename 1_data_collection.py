import cv2
import csv
from ultralytics import YOLO

# --- CONFIGURATION ---
VIDEO_PATH = "traffic_video.mp4" # Make sure this matches your video name
DATA_FILE = "traffic_dataset.csv"

# 1. Load the AI Model (This counts the cars for us)
print("Loading AI Model...")
model = YOLO('yolov8n.pt') 

# 2. Open the video
cap = cv2.VideoCapture(VIDEO_PATH)

# 3. Open a CSV file to save our data
# We are creating a table with headers: Frame, Car_Count, Traffic_Level, Wait_Time
f = open(DATA_FILE, 'w', newline='')
writer = csv.writer(f)
writer.writerow(["Frame_ID", "Car_Count", "Traffic_Level", "Wait_Time"])

print("🎥 Watching video and collecting data... (This might take a minute)")

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break # Video ended

    frame_id += 1
    
    # Analyze only every 10th frame (to make it faster)
    if frame_id % 10 != 0:
        continue

    # Detect objects in the frame
    results = model(frame, verbose=False)
    
    # Count cars, buses, trucks
    vehicle_count = 0
    for box in results[0].boxes:
        # Class IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
        if int(box.cls) in [2, 3, 5, 7]:
            vehicle_count += 1

    # --- SIMULATE THE GROUND TRUTH ---
    # In a real project, a human would measure this. Here, we simulate logic 
    # so our Data Science model has something to learn later.
    
    wait_time = 0
    traffic_level = "Low"

    if vehicle_count < 5:
        traffic_level = "Low"
        wait_time = 10  # Seconds
    elif vehicle_count < 15:
        traffic_level = "Medium"
        wait_time = 45 # Seconds
    else:
        traffic_level = "High"
        wait_time = 90 # Seconds

    # Save to CSV
    writer.writerow([frame_id, vehicle_count, traffic_level, wait_time])
    
    # Show progress every 50 frames
    if frame_id % 50 == 0:
        print(f"Frame {frame_id}: Saw {vehicle_count} vehicles -> Saved to CSV")

# Cleanup
cap.release()
f.close()
print(f"✅ SUCCESS! Data saved to {DATA_FILE}")