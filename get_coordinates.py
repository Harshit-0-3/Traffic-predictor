import cv2
import numpy as np

# 1. Load the video
video_path = "traffic_video.mp4" 
cap = cv2.VideoCapture(video_path)

# This list will hold the points you click
points = []

def draw_polygon(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"📍 Point clicked: [{x}, {y}]")

cv2.namedWindow("Click 4 Corners of the Lane")
cv2.setMouseCallback("Click 4 Corners of the Lane", draw_polygon)

# 🔴 FIX 1: Skip black frames
cap.set(cv2.CAP_PROP_POS_FRAMES, 100) 

ret, frame = cap.read()

if not ret:
    print("❌ Error: Could not read video.")
    exit()

# 🔴 FIX 2: Resize frame to match the App EXACTLY
frame = cv2.resize(frame, (640, 480))

print("👉 INSTRUCTIONS:")
print("1. Click 4 points around the ONE LANE you want to track.")
print("2. Make sure to click in a CIRCLE (Clockwise): Bottom-Left -> Top-Left -> Top-Right -> Bottom-Right")
print("3. Press 'q' to close and get your code.")

while True:
    display_frame = frame.copy()
    
    # Draw circles
    for point in points:
        cv2.circle(display_frame, (point[0], point[1]), 5, (0, 0, 255), -1)
    
    # Draw lines
    if len(points) > 1:
        cv2.polylines(display_frame, [np.array(points, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow("Click 4 Corners of the Lane", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*40)
print("✅ PASTE THIS INTO YOUR APP (Replace the old numbers):")
print(f"lane_zone = np.array({points}, np.int32)")
print("="*40)