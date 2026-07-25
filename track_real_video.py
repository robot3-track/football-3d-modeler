import cv2
import numpy as np
import json
import os

print("Initializing Pure-CV Local Tracking Engine (No PyTorch)...")

# 1. SETUP HOMOGRAPHY (Maps pixels to the 105m x 68m grid)
pts_src = np.array([
    [350, 200], [800, 220], [950, 600], [200, 550]
], dtype=float)

pts_dst = np.array([
    [0, 18], [16.5, 18], [16.5, 50], [0, 50]
], dtype=float)

H, _ = cv2.findHomography(pts_src, pts_dst)

def convert_to_pitch_coords(pixel_x, pixel_y, homography_matrix):
    point = np.array([[[pixel_x, pixel_y]]], dtype='float32')
    transformed = cv2.perspectiveTransform(point, homography_matrix)
    return round(float(transformed[0][0][0]), 2), round(float(transformed[0][0][1]), 2)

# 2. LOAD OPENCV DNN FIELD DETECTOR 
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels (we only care about class 0: person)
with open(os.devnull, "w") as f:
    # Just an array filter check for person objects
    classes = ["person"]

# 3. OPEN REAL VIDEO FEED
video_path = "match_sample.mp4"
if not os.path.exists(video_path):
    raise FileNotFoundError(f"Missing file: Place your video clip in this folder and name it '{video_path}'")

cap = cv2.VideoCapture(video_path)
tracking_timeline = {}
frame_count = 0

print("🎥 Processing video frames via DNN matrix...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1
    
    height, width, channels = frame.shape
    
    # Preprocess image frame for the network
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    frame_data = {}
    person_idx = 0
    
    # Parse bounding boxes
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Confidence threshold for human tracking
            if class_id == 0 and confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Bottom mid point of the bounding box (the player's feet)
                feet_x = center_x
                feet_y = center_y + (h / 2)
                
                pitch_x, pitch_z = convert_to_pitch_coords(feet_x, feet_y, H)
                
                person_idx += 1
                # Alternate teams simply by object indexing parity for data structuring
                team = "A" if person_idx % 2 == 0 else "B"
                
                frame_data[str(person_idx)] = {"x": pitch_x, "z": pitch_z, "team": team}
                
    tracking_timeline[str(frame_count)] = frame_data
    
    if frame_count >= 200: # process a 200 frame sample segment
        break

cap.release()

# 4. EXPORT JSON DIRECTLY INTO THE NEXT.JS PATHWAY
output_path = os.path.join("public", "tracking_data.json")
os.makedirs("public", exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(tracking_timeline, f, indent=2)

print(f"Success! Real video positions saved directly to: {output_path}")