import cv2
import numpy as np
import json
import os

print("Initializing Pure-CV Local Tracking Engine (No PyTorch)...")

# 1. SETUP HOMOGRAPHY Matrix Mappings
# These pixel markers must be adjusted if your UI resolution scale changes
pts_src = np.array([
    [350, 200], # 1. Top-Left click
    [800, 220], # 2. Top-Right click
    [950, 600], # 3. Bottom-Right click
    [200, 550]  # 4. Bottom-Left click
], dtype=float)

# FIXED: Mapped click points to the full 105m x 68m boundaries instead of a partial penalty box
pts_dst = np.array([
    [0, 0],      # 1. Top-Left Corner of Pitch
    [105, 0],    # 2. Top-Right Corner of Pitch
    [105, 68],   # 3. Bottom-Right Corner of Pitch
    [0, 68]      # 4. Bottom-Left Corner of Pitch
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
    
    boxes = []
    confidences = []
    
    # Parse raw detection footprints
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # Extract person class (0) with structural confidence threshold filter
            if class_id == 0 and confidence > 0.4:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Transform to standard top-left bounding box anchors
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
    
    # FIXED: Non-Maximum Suppression added to deduplicate hundreds of duplicate ghost overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
    
    frame_players = {}
    person_idx = 0
    
    if len(indices) > 0:
        flat_indices = indices.flatten() if hasattr(indices, 'flatten') else indices
        
        for index in flat_indices:
            bx, by, bw, bh = boxes[index]
            
            # Ground anchor alignment logic: bottom-mid point of the bounding box
            feet_x = bx + (bw / 2)
            feet_y = by + bh
            
            pitch_x, pitch_z = convert_to_pitch_coords(feet_x, feet_y, H)
            
            # Check bounding frame threshold limits
            if -10 <= pitch_x <= 115 and -10 <= pitch_z <= 78:
                person_idx += 1
                team = "A" if person_idx % 2 == 0 else "B"
                
                # FIXED: Maps out both 'y' and 'z' coordinate properties to prevent 3D engine flattening bugs
                frame_players[str(person_idx)] = {
                    "x": pitch_x, 
                    "y": pitch_z, 
                    "z": pitch_z, 
                    "team": team
                }
                
    tracking_timeline[str(frame_count)] = {
        "players": frame_players,
        "ball": None
    }
    
    if frame_count >= 200: 
        break

cap.release()

# 4. EXPORT JSON DATA MATRIX DIRECTLY INTO THE PUBLIC FOLDER
output_path = os.path.join("public", "tracking_data.json")
os.makedirs("public", exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(tracking_timeline, f, indent=2)

print(f"\nSUCCESS: Radar tracking telemetry assets exported to -> {output_path}")