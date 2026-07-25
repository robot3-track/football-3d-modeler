import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import cv2
import numpy as np
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--points', type=str, default="0,0,100,0,100,100,0,100")
parser.add_argument('--ui_w', type=float, default=1.0)
parser.add_argument('--ui_h', type=float, default=1.0)
parser.add_argument('--vid_w', type=float, default=1.0)
parser.add_argument('--vid_h', type=float, default=1.0)
args = parser.parse_args()

# Ratio mapping between layout display window and file dimensions
scale_x = args.vid_w / args.ui_w
scale_y = args.vid_h / args.ui_h

p = [float(x) for x in args.points.split(',')]

# Explicit raw pixel matrix mappings derived from layout click events
pts_src = np.array([
    [p[0] * scale_x, p[1] * scale_y], # Top-Left
    [p[2] * scale_x, p[3] * scale_y], # Top-Right
    [p[4] * scale_x, p[5] * scale_y], # Bottom-Right
    [p[6] * scale_x, p[7] * scale_y]  # Bottom-Left
], dtype=float)

# Maps precisely onto a standard 105m long by 68m wide coordinate grid system
pts_dst = np.array([
    [0, 0], [105, 0], [105, 68], [0, 68]
], dtype=float)

H, _ = cv2.findHomography(pts_src, pts_dst)

def convert_to_pitch_coords(pixel_x, pixel_y, homography_matrix):
    point = np.array([[[pixel_x, pixel_y]]], dtype='float32')
    transformed = cv2.perspectiveTransform(point, homography_matrix)
    return round(float(transformed[0][0][0]), 2), round(float(transformed[0][0][1]), 2)

onnx_path = "yolov8n.onnx"
net = cv2.dnn.readNet(onnx_path)

video_path = "match_sample.mp4"
cap = cv2.VideoCapture(video_path)
tracking_timeline = {}
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_count += 1
    
    img_h, img_w, _ = frame.shape
    
    # 1. Standard YOLOv8 ONNX Input Conformation Preprocessing
    input_width, input_height = 640, 640
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (input_width, input_height), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward()
    
    # Transpose format to isolate candidate coordinate records array
    predictions = np.squeeze(outputs[0]).T  
    
    # Coordinate scaling vectors mapping predictions straight back to the original video resolution
    x_factor = img_w / input_width
    y_factor = img_h / input_height
    
    boxes, scores, class_ids = [], [], []
    
    for pred in predictions:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        score = float(class_scores[class_id])
        
        # Isolate Person (0) and Sports Ball (32) classes
        if (class_id == 0 and score > 0.30) or (class_id == 32 and score > 0.10):
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            
            # Map back out of 640 grid space directly to original video canvas resolution
            left = int((cx - 0.5 * w) * x_factor)
            top = int((cy - 0.5 * h) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)
            
            boxes.append([left, top, width, height])
            scores.append(score)
            class_ids.append(class_id)
            
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.10, 0.45)
    
    frame_players = {}
    frame_ball = None
    player_id = 0
    
    if len(indices) > 0:
        flat_indices = indices.flatten()
        for index in flat_indices:
            bx, by, bw, bh = boxes[index]
            c_id = class_ids[index]
            
            # Use lower center edge anchor point for precise pitch floor alignment
            feet_x = bx + (bw / 2)
            feet_y = by + bh
            
            pitch_x, pitch_z = convert_to_pitch_coords(feet_x, feet_y, H)
            
            # Allow clean boundary tolerances
            if -10 <= pitch_x <= 115 and -10 <= pitch_z <= 78:
                if c_id == 0:
                    player_id += 1
                    team = "A" if player_id % 2 == 0 else "B"
                    frame_players[str(player_id)] = {"x": pitch_x, "z": pitch_z, "team": team}
                elif c_id == 32:
                    frame_ball = {"x": pitch_x, "z": pitch_z}
                    
    tracking_timeline[str(frame_count)] = {
        "players": frame_players,
        "ball": frame_ball
    }
    
    if frame_count >= 150:
        break

cap.release()

output_path = os.path.join("public", "tracking_data.json")
with open(output_path, 'w') as f:
    json.dump(tracking_timeline, f, indent=2)

print("SUCCESS: Target alignment tracking maps successfully generated.")