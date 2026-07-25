import cv2
import numpy as np
import onnxruntime as ort
import json
import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared Core Functions
def preprocess_frame(frame, target_dim=(640, 640)):
    img = cv2.resize(frame, target_dim)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC to CHW
    img = np.expand_dims(img, axis=0)
    return img

def run_onnx_inference(session, frame, orig_shape, conf_threshold=0.35):
    h_orig, w_orig = orig_shape[:2]
    blob = preprocess_frame(frame)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    predictions = np.squeeze(outputs[0])
    
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T
        
    boxes = []
    confidences = []
    
    for pred in predictions:
        class_scores = pred[4:]
        class_id = np.argmax(class_scores)
        
        # COCO Class 0 = Person
        if class_id == 0 and class_scores[class_id] > conf_threshold:
            conf = class_scores[class_id]
            cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
            
            x1 = int((cx - w / 2) * (w_orig / 640.0))
            y1 = int((cy - h / 2) * (h_orig / 640.0))
            box_w = int(w * (w_orig / 640.0))
            box_h = int(h * (h_orig / 640.0))
            
            boxes.append([x1, y1, box_w, box_h])
            confidences.append(float(conf))
            
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, 0.45)
    final_boxes = []
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            
    return final_boxes

def compute_jersey_team(frame, box):
    x, y, w, h = box
    h_orig, w_orig = frame.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_orig, x + w), min(h_orig, y + int(h * 0.35))
    
    if (x2 - x1) <= 0 or (y2 - y1) <= 0:
        return "Light"
        
    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return "Light" if np.mean(gray_roi) > 120 else "Dark"

@app.websocket("/radar-stream")
async def radar_stream_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 React Dashboard connected.")
    
    try:
        raw_config = await websocket.receive_text()
        config = json.loads(raw_config)
        
        video_source = config.get("videoPath")
        ui_corners = np.array(config.get("corners"), dtype=np.float32)
        
        # Always use the root model asset
        ort_session = ort.InferenceSession("yolov8n.onnx", providers=['CPUExecutionProvider'])
        
        dst_field_corners = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)
        H_matrix, _ = cv2.findHomography(ui_corners, dst_field_corners)
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            await websocket.send_text(json.dumps({"error": f"Unable to read video: {video_source}"}))
            return

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            detected_players = run_onnx_inference(ort_session, frame, frame.shape)
            players_map = {}
            debug_boxes_map = []
            
            for index, box in enumerate(detected_players):
                x, y, w, h = box
                feet_px_x = x + (w // 2)
                feet_px_y = y + h
                
                pixel_vector = np.array([[[feet_px_x, feet_px_y]]], dtype=np.float32)
                warped_vector = cv2.perspectiveTransform(pixel_vector, H_matrix)
                
                pitch_x = float(warped_vector[0][0][0])
                pitch_z = float(warped_vector[0][0][1])
                
                pitch_x = max(0.5, min(104.5, pitch_x))
                pitch_z = max(0.5, min(67.5, pitch_z))
                
                team_group = compute_jersey_team(frame, box)
                player_key = f"ai_p_{index + 1}"
                
                players_map[player_key] = {
                    "id": player_key, "x": round(pitch_x, 2), "z": round(pitch_z, 2), "team": team_group
                }
                
                debug_boxes_map.append({
                    "minX": x, "minY": y, "maxX": x + w, "maxY": y + h, "team": team_group, "weight": 100
                })
                
            await websocket.send_text(json.dumps({
                "players": players_map, "ball": None, "debugBoxes": debug_boxes_map
            }))
            
            await asyncio.sleep(0.033)
            
        cap.release()
    except Exception as error_msg:
        print(f"⚠️ Session encountered runtime drop: {error_msg}")
    finally:
        print("🔌 Stream disconnected.")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)