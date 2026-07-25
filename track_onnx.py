import cv2
import numpy as np
import onnxruntime as ort
import sys

# --- INPUT CONFIGURATION ---
VIDEO_FILE = "your_local_video_file.mp4" # Path to your local video asset

# Match the ordering: Top-Left, Top-Right, Bottom-Right, Bottom-Left
MANUAL_PIXEL_CORNERS = np.array([
    [320, 180],  
    [960, 180], 
    [1150, 680], 
    [130, 680]
], dtype=np.float32)

# Shared Core Functions
def preprocess_frame(frame, target_dim=(640, 640)):
    img = cv2.resize(frame, target_dim)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
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

def main():
    print("⚙️ Booting Standalone Test Bench using yolov8n.onnx...")
    
    try:
        session = ort.InferenceSession("yolov8n.onnx", providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"❌ Error loading yolov8n.onnx: {e}")
        sys.exit(1)
        
    dst_field_dims = np.array([[0, 0], [105, 0], [105, 68], [0, 68]], dtype=np.float32)
    H_matrix, _ = cv2.findHomography(MANUAL_PIXEL_CORNERS, dst_field_dims)
    
    cap = cv2.VideoCapture(VIDEO_FILE)
    if not cap.isOpened():
        print(f"❌ Error: Could not parse video file at {VIDEO_FILE}")
        sys.exit(1)
        
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        detected_players = run_onnx_inference(session, frame, frame.shape)
        
        for box in detected_players:
            x, y, w, h = box
            feet_x = x + (w // 2)
            feet_y = y + h
            
            px_vector = np.array([[[feet_x, feet_y]]], dtype=np.float32)
            transformed_vector = cv2.perspectiveTransform(px_vector, H_matrix)
            
            real_x = float(transformed_vector[0][0][0])
            real_z = float(transformed_vector[0][0][1])
            
            team_color = compute_jersey_team(frame, box)
            draw_color = (255, 255, 255) if team_color == "Light" else (0, 0, 0)
            
            # Draw visual test framework layout on display window
            cv2.rectangle(frame, (x, y), (x + w, y + h), draw_color, 2)
            cv2.circle(frame, (feet_x, feet_y), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"{real_x:.1f}m, {real_z:.1f}m", (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                
        cv2.polylines(frame, [MANUAL_PIXEL_CORNERS.astype(np.int32)], True, (0, 165, 255), 2)
        cv2.imshow("Tactical Radar Local Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()