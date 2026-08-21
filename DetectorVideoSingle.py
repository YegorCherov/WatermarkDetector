import cv2
import os
from ultralytics import YOLO

WATER_DIR = r"C:\Users\yegor\Downloads\water\watermark_detector\videos\LGlIBktB.mp4"

# Load the trained model
model = YOLO('yolov8n.pt')

def predict_watermark_on_frame(frame):
    # YOLO handles BGR frames natively
    results = model(frame, verbose=False)

    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].tolist()
            c = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]

            detections.append({
                'box': b,
                'confidence': c,
                'class_name': cls_name
            })

    return detections

def process_video(video_path, frame_interval=30, watermark_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    watermark_detected = False
    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 'frame_interval' frames
        if frame_count % frame_interval == 0:
            detections = predict_watermark_on_frame(frame)

            # Check if any detection in this frame is above threshold
            high_conf_detections = [det for det in detections if det['confidence'] > watermark_threshold]

            if high_conf_detections:
                watermark_detected = True
                all_detections.extend(high_conf_detections)
                # You might want to break here if you just need a boolean yes/no
                # break

        frame_count += 1

    cap.release()
    return watermark_detected, all_detections

if __name__ == "__main__":
    # Process a video file
    video_path = WATER_DIR
    if os.path.exists(video_path):
        is_watermarked, detections = process_video(video_path)

        print(f"Video '{video_path}' is {'watermarked' if is_watermarked else 'not watermarked'}")
        if is_watermarked:
            print(f"Total significant detections found across sampled frames: {len(detections)}")
            # Optional: print first few detections
            for i, det in enumerate(detections[:5]):
                print(f"  Sample Detection {i+1}: Class: {det['class_name']}, Confidence: {det['confidence']:.2f}, Box: {det['box']}")
    else:
         print(f"Video file {video_path} does not exist. Please update the path.")
