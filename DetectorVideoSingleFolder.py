import cv2
import os
import shutil
from ultralytics import YOLO

WATER_DIR = r"C:\Users\yegor\Downloads\water\watermark_detector\videos"

# Load the trained model
model = YOLO('yolov8n.pt')

def predict_watermark_on_frame(frame):
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

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            detections = predict_watermark_on_frame(frame)

            # Check if any detection is above threshold
            if any(det['confidence'] > watermark_threshold for det in detections):
                watermark_detected = True
                break

        frame_count += 1

    cap.release()
    return watermark_detected

def sort_videos(video_dir):
    watermarked_dir = os.path.join(video_dir, 'Watermarked')
    not_watermarked_dir = os.path.join(video_dir, 'Not_Watermarked')

    os.makedirs(watermarked_dir, exist_ok=True)
    os.makedirs(not_watermarked_dir, exist_ok=True)

    for video_file in os.listdir(video_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_dir, video_file)
            is_watermarked = process_video(video_path)

            target_dir = watermarked_dir if is_watermarked else not_watermarked_dir
            shutil.move(video_path, os.path.join(target_dir, video_file))
            print(f"Video '{video_file}' moved to {'watermarked' if is_watermarked else 'not_watermarked'} folder.")

if __name__ == "__main__":
    video_directory = WATER_DIR
    if os.path.exists(video_directory):
        sort_videos(video_directory)
    else:
        print(f"Directory {video_directory} does not exist. Please update the path.")
