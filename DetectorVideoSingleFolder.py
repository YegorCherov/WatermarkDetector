import cv2
import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
WATER_DIR = r"C:\Users\yegor\Downloads\water\watermark_detector\videos"
# Load the trained model
model = load_model(r"C:\Users\yegor\Downloads\water\wm-nowm\watermark_detection_model_V2_60000_Data_Set.h5")

def preprocess_frame(frame, target_size=(224, 224)):
    frame = cv2.resize(frame, target_size)
    frame_array = image.img_to_array(frame)
    frame_array /= 255.0  # Normalize
    return np.expand_dims(frame_array, axis=0)  # Add batch dimension

def predict_watermark_on_frame(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)
    return prediction[0][0]

def process_video(video_path, frame_interval=30, watermark_threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    watermark_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            prediction = predict_watermark_on_frame(frame)
            if prediction > watermark_threshold:
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
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            is_watermarked = process_video(video_path)

            target_dir = watermarked_dir if is_watermarked else not_watermarked_dir
            shutil.move(video_path, os.path.join(target_dir, video_file))
            print(f"Video '{video_file}' moved to {'watermarked' if is_watermarked else 'not_watermarked'} folder.")

if __name__ == "__main__":
    video_directory = WATER_DIR
    sort_videos(video_directory)
