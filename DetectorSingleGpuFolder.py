import os
import shutil
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
WATER_DIR = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\val"
# Load the trained model
model = load_model(r"C:\Users\yegor\Downloads\water\wm-nowm\watermark_detection_model_V2_60000_Data_Set.h5")

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Normalize
    return img_array

def predict_watermark(image_paths, batch_size=64):
    predictions = {}
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = np.array([preprocess_image(path) for path in batch_paths])
        batch_predictions = model.predict(batch_images)
        
        for path, prediction in zip(batch_paths, batch_predictions.flatten()):
            predictions[path] = prediction

    return predictions

def filter_images(image_paths, predictions, watermark_threshold=0.5):
    watermarked_dir = fr"{WATER_DIR}\Watermarked"
    not_watermarked_dir = fr"{WATER_DIR}\Not_Watermarked"

    os.makedirs(watermarked_dir, exist_ok=True)
    os.makedirs(not_watermarked_dir, exist_ok=True)

    for img_path, score in predictions.items():
        dest_dir = watermarked_dir if score > watermark_threshold else not_watermarked_dir
        shutil.move(img_path, os.path.join(dest_dir, os.path.basename(img_path)))

if __name__ == "__main__":
    image_dir = WATER_DIR
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    predictions = predict_watermark(image_paths)
    filter_images(image_paths, predictions)
