import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model(r"C:\Users\yegor\Downloads\water\wm-nowm\watermark_detection_model.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

def predict_watermark(img_path):
    img_to_predict = preprocess_image(img_path)
    prediction = model.predict(img_to_predict)
    return img_path, prediction[0][0]

def organize_images(image_paths, watermarked_dir, not_watermarked_dir):
    for img_path in image_paths:
        _, score = predict_watermark(img_path)
        print(score, img_path)
        if score > 0.3:
            # Move to watermarked folder
            shutil.move(img_path, os.path.join(watermarked_dir, os.path.basename(img_path)))
        else:
            # Move to not watermarked folder
            shutil.move(img_path, os.path.join(not_watermarked_dir, os.path.basename(img_path)))

if __name__ == "__main__":
    # Directory containing images
    image_dir = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420"
    watermarked_dir = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420\Watermarked"
    not_watermarked_dir = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420\Not_Watermarked"

    # Create directories if they don't exist
    os.makedirs(watermarked_dir, exist_ok=True)
    os.makedirs(not_watermarked_dir, exist_ok=True)

    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # Organize images
    organize_images(image_paths, watermarked_dir, not_watermarked_dir)
