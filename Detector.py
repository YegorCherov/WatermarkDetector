import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from multiprocessing import Pool, cpu_count

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

def process_images(image_paths):
    with Pool(cpu_count()) as p:
        results = p.map(predict_watermark, image_paths)
    return results

if __name__ == "__main__":
    # Directory containing images
    image_dir = r'C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420'
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # Process images
    predictions = process_images(image_paths)

    # Output results
    for img_path, score in predictions:
        print(f"{img_path}: {'Watermark detected' if score > 0.5 else 'No watermark detected'}")

