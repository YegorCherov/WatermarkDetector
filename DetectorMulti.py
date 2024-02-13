import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from multiprocessing import Pool

# Load the trained model
model = load_model(r"C:\Users\yegor\Downloads\water\wm-nowm\watermark_detection_model.h5")

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

def predict_watermark_batch(image_batch):
    results = []
    for img_path in image_batch:
        img_to_predict = preprocess_image(img_path)
        prediction = model.predict(img_to_predict)
        results.append((img_path, prediction[0][0]))
    return results

def process_images_in_batches(image_paths, num_threads):
    # Split the image paths into batches
    batch_size = len(image_paths) // num_threads
    image_batches = [image_paths[i:i + batch_size] for i in range(0, len(image_paths), batch_size)]

    # Process each batch in a separate thread
    with Pool(num_threads) as p:
        all_results = p.map(predict_watermark_batch, image_batches)
    return [item for sublist in all_results for item in sublist]

if __name__ == "__main__":
    # Directory containing images
    image_dir = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420"
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # Number of threads equal to CPU threads
    num_threads = os.cpu_count()

    # Process images in batches
    predictions = process_images_in_batches(image_paths, num_threads)

    # Output results
    for img_path, score in predictions:
        print(f"{img_path}: {'Watermark detected' if score > 0.5 else 'No watermark detected'}")
