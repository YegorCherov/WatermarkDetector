import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model (this should automatically use the GPU)
model = load_model(r"C:\Users\yegor\Downloads\water\wm-nowm\watermark_detection_model.h5")

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Normalize
    return img_array

def predict_watermark(image_paths, batch_size=64):  # Updated batch size
    predictions = {}
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = np.array([preprocess_image(path) for path in batch_paths])
        batch_predictions = model.predict(batch_images)
        
        for path, prediction in zip(batch_paths, batch_predictions.flatten()):
            predictions[path] = prediction

    return predictions

# Directory containing images
image_dir = r'C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420'
image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg')]

# Process images and get predictions
predictions = predict_watermark(image_paths)

# Output results
for img_path, score in predictions.items():
    print(f"{img_path}: {'Watermark detected' if score > 0.5 else 'No watermark detected'}")
