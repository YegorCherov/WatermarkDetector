import os
import shutil
from ultralytics import YOLO

WATER_DIR = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\val"

# Load the trained model
model = YOLO('yolov8n.pt')

def predict_watermark(image_paths, batch_size=64):
    predictions = {}
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        results = model(batch_paths, verbose=False)
        
        for path, r in zip(batch_paths, results):
            detections = []
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
            predictions[path] = detections

    return predictions

def filter_images(image_paths, predictions, watermark_threshold=0.5):
    watermarked_dir = fr"{WATER_DIR}\Watermarked"
    not_watermarked_dir = fr"{WATER_DIR}\Not_Watermarked"

    os.makedirs(watermarked_dir, exist_ok=True)
    os.makedirs(not_watermarked_dir, exist_ok=True)

    for img_path, detections in predictions.items():
        is_watermarked = any(det['confidence'] > watermark_threshold for det in detections)
        dest_dir = watermarked_dir if is_watermarked else not_watermarked_dir
        shutil.move(img_path, os.path.join(dest_dir, os.path.basename(img_path)))
        status = "Watermarked" if is_watermarked else "Not_Watermarked"
        print(f"Moved {img_path} to {status} (Detections: {len(detections)})")

if __name__ == "__main__":
    image_dir = WATER_DIR
    if os.path.exists(image_dir):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

        predictions = predict_watermark(image_paths)
        filter_images(image_paths, predictions)
    else:
        print(f"Directory {image_dir} does not exist. Please update the path.")
