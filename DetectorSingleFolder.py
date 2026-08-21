import os
import shutil
from ultralytics import YOLO

# Load the trained model
model = YOLO('yolov8n.pt')

def predict_watermark(img_path):
    results = model(img_path, verbose=False)

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

    return img_path, detections

def organize_images(image_paths, watermarked_dir, not_watermarked_dir, threshold=0.3):
    for img_path in image_paths:
        _, detections = predict_watermark(img_path)

        # Check if any detection has confidence above threshold
        is_watermarked = any(det['confidence'] > threshold for det in detections)

        if is_watermarked:
            # Move to watermarked folder
            shutil.move(img_path, os.path.join(watermarked_dir, os.path.basename(img_path)))
            print(f"Moved {img_path} to Watermarked (Detections: {len(detections)})")
        else:
            # Move to not watermarked folder
            shutil.move(img_path, os.path.join(not_watermarked_dir, os.path.basename(img_path)))
            print(f"Moved {img_path} to Not_Watermarked")

if __name__ == "__main__":
    # Directory containing images
    image_dir = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420"
    watermarked_dir = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420\Watermarked"
    not_watermarked_dir = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420\Not_Watermarked"

    if os.path.exists(image_dir):
        # Create directories if they don't exist
        os.makedirs(watermarked_dir, exist_ok=True)
        os.makedirs(not_watermarked_dir, exist_ok=True)

        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

        # Organize images
        organize_images(image_paths, watermarked_dir, not_watermarked_dir)
    else:
        print(f"Directory {image_dir} does not exist. Please update the path.")
