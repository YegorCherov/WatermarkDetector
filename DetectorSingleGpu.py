import os
from ultralytics import YOLO

# Load the trained model (YOLO natively uses GPU if available)
model = YOLO('yolov8n.pt')

def predict_watermark(image_paths, batch_size=64):
    predictions = {}

    # Process in batches manually or pass the whole list to YOLO
    # For very large lists, manual batching avoids OOM
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

if __name__ == "__main__":
    # Directory containing images
    image_dir = r'C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420'

    if os.path.exists(image_dir):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

        # Process images and get predictions
        predictions = predict_watermark(image_paths)

        # Output results
        for img_path, detections in predictions.items():
            if detections:
                print(f"{img_path}: Watermark detected!")
                for i, det in enumerate(detections):
                    print(f"  Detection {i+1}: Class: {det['class_name']}, Confidence: {det['confidence']:.2f}, Box: {det['box']}")
            else:
                print(f"{img_path}: No watermark detected")
    else:
        print(f"Directory {image_dir} does not exist. Please update the path.")
