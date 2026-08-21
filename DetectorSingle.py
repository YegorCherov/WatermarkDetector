import os
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

if __name__ == "__main__":
    # Directory containing images
    image_dir = r"C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420"

    if os.path.exists(image_dir):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

        predictions = []
        for img_path in image_paths:
            predictions.append(predict_watermark(img_path))

        # Output results
        for img_path, detections in predictions:
            if detections:
                print(f"{img_path}: Watermark detected!")
                for i, det in enumerate(detections):
                    print(f"  Detection {i+1}: Class: {det['class_name']}, Confidence: {det['confidence']:.2f}, Box: {det['box']}")
            else:
                print(f"{img_path}: No watermark detected")
    else:
        print(f"Directory {image_dir} does not exist. Please update the path.")
