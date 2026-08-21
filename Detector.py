import os
from ultralytics import YOLO

# Load the trained model
# Note: Update this path to the location of your best.pt file from training
model = YOLO('yolov8n.pt')

def predict_watermark(img_path):
    # YOLO handles preprocessing internally
    results = model(img_path, verbose=False)

    # We can process the results
    # Since we might have multiple detections per image, we collect them
    detections = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Extract bounding box, confidence, and class
            b = box.xyxy[0].tolist()  # get box coordinates in (left, top, right, bottom) format
            c = box.conf[0].item()    # get confidence score
            cls_id = int(box.cls[0].item()) # get class id
            cls_name = model.names[cls_id] # get class name

            detections.append({
                'box': b,
                'confidence': c,
                'class_name': cls_name
            })

    return img_path, detections

if __name__ == "__main__":
    # Directory containing images
    image_dir = r'C:\Users\yegor\Downloads\water\WatermarkDataset\images\train 420'

    # Check if directory exists before trying to list it (useful for testing on different machines)
    if os.path.exists(image_dir):
        image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

        # Process images sequentially (YOLO is fast enough without multiprocessing for basic loops, and multiprocessing with large models can have overhead)
        # We can also use YOLO's batch inference natively later.
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
