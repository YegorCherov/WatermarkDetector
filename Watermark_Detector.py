from ultralytics import YOLO

# Note: For YOLOv8 training, your dataset needs to be in YOLO format.
# This means you should have a data.yaml file defining paths to your
# train and val image directories, and the number of classes and class names.
# Additionally, each image should have a corresponding .txt file with the
# same name containing bounding box coordinates and class labels for each object.
# This supports localization, multi-scale analysis, and identifying different
# types of watermarks.

# Example data.yaml structure:
# train: C:\Users\yegor\Downloads\water\wm-nowm\train\images
# val: C:\Users\yegor\Downloads\water\wm-nowm\valid\images
# nc: 1  # number of classes
# names: ['watermark']  # class names (can add more for different watermark types)

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)

    # Train the model
    # Update the 'data' argument to point to your data.yaml file
    results = model.train(
        data='data.yaml',
        epochs=10,
        imgsz=640,
        batch=16,
        project='watermark_detection',
        name='yolov8_watermark_model'
    )

    # Save the model is handled automatically by Ultralytics YOLO,
    # it will be saved in watermark_detection/yolov8_watermark_model/weights/best.pt
