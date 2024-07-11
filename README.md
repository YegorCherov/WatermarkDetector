# Watermark Detector

![Demo GIF](/Water-Gif.gif)

## Table of Contents
1. [Introduction](#introduction)
2. [Project Components](#project-components)
3. [How It Works](#how-it-works)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Model Training](#model-training)
7. [Performance](#performance)
8. [Future Improvements](#future-improvements)

## Introduction

This project is a Watermark Detector that uses deep learning to identify watermarks in images and videos. It employs a Convolutional Neural Network (CNN) based on the VGG16 architecture to classify images as either watermarked or not watermarked. The system can process individual images, entire folders of images, and even video files.

## Project Components

The project consists of several Python scripts, each serving a specific purpose:

1. `Watermark_Detector.py`: The main script for training the watermark detection model.
2. `Detector.py`: A multi-threaded image detector using the trained model.
3. `DetectorMulti.py`: An optimized version of the detector using batch processing.
4. `DetectorSingle.py`: A single-threaded version of the detector.
5. `DetectorSingleFolder.py`: Processes and organizes images from a folder.
6. `DetectorSingleGpu.py`: GPU-accelerated version of the detector.
7. `DetectorSingleGpuFolder.py`: GPU-accelerated folder processing.
8. `DetectorVideoSingle.py`: Detects watermarks in a single video file.
9. `DetectorVideoSingleFolder.py`: Processes multiple videos in a folder.

## How It Works

### Model Training
The system uses a pre-trained VGG16 model as a base, fine-tuned on a dataset of watermarked and non-watermarked images.

### Image Processing
- Images are resized to 224x224 pixels and normalized.
- The trained model predicts the likelihood of a watermark being present.

### Video Processing
- Videos are processed frame by frame at specified intervals.
- If any frame is detected to have a watermark, the entire video is classified as watermarked.

### Batch Processing and GPU Acceleration
- For large datasets, batch processing is implemented to improve efficiency.
- GPU acceleration is utilized when available to speed up predictions.

### Organization
- Processed images and videos are sorted into 'Watermarked' and 'Not_Watermarked' folders.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/YegorCherov/watermark-detector.git
   ```

2. Install the required dependencies:
   ```
   pip install tensorflow opencv-python numpy pillow
   ```

3. Download the pre-trained model file `watermark_detection_model_V2_60000_Data_Set.h5` and place it in the project directory.

## Usage

### For Image Detection:

```python
python DetectorSingleGpuFolder.py
```

### For Video Detection:

```python
python DetectorVideoSingleFolder.py
```

Ensure to set the correct paths for input and output directories in the respective scripts.

## Model Training

To train the model on your own dataset:

1. Organize your dataset into two folders: 'watermarked' and 'not_watermarked'.
2. Update the paths in `Watermark_Detector.py`.
3. Run the training script:

```python
python Watermark_Detector.py
```

## Performance

The model's performance can vary depending on the complexity and diversity of watermarks in your dataset. On the test set, the model achieved:

- Accuracy: ~90%
- Precision: ~85%
- Recall: ~89%

## Future Improvements

1. Implement more advanced architectures like ResNet or EfficientNet.
2. Develop a user-friendly GUI for easier interaction.
3. Optimize for mobile deployment.
4. Expand the dataset to improve generalization to various types of watermarks.
