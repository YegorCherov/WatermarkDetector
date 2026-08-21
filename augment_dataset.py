import albumentations as A
import cv2
import os
import glob
import shutil

# Example usage: python augment_dataset.py --input_dir data/images --output_dir data/augmented

def create_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
        A.GaussianBlur(p=0.1),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5)
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def augment_dataset(input_img_dir, input_label_dir, output_img_dir, output_label_dir, augmentations_per_image=3):
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)

    transform = create_augmentation_pipeline()

    image_paths = glob.glob(os.path.join(input_img_dir, '*.jpg')) + glob.glob(os.path.join(input_img_dir, '*.png'))

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        base_name, ext = os.path.splitext(filename)
        label_path = os.path.join(input_label_dir, f"{base_name}.txt")

        # Copy original
        shutil.copy(img_path, os.path.join(output_img_dir, filename))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(output_label_dir, f"{base_name}.txt"))
        else:
            open(os.path.join(output_label_dir, f"{base_name}.txt"), 'a').close()

        img = cv2.imread(img_path)
        if img is None:
            continue

        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        bboxes.append([x, y, w, h])
                        class_labels.append(cls_id)

        for i in range(augmentations_per_image):
            try:
                transformed = transform(image=img, bboxes=bboxes, class_labels=class_labels)
                aug_img = transformed['image']
                aug_bboxes = transformed['bboxes']
                aug_labels = transformed['class_labels']

                aug_base_name = f"{base_name}_aug_{i}"
                cv2.imwrite(os.path.join(output_img_dir, f"{aug_base_name}{ext}"), aug_img)

                with open(os.path.join(output_label_dir, f"{aug_base_name}.txt"), 'w') as f:
                    for bbox, cls_id in zip(aug_bboxes, aug_labels):
                        x, y, w, h = bbox
                        f.write(f"{cls_id} {x} {y} {w} {h}\n")
            except Exception as e:
                print(f"Failed to augment {filename}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Augment YOLO dataset to improve generalization.")
    parser.add_argument('--input_img_dir', default='data/images', help='Directory with input images')
    parser.add_argument('--input_label_dir', default='data/labels', help='Directory with YOLO labels')
    parser.add_argument('--output_img_dir', default='augmented/images', help='Directory for augmented images')
    parser.add_argument('--output_label_dir', default='augmented/labels', help='Directory for augmented labels')
    parser.add_argument('--aug_count', type=int, default=3, help='Number of augmentations per image')

    args = parser.parse_args()
    print("Starting augmentation process...")
    augment_dataset(args.input_img_dir, args.input_label_dir, args.output_img_dir, args.output_label_dir, args.aug_count)
    print("Done!")
