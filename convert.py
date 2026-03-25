import os
import cv2

def convert(ann_dir, img_dir, label_dir):
    os.makedirs(label_dir, exist_ok=True)

    for ann_file in os.listdir(ann_dir):
        img_file = ann_file.replace(".txt", ".jpg")

        img_path = os.path.join(img_dir, img_file)
        ann_path = os.path.join(ann_dir, ann_file)

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w, _ = img.shape

        with open(ann_path, "r") as f:
            lines = f.readlines()

        yolo_lines = []

        for line in lines:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue

            x, y, bw, bh = map(float, parts[:4])
            cls = int(parts[5])

            if cls == 0:
                continue

            # Normalize
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw /= w
            bh /= h

            yolo_lines.append(f"{cls-1} {x_center} {y_center} {bw} {bh}")

        with open(os.path.join(label_dir, ann_file), "w") as f:
            f.write("\n".join(yolo_lines))


# Run for train
convert(
    "VisDrone2019-DET-train/annotations",
    "VisDrone2019-DET-train/images",
    "dataset/labels/train"
)

# Run for val
convert(
    "VisDrone2019-DET-val/annotations",
    "VisDrone2019-DET-val/images",
    "dataset/labels/val"
)

print("Conversion Done ✅")