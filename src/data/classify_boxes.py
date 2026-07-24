#This script automatically addes the class notation for each bounding box to the image label.
#Note: as a prerequisite to adding the classes to the labels, the crops of the images must first have been classified from (0-5) using label_crops.py
from pathlib import Path
import re
import shutil

labels_dir = Path("../../data/custom/labels")
images_dir = Path("../../data/custom/images")
crops_dir = Path("../../data/custom/crops")

excluded_labels_dir = Path("../../data/custom/excluded/labels")
excluded_images_dir = Path("../../data/custom/excluded/images")

excluded_labels_dir.mkdir(parents=True, exist_ok=True)
excluded_images_dir.mkdir(parents=True, exist_ok=True)

mismatches = []


for label_path in sorted(labels_dir.glob("*.txt")):
    if label_path.stat().st_size == 0:
        continue

    frame_id = label_path.stem
    frame_num = frame_id[-5:]

    matched_crops = []
    for digit in range(6):
        digit_dir = crops_dir / str(digit)
        for crop_path in digit_dir.glob(f"crop_{frame_num}-*.jpg"):
            suffix = int(re.search(r'-(\d+)\.jpg$', crop_path.name).group(1))
            matched_crops.append((suffix, digit, crop_path))

    matched_crops.sort(key=lambda x: x[0])

    with open(label_path, "r") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    if len(matched_crops) != len(lines):
        mismatches.append(frame_id)
        shutil.move(str(label_path), str(excluded_labels_dir / label_path.name))
        image_path = images_dir / f"{frame_id}.jpg"
        if image_path.exists():
            shutil.move(str(image_path), str(excluded_images_dir / image_path.name))
        continue

    with open(label_path, "w") as f:
        for (suffix, digit, crop_path), line in zip(matched_crops, lines):
            parts = line.split()
            f.write(f"{digit} {parts[1]} {parts[2]} {parts[3]} {parts[4]}\n")

print(f"{len(mismatches)} mismatches moved to excluded directory")
