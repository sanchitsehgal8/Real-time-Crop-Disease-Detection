import os
import random
import shutil

DATA_DIR = "data"
UNKNOWN_DIR = "unknown_backround"

SPLITS = ["train", "val", "test"]
IMG_EXT = (".jpg", ".jpeg", ".png")

# Step 1: count images already present in dataset splits
split_counts = {}

for split in SPLITS:
    split_path = os.path.join(DATA_DIR, split)
    count = 0

    for root, _, files in os.walk(split_path):
        count += len([f for f in files if f.lower().endswith(IMG_EXT)])

    split_counts[split] = count

total = sum(split_counts.values())

split_ratios = {
    split: split_counts[split] / total
    for split in SPLITS
}

print("Detected dataset ratios:", split_ratios)


# Step 2: load unknown_background images
images = [
    f for f in os.listdir(UNKNOWN_DIR)
    if f.lower().endswith(IMG_EXT)
]

random.shuffle(images)

n = len(images)

train_n = int(split_ratios["train"] * n)
val_n = int(split_ratios["val"] * n)

train_imgs = images[:train_n]
val_imgs = images[train_n:train_n + val_n]
test_imgs = images[train_n + val_n:]


split_map = {
    "train": train_imgs,
    "val": val_imgs,
    "test": test_imgs
}


# Step 3: copy images into dataset folders
for split, img_list in split_map.items():
    target_dir = os.path.join(DATA_DIR, split, "unknown_background")
    os.makedirs(target_dir, exist_ok=True)

    for img in img_list:
        src = os.path.join(UNKNOWN_DIR, img)
        dst = os.path.join(target_dir, img)
        shutil.copy(src, dst)

print("✅ unknown_background successfully added to train/val/test")