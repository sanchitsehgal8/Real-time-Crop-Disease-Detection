import os
import shutil
import random

source_dir = "data"
output_dir = "dataset"

train_ratio = 0.70
val_ratio = 0.15
test_ratio = 0.15

random.seed(42)

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(output_dir, split), exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)

    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    total = len(images)

    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    splits = {
        "train": train_images,
        "val": val_images,
        "test": test_images
    }

    for split_name in splits:
        split_class_dir = os.path.join(output_dir, split_name, class_name)
        os.makedirs(split_class_dir, exist_ok=True)

        for image in splits[split_name]:
            src = os.path.join(class_path, image)
            new_name = f"{split_name}_{image}"
            dst = os.path.join(split_class_dir, new_name)
            shutil.copy(src, dst)

    print(f"Processed class: {class_name}")

print("Dataset split completed successfully.")