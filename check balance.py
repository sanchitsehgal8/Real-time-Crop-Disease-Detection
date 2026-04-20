import os

BASE_DIR = "data"

for split in ["train", "val", "test"]:
    print(f"\n📂 {split.upper()} SET\n")

    split_path = os.path.join(BASE_DIR, split)

    for class_name in sorted(os.listdir(split_path)):
        class_path = os.path.join(split_path, class_name)

        if os.path.isdir(class_path):
            count = len(os.listdir(class_path))
            print(f"{class_name}: {count}")