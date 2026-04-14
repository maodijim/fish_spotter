import os
import shutil
import random
import yaml
from pathlib import Path

def setup_yolo_structure(base_dir="datasets/fish"):
    """Creates the YOLOv8 dataset directory structure."""
    dirs = [
        f"{base_dir}/images/train",
        f"{base_dir}/images/val",
        f"{base_dir}/labels/train",
        f"{base_dir}/labels/val",
        f"{base_dir}/raw_images"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print(f"Created YOLO structure in {base_dir}")

def copy_images_to_raw(src_dir="fish_photos", dest_dir="datasets/fish/raw_images"):
    """Copies all images from fish_photos to a raw_images directory for labeling."""
    if not os.path.exists(src_dir):
        print(f"Source directory {src_dir} does not exist.")
        return

    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for img in images:
        shutil.copy(os.path.join(src_dir, img), os.path.join(dest_dir, img))
    print(f"Copied {len(images)} images to {dest_dir} for labeling.")

def create_data_yaml(base_dir="datasets/fish", classes=["fish"]):
    """Creates the data.yaml file required by YOLOv8."""
    data = {
        'path': os.path.abspath(base_dir),
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)}
    }
    with open(os.path.join(base_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"Created data.yaml in {base_dir}")

def split_dataset(base_dir="datasets/fish", train_ratio=0.8):
    """Splits the labeled dataset into train and val sets."""
    raw_images_dir = Path(base_dir) / "raw_images"
    # Labels should be exported from label-studio to this temporary dir first
    exported_labels_dir = Path(base_dir) / "exported_labels/labels"

    if not exported_labels_dir.exists():
        print(f"Please export your labels from Label Studio to {exported_labels_dir} in YOLO format first.")
        return

    images = [f for f in raw_images_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    random.shuffle(images)

    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def resolve_label_path(img_stem):
        # Support both "<stem>.txt" and "<random>-<stem>.txt" exports.
        exact = exported_labels_dir / f"{img_stem}.txt"
        if exact.exists():
            return exact

        candidates = sorted(exported_labels_dir.glob(f"*-{img_stem}.txt"))
        if candidates:
            return candidates[0]

        return None

    def move_files(file_list, subset):
        negatives = 0
        skipped = 0
        for img_path in file_list:
            label_path = resolve_label_path(img_path.stem)

            if label_path:
                # Move image
                shutil.copy(img_path, Path(base_dir) / "images" / subset / img_path.name)
                # Normalize label filename so it always matches the image stem for YOLO training.
                target_label = Path(base_dir) / "labels" / subset / f"{img_path.stem}.txt"
                shutil.copy(label_path, target_label)
            else:
                if subset == "train":
                    skipped += 1
                    print(f"No label found for {img_path.name}; skipped from training.")
                else:
                    # Keep unlabeled validation images as no-fish negatives.
                    shutil.copy(img_path, Path(base_dir) / "images" / subset / img_path.name)
                    target_label = Path(base_dir) / "labels" / subset / f"{img_path.stem}.txt"
                    target_label.write_text("", encoding="utf-8")
                    negatives += 1
                    print(f"No label found for {img_path.name}; added as no-fish negative in val.")

        return negatives, skipped

    train_negatives, train_skipped = move_files(train_images, "train")
    val_negatives, val_skipped = move_files(val_images, "val")
    print(f"Dataset split: {len(train_images)} train, {len(val_images)} val.")
    print(f"No-fish negatives: {train_negatives} train, {val_negatives} val.")
    print(f"Missing labels skipped: {train_skipped} train, {val_skipped} val.")

def verify_dataset(base_dir="datasets/fish"):
    """Checks the dataset structure and reports label/image counts and potential issues."""
    base = Path(base_dir)
    ok = True

    print("\n=== Dataset Verification ===")

    # Check data.yaml
    yaml_path = base / "data.yaml"
    if not yaml_path.exists():
        print(f"[ERROR] data.yaml not found at {yaml_path}")
        ok = False
    else:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        print(f"[OK]    data.yaml found: path={cfg.get('path')}, classes={cfg.get('names')}")

    for subset in ["train", "val"]:
        img_dir = base / "images" / subset
        lbl_dir = base / "labels" / subset

        images = sorted(img_dir.glob("*")) if img_dir.exists() else []
        labels = sorted(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        non_empty = [l for l in labels if l.stat().st_size > 0]
        empty = [l for l in labels if l.stat().st_size == 0]

        print(f"\n  [{subset}]")
        print(f"    Images : {len(images)}")
        print(f"    Labels : {len(labels)} total | {len(non_empty)} with annotations | {len(empty)} empty")

        if len(images) == 0:
            print(f"    [WARNING] No images found in {img_dir}")
            ok = False

        if len(non_empty) == 0:
            print(f"    [ERROR] No annotated labels found — YOLO cannot train/validate without at least some positive boxes.")
            ok = False
        elif len(non_empty) < 5:
            print(f"    [WARNING] Very few annotated images ({len(non_empty)}). Consider labeling more.")

        # Check every image has a matching label file
        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}
        missing_labels = img_stems - lbl_stems
        if missing_labels:
            print(f"    [WARNING] {len(missing_labels)} image(s) have no label file: {sorted(missing_labels)[:5]} ...")

    print(f"\n{'[PASS] Dataset looks ready for training.' if ok else '[FAIL] Fix the issues above before training.'}\n")
    return ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Setup dataset for YOLO training.")
    parser.add_argument("--setup", action="store_true", help="Setup folder structure and copy raw images.")
    parser.add_argument("--split", action="store_true", help="Split labeled images into train/val.")
    parser.add_argument("--verify", action="store_true", help="Verify dataset structure and label counts.")
    parser.add_argument("--classes", nargs="+", default=["Airplane", "Car", "fish"],
                        help="Class names matching your Label Studio labels exactly (case-sensitive, same order). "
                             "Label Studio defaults to 'Airplane Car' — override this to match your project. "
                             "Example: --classes fish")

    args = parser.parse_args()

    if args.setup:
        setup_yolo_structure()
        copy_images_to_raw()
        create_data_yaml(classes=args.classes)
        print("\nNext Steps:")
        print("1. Run 'label-studio start' to begin labeling.")
        print("2. Import images from 'datasets/fish/raw_images'.")
        print("3. Export labels in 'YOLO' format to 'datasets/fish/exported_labels'.")
        print("4. Run this script with --split to prepare for training.")

    if args.split:
        split_dataset()

    if args.verify:
        verify_dataset()

