from ultralytics import YOLO
import os
from pathlib import Path
from device_utils import resolve_device


def check_dataset_labels(data_yaml):
    """Quick pre-flight: warn if val set has no annotated labels."""
    import yaml
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    base = Path(cfg.get("path", "."))
    issues = []
    for subset in ["train", "val"]:
        lbl_dir = base / "labels" / subset
        if not lbl_dir.exists():
            issues.append(f"  [ERROR] labels/{subset}/ not found at {lbl_dir}")
            continue
        non_empty = [p for p in lbl_dir.glob("*.txt") if p.stat().st_size > 0]
        if len(non_empty) == 0:
            issues.append(
                f"  [ERROR] labels/{subset}/ has no annotated files — "
                f"YOLO will fail with 'no labels found'. "
                f"Make sure you have labeled images in {subset}."
            )
    if issues:
        print("\nDataset pre-flight FAILED:")
        for i in issues:
            print(i)
        print("\nRun: python setup_dataset.py --verify   for a full report.\n")
        return False
    return True

def train_model(data_yaml="datasets/fish/data.yaml", epochs=50, imgsz=640, model_type="yolov8n.pt", device="auto"):
    """Trains a YOLOv8 model."""
    if not os.path.exists(data_yaml):
        print(f"Data YAML file not found at {data_yaml}. Have you run 'setup_dataset.py' and split your data?")
        return

    if not check_dataset_labels(data_yaml):
        return

    # Load a model
    model = YOLO(model_type)  # load a pretrained model (recommended for training)

    resolved_device = resolve_device(device)
    print(f"Using device: {resolved_device}")

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        plots=True,
        device=resolved_device,
    )

    print(f"Training complete. Results saved in {results.save_dir}")
    print(f"Best model saved at {results.save_dir}/weights/best.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train a YOLO model for fish detection.")
    parser.add_argument("--data", default="datasets/fish/data.yaml", help="Path to data.yaml.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model to use.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, mps, or cpu.")

    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        model_type=args.model,
        device=args.device,
    )
