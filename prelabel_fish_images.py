import argparse
from pathlib import Path

from ultralytics import YOLO
from device_utils import resolve_device


VALID_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_images(input_dir: Path):
    return sorted(
        [
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_SUFFIXES
        ]
    )


def format_yolo_line(class_id: int, x_center: float, y_center: float, width: float, height: float):
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def run_prelabeling(model_path: str, input_dir: str, output_dir: str, conf: float, overwrite: bool, class_id: int, device: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    images = collect_images(input_path)
    if not images:
        print(f"No images found in {input_path}.")
        return

    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    resolved_device = resolve_device(device)
    print(f"Using device: {resolved_device}")

    created = 0
    skipped = 0
    empty = 0

    for image_path in images:
        label_path = output_path / f"{image_path.stem}.txt"
        if label_path.exists() and not overwrite:
            skipped += 1
            continue

        try:
            results = model.predict(
                source=str(image_path),
                conf=conf,
                verbose=False,
                device=resolved_device,
            )
        except Exception as exc:
            print(f"Failed inference for {image_path.name}: {exc}")
            continue

        lines = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xywhn = boxes.xywhn.cpu().tolist()
                for x_center, y_center, width, height in xywhn:
                    lines.append(format_yolo_line(class_id, x_center, y_center, width, height))

        if not lines:
            empty += 1

        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        created += 1

    print("\nPre-labeling complete")
    print(f"Images scanned : {len(images)}")
    print(f"Labels written : {created}")
    print(f"Existing skipped: {skipped}")
    print(f"Empty labels   : {empty}")
    print(f"Output folder  : {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pre-label fish images with YOLO and write YOLO txt files.")
    parser.add_argument("--model", default="yolov8n.pt", help="Model path or name for Ultralytics YOLO.")
    parser.add_argument("--input-dir", default="datasets/fish/raw_images", help="Folder with unlabeled images.")
    parser.add_argument("--output-dir", default="datasets/fish/exported_labels", help="Folder to write YOLO txt labels.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt labels.")
    parser.add_argument("--class-id", type=int, default=0, help="Class id to write for every detection (fish=0).")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, mps, or cpu.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_prelabeling(
        model_path=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        conf=args.conf,
        overwrite=args.overwrite,
        class_id=args.class_id,
        device=args.device,
    )
