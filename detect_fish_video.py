import cv2
from ultralytics import YOLO
import argparse
import platform
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from device_utils import resolve_device


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Minimum seconds between alert sounds to avoid rapid-fire beeping.
ALERT_COOLDOWN_SECONDS = 3.0
_last_alert_time = 0.0
_alert_lock = threading.Lock()


def play_alert_sound():
    """Play a short alert sound in a background thread (non-blocking)."""
    def _play():
        global _last_alert_time
        now = time.monotonic()
        with _alert_lock:
            if now - _last_alert_time < ALERT_COOLDOWN_SECONDS:
                return
            _last_alert_time = now

        system = platform.system()
        try:
            if system == "Darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Ping.aiff"],
                               check=False, capture_output=True)
            elif system == "Windows":
                import winsound
                winsound.Beep(1000, 200)
            else:
                # Linux / other: terminal bell
                print("\a", end="", flush=True)
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()


def is_image_file(source):
    source_path = Path(source)
    return source_path.exists() and source_path.is_file() and source_path.suffix.lower() in IMAGE_SUFFIXES

def run_inference(model_path, source, conf=0.35, show=True, save=False, device="auto"):
    """
    Runs inference on a video source.

    Args:
        model_path (str): Path to the trained model (.pt).
        source (str): Video file path, stream URL, or camera index.
        conf (float): Confidence threshold.
        show (bool): Whether to display the video stream.
        save (bool): Whether to save the output video.
    """
    # Load the model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    resolved_device = resolve_device(device)
    print(f"Using device: {resolved_device}")

    if is_image_file(source):
        print(f"Starting image inference on {source}...")
        results = model.predict(
            source=source,
            conf=conf,
            show=False,   # cv2 handles display manually below
            save=save,
            stream=False,
            device=resolved_device,
        )
        if results:
            frame = results[0].orig_img.copy()
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf_score = float(box.conf[0])
                    label = f"fish {conf_score:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if show:
                cv2.imshow("Fish Detector", frame)
                print("Press any key to close.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return

    # Run inference on webcam/video/stream sources.
    results = model.predict(
        source=source,
        conf=conf,
        stream=True,  # use generator for memory efficiency
        show=False,   # cv2 handles display manually below
        save=False,   # manual save below so we only write frames with detections
        device=resolved_device,
    )

    print(f"Starting stream inference on {source}...")
    print("Press 'q' to quit.")
    writer = None
    output_path = None
    saved_frames = 0

    try:
        for result in results:
            # Keep original clean frame before drawing boxes.
            original_frame = result.orig_img.copy()
            frame = original_frame.copy()
            detections = 0
            max_conf = 0.0
            if result.boxes is not None:
                detections = len(result.boxes)
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf_score = float(box.conf[0])
                    max_conf = max(max_conf, conf_score)
                    label = f"fish {conf_score:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            if save and detections > 0:
                if writer is None:
                    output_dir = Path("runs/detect/stream")
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter.fourcc(*"mp4v"), 20.0, (w, h))
                writer.write(frame)
                saved_frames += 1

                img_dir = Path("runs/detect/stream/images")
                img_dir.mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

                # Always save the annotated (boxed) frame.
                boxed_path = img_dir / f"detection_{ts}.jpg"
                cv2.imwrite(str(boxed_path), frame)

                if max_conf < 0.80:
                    # Low confidence: also save the clean original for manual review.
                    orig_path = img_dir / f"detection_{ts}_original.jpg"
                    cv2.imwrite(str(orig_path), original_frame)
                    print(f"Low-conf detection saved: {boxed_path.name} + original ({detections} fish, conf={max_conf:.2f})")
                else:
                    print(f"Detection saved: {boxed_path.name} ({detections} fish, conf={max_conf:.2f})")

            if detections > 0:
                play_alert_sound()

            if show:
                cv2.imshow("Fish Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        if writer is not None:
            writer.release()
            print(f"Saved {saved_frames} detection frame(s) to: {output_path}")
        elif save:
            print("No detections found in stream; no video file was saved.")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fish detection on a video stream.")
    parser.add_argument("--model", default="runs/detect/train/weights/best.pt", help="Path to best.pt.")
    parser.add_argument("--source", default="0", help="Source input (webcam index, video path/URL, or image file path).")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--noshow", action="store_true", help="Don't display the video window.")
    parser.add_argument("--save", action="store_true", help="Save the results to a file.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, mps, or cpu.")

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        show=not args.noshow,
        save=args.save,
        device=args.device,
    )
