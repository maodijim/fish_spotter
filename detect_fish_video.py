import cv2
from ultralytics import YOLO
import argparse
import platform
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from device_utils import resolve_device


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
WINDOW_NAME = "Fish Detector"
WINDOW_SIZE = (1280, 720)

# Minimum seconds between alert sounds to avoid rapid-fire beeping.
ALERT_COOLDOWN_SECONDS = 3.0
_last_alert_time = 0.0
_alert_lock = threading.Lock()
_window_prepared = False

# Detection counter state
_session_detections = 0
_counter_lock = threading.Lock()

# Cooldown for auto-incrementing the detection count (prevents same fish counting many times)
DETECTION_COUNT_COOLDOWN_SECONDS = 3.0
_last_count_time = 0.0


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


def prepare_display_window():
    """Create a resizable OpenCV window for preview output."""
    global _window_prepared
    if _window_prepared:
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *WINDOW_SIZE)
    _window_prepared = True

def get_font():
    """Get a font that supports CJK characters, with fallback options."""
    sys_platform = platform.system()

    font_candidates = []

    if sys_platform == "Darwin":  # macOS
        font_candidates = [
            "/Library/Fonts/Arial Unicode.ttf",          # best CJK coverage on macOS
            "/System/Library/Fonts/PingFang.ttc",
            "/Library/Fonts/SimSun.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/STHeiti Medium.ttc",
        ]
    elif sys_platform == "Windows":
        font_candidates = [
            "C:\\Windows\\Fonts\\msyh.ttf",
            "C:\\Windows\\Fonts\\SimHei.ttf",
            "C:\\Windows\\Fonts\\SimSun.ttc",
            "C:\\Windows\\Fonts\\arial.ttf",
        ]
    elif sys_platform == "Linux":
        font_candidates = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]

    # Try each candidate in order
    for font_path in font_candidates:
        try:
            if Path(font_path).exists():
                return ImageFont.truetype(font_path, 28)
        except Exception:
            continue

    # Fallback to PIL default (will not render CJK correctly)
    try:
        return ImageFont.load_default()
    except Exception:
        return None


def draw_counter_overlay(frame, frame_detections, show_counter=True):
    """Draw detection counter as text overlay on frame with Unicode support.

    Args:
        frame: Video frame to draw on (BGR numpy array)
        frame_detections: Number of detections in current frame
        show_counter: True/False for display, or int 1-10 for style/position
    """

    # Convert True to 1, False/None to 0
    if show_counter is True:
        show_counter = 1
    elif show_counter is False or show_counter is None:
        show_counter = 0

    if not show_counter:
        return frame

    h, w = frame.shape[:2]

    # Position based on counter value (1-5 cycles through corners and edges)
    position_map = {
        1: (10, 10),      # top-left
        2: (w-350, 10),   # top-right
        3: (10, h-80),    # bottom-left
        4: (w-350, h-80), # bottom-right
        5: (w//2-175, 10),# top-center
    }

    x, y = position_map.get(show_counter % 5 if show_counter > 0 else 1, (10, 10))

    # Semi-transparent background rectangle
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x+340, y+70), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Counter text with Unicode support using PIL
    with _counter_lock:
        total = _session_detections

    text_content = f"今天总共捉鱼: {total}"

    # Convert BGR frame to RGB numpy array for PIL
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(np.uint8(frame_rgb))
    draw = ImageDraw.Draw(pil_image)

    # Get a font that supports Chinese characters
    font = get_font()

    # Draw text on PIL image with proper encoding
    text_color = (0, 255, 0)  # Green in RGB
    if font is not None:
        draw.text((x+20, y+20), text_content, font=font, fill=text_color)
    else:
        # Fallback: draw text without font (will use default)
        draw.text((x+20, y+20), text_content, fill=text_color)

    # Convert back to BGR numpy array for OpenCV
    frame_rgb_array = np.array(pil_image)
    frame_bgr = cv2.cvtColor(frame_rgb_array, cv2.COLOR_RGB2BGR)
    return frame_bgr


def run_inference(
    model_path,
    source,
    conf=0.35,
    show=True,
    save=False,
    device="auto",
    show_counter=True,
    initial_count=0,
):
    """
    Runs inference on a video source.

    Args:
        model_path (str): Path to the trained model (.pt).
        source (str): Video file path, stream URL, or camera index.
        conf (float): Confidence threshold.
        show (bool): Whether to display the video stream.
        save (bool): Whether to save the output video.
        show_counter (bool): Whether to display detection counter overlay.
        initial_count (int): Initial value for session detection counter.
    """
    # Load the model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    resolved_device = resolve_device(device)
    print(f"Using device: {resolved_device}")

    if show:
        prepare_display_window()

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
                cv2.imshow(WINDOW_NAME, frame)
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
    print("Press 'q' to quit. Press '+'/'-' to manually adjust fish count.")
    writer = None
    output_path = None
    saved_frames = 0
    global _session_detections, _window_prepared
    # Reset session counter and cooldown timer
    _session_detections = max(0, int(initial_count))
    global _last_count_time
    _last_count_time = 0.0

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

                # Increment session counter with cooldown to avoid counting the same fish repeatedly.
                now = time.monotonic()
                with _counter_lock:
                    if now - _last_count_time >= DETECTION_COUNT_COOLDOWN_SECONDS:
                        _session_detections += detections
                        _last_count_time = now
                        print(f"Fish counted: +{detections} → total {_session_detections} (next count in {DETECTION_COUNT_COOLDOWN_SECONDS}s)")

            # Draw counter overlay on frame
            frame = draw_counter_overlay(frame, detections, show_counter)

            if show:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key in (ord("+"), ord("=")):  # '+' or '=' (no-shift)
                    with _counter_lock:
                        _session_detections += 1
                    print(f"Manual adjust: +1 → {_session_detections}")
                elif key == ord("-"):
                    with _counter_lock:
                        _session_detections = max(0, _session_detections - 1)
                    print(f"Manual adjust: -1 → {_session_detections}")
    finally:
        global _window_prepared
        _window_prepared = False
        if writer is not None:
            writer.release()
            print(f"Saved {saved_frames} detection frame(s) to: {output_path}")
        elif save:
            print("No detections found in stream; no video file was saved.")
        with _counter_lock:
            print(f"Stream session complete. Total detections: {_session_detections}")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fish detection on a video stream.")
    parser.add_argument("--model", default="runs/detect/train/weights/best.pt", help="Path to best.pt.")
    parser.add_argument("--source", default="0", help="Source input (webcam index, video path/URL, or image file path).")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    parser.add_argument("--noshow", action="store_true", help="Don't display the video window.")
    parser.add_argument("--save", action="store_true", help="Save the results to a file.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cuda, mps, or cpu.")
    parser.add_argument("--counter", nargs='?', const=1, type=int, default=None,
                        help="Show detection counter overlay on stream (0=off, 1-10 for styles/positions).")
    parser.add_argument("--initial-count", type=int, default=0,
                        help="Initial fish count shown by the counter (default: 0).")

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        show=not args.noshow,
        save=args.save,
        device=args.device,
        show_counter=args.counter,
        initial_count=args.initial_count,
    )
