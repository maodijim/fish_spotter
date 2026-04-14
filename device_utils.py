import platform


def resolve_device(device="auto"):
    """Resolve runtime device for Ultralytics (auto, cuda, mps, cpu)."""
    normalized = (device or "auto").strip().lower()
    if normalized not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device '{device}'. Use: auto, cuda, mps, cpu.")

    if normalized == "cpu":
        return "cpu"

    try:
        import torch
    except Exception:
        # If torch import fails, safest fallback is CPU.
        return "cpu"

    cuda_available = bool(torch.cuda.is_available())
    mps_available = bool(torch.backends.mps.is_available())

    if normalized == "cuda":
        if cuda_available:
            return "cuda"
        print("CUDA is not available on this machine; falling back to CPU.")
        return "cpu"

    if normalized == "mps":
        if mps_available:
            return "mps"
        print("MPS is not available on this machine; falling back to CPU.")
        return "cpu"

    # auto mode: prefer CUDA, then MPS on macOS, else CPU.
    if cuda_available:
        return "cuda"

    if platform.system() == "Darwin" and mps_available:
        return "mps"

    return "cpu"

