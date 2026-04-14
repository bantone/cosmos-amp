"""
GPU Availability Check
======================
Run this script as a CML Job (or directly) before launching the Cosmos app
to verify that the environment has the GPU resources required for Cosmos Reason2-8B.

Cosmos Reason2-8B requirements (float16):
  - Minimum ~18 GB VRAM for inference
  - Recommended: A100 40 GB / A100 80 GB / H100

Exit codes:
  0 — at least one compatible GPU found
  1 — no CUDA devices, or insufficient VRAM
"""

import sys

SEP    = "─" * 62
BANNER = "🪐  Cosmos Reason2-8B — GPU Availability Check"

# Minimum VRAM in bytes for float16 inference (~18 GB)
MIN_VRAM_BYTES = 18 * (1024 ** 3)


def hr() -> None:
    print(SEP)


def section(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def check_python() -> None:
    section("Python environment")
    import platform
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Platform: {platform.platform()}")


def check_torch() -> None:
    section("PyTorch")
    try:
        import torch
        print(f"  torch version : {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version  : {torch.version.cuda}")
            print(f"  cuDNN version : {torch.backends.cudnn.version()}")
    except ImportError:
        print("  ✗  torch is not installed")
        print("     Run: pip install torch")


def check_gpus() -> bool:
    """Return True if at least one GPU meets the minimum VRAM requirement."""
    section("GPU devices")
    try:
        import torch
    except ImportError:
        print("  ✗  torch not available — cannot inspect GPUs")
        return False

    if not torch.cuda.is_available():
        print("  ✗  No CUDA-capable GPU detected.")
        print()
        print("  Possible reasons:")
        print("    • This machine has no NVIDIA GPU")
        print("    • CUDA drivers are not installed")
        print("    • The container/session was started without GPU resources")
        print()
        print("  In Cloudera AI Workbench, select a GPU-enabled resource profile")
        print("  when creating your Session or Model deployment.")
        return False

    n_gpus     = torch.cuda.device_count()
    compatible = []

    print(f"  {n_gpus} GPU(s) detected:\n")
    for i in range(n_gpus):
        props     = torch.cuda.get_device_properties(i)
        vram_gb   = props.total_memory / (1024 ** 3)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        free_gb   = (props.total_memory - torch.cuda.memory_reserved(i)) / (1024 ** 3)
        meets     = props.total_memory >= MIN_VRAM_BYTES

        status = "✓  compatible" if meets else "⚠  low VRAM"
        print(f"  [{i}] {props.name}")
        print(f"       Total VRAM : {vram_gb:.1f} GB   ({status})")
        print(f"       Free VRAM  : {free_gb:.1f} GB")
        print(f"       Allocated  : {allocated:.2f} GB")
        print(f"       Compute cap: {props.major}.{props.minor}")
        print()

        if meets:
            compatible.append(i)

    if compatible:
        print(f"  ✓  {len(compatible)} compatible GPU(s): indices {compatible}")
    else:
        print(f"  ✗  No GPU meets the {MIN_VRAM_BYTES/(1024**3):.0f} GB VRAM minimum.")
        print("     Consider using a larger GPU profile or enabling quantization.")

    return bool(compatible)


def check_transformers() -> None:
    section("Transformers / model dependencies")
    packages = {
        "transformers":   "nvidia/Cosmos-Reason2-8B uses Qwen3VLForConditionalGeneration",
        "accelerate":     "required for device_map='auto'",
        "huggingface_hub":"used for local cache detection",
        "cv2":            "video frame decoding",
        "PIL":            "image processing (Pillow)",
        "streamlit":      "web application framework",
    }
    for pkg, note in packages.items():
        real = "cv2" if pkg == "cv2" else pkg
        try:
            mod = __import__(real)
            ver = getattr(mod, "__version__", "unknown")
            print(f"  ✓  {pkg:<18} {ver:<12}  — {note}")
        except ImportError:
            print(f"  ✗  {pkg:<18} NOT INSTALLED  — {note}")


def check_model_cache() -> None:
    section("Model cache")
    try:
        from huggingface_hub import try_to_load_from_cache, scan_cache_dir
        import os

        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        print(f"  Cache directory: {cache_dir}")

        cached = try_to_load_from_cache("nvidia/Cosmos-Reason2-8B", "config.json")
        if cached:
            print("  ✓  Model weights found in local cache — no download needed")
        else:
            print("  ⚠  Model NOT in local cache")
            print("     First run will download ~16 GB from HuggingFace Hub")
            print("     Run scripts/download_model.py as a Job to pre-fetch weights")

    except Exception as exc:
        print(f"  Could not inspect cache: {exc}")


def main() -> None:
    print(f"\n{BANNER}")
    hr()

    check_python()
    check_torch()
    gpu_ok = check_gpus()
    check_transformers()
    check_model_cache()

    section("Summary")
    if gpu_ok:
        print("  ✓  Environment looks good — ready to run Cosmos Reason2-8B")
        print()
        print("  Next steps:")
        print("    1. (Optional) Pre-download weights: python scripts/download_model.py")
        print("    2. Launch the app              : streamlit run cosmos_app.py")
        print()
        sys.exit(0)
    else:
        print("  ✗  GPU requirements not met — Cosmos Reason2-8B will not run")
        print()
        print("  In Cloudera AI Workbench:")
        print("    • Edit your Session / Application resource profile")
        print("    • Select a GPU-enabled runtime with ≥ 18 GB VRAM")
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
