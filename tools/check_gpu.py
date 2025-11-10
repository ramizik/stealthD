"""
GPU Diagnostic Script for Soccer Analysis System
Run this script to verify your GPU setup before running the main analysis.
"""

import sys
from pathlib import Path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_DIR))

def main():
    print("=" * 70)
    print("GPU AVAILABILITY CHECK")
    print("=" * 70)

    # Check PyTorch installation
    try:
        import torch
        print(f"\n✓ PyTorch installed: {torch.__version__}")
    except ImportError:
        print("\n✗ PyTorch NOT installed!")
        print("  Install with: pip install torch torchvision torchaudio")
        sys.exit(1)

    # Check CUDA availability
    print(f"\nCUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ GPU Device Count: {torch.cuda.device_count()}")
        print(f"✓ Current GPU Device: {torch.cuda.current_device()}")
        print(f"✓ GPU Device Name: {torch.cuda.get_device_name(0)}")

        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        print(f"✓ GPU Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"✓ GPU Compute Capability: {props.major}.{props.minor}")

        # Test GPU memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, device='cuda')
            print(f"✓ GPU Memory Test: PASSED")
            print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ GPU Memory Test: FAILED - {e}")
    else:
        print("\n" + "=" * 70)
        print("⚠️ CUDA is NOT available!")
        print("=" * 70)
        print("Possible reasons:")
        print("1. PyTorch CPU-only version installed")
        print("2. CUDA drivers not installed")
        print("3. Incompatible CUDA/PyTorch versions")
        print("\nSolutions:")
        print("1. Check NVIDIA drivers:")
        print("   Run: nvidia-smi")
        print("\n2. Reinstall PyTorch with CUDA support:")
        print("   pip uninstall torch torchvision torchaudio")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("=" * 70)
        sys.exit(1)

    # Check YOLO model GPU usage
    print("\n" + "=" * 70)
    print("YOLO MODEL GPU CHECK")
    print("=" * 70)

    try:
        from ultralytics import YOLO
        print(f"\n✓ Ultralytics YOLO installed")
    except ImportError:
        print("\n✗ Ultralytics YOLO NOT installed!")
        print("  Install with: pip install ultralytics")
        sys.exit(1)

    # Check if model files exist
    from constants import model_path
    from keypoint_detection.keypoint_constants import keypoint_model_path

    print(f"\nChecking model files...")
    if model_path.exists():
        print(f"✓ Detection model found: {model_path.name}")
    else:
        print(f"✗ Detection model NOT found: {model_path}")

    if keypoint_model_path.exists():
        print(f"✓ Keypoint model found: {keypoint_model_path.name}")
    else:
        print(f"✗ Keypoint model NOT found: {keypoint_model_path}")

    # Test model loading on GPU
    if model_path.exists():
        print(f"\nTesting detection model GPU loading...")
        try:
            from player_detection import load_detection_model
            model = load_detection_model(str(model_path))
            print(f"✓ Model device: {model.device}")

            # Test inference
            import numpy as np
            test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            print(f"  Running test inference...")
            results = model(test_frame, verbose=False)
            print(f"✓ Test inference completed successfully on GPU")

        except Exception as e:
            print(f"✗ Model loading/inference failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if torch.cuda.is_available():
        print("✓ GPU is available and ready for use")
        print("✓ You can run main.py to start soccer analysis")
        print("\nGPU will be used for:")
        print("  - Player detection (YOLO)")
        print("  - Keypoint detection (YOLO Pose)")
        print("  - Player tracking")
        print("  - Team clustering")
    else:
        print("✗ GPU is NOT available")
        print("✗ Fix GPU issues before running main.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
