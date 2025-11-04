"""
YOLOv8 Pose Training Script for Football Pitch Keypoint Detection
Trains a model to detect 32 keypoints on a football pitch.
Dataset should be in the same folder as this script.
Trained models will be saved to Models/Trained folder in repo root.
"""

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def update_data_yaml(dataset_path: str):
    """
    Update the data.yaml file with correct paths for training.

    Args:
        dataset_path: Path to the dataset directory containing data.yaml
    """
    data_yaml_path = Path(dataset_path) / "data.yaml"

    if not data_yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at {data_yaml_path}")

    # Read existing data.yaml
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Update paths to be relative to data.yaml location
    data['train'] = '../train/images'
    data['val'] = '../valid/images'
    data['test'] = '../test/images'

    # Write updated data.yaml
    with open(data_yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Updated {data_yaml_path} with correct paths")


def train_pitch_detection(
    data_yaml: str,
    model_size: str = 'n',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = '0',
    name: str = 'pitch_detection',
    pretrained: bool = True
):
    """
    Train YOLOv8-Pose model for pitch keypoint detection.

    Args:
        data_yaml: Path to data.yaml file
        model_size: YOLOv8 model size (n, s, m, l, x)
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
        device: Device to train on ('0' for GPU, 'cpu' for CPU)
        name: Name of the training run
        pretrained: Whether to use pretrained weights
    """
    # Get the repo root (parent of tools folder)
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    project_dir = repo_root / "Models" / "Trained"

    # Ensure the output directory exists
    project_dir.mkdir(parents=True, exist_ok=True)

    # Initialize YOLOv8-Pose model
    model_name = f'yolov8{model_size}-pose.pt' if pretrained else f'yolov8{model_size}-pose.yaml'
    model = YOLO(model_name)

    print(f"\n{'='*60}")
    print(f"Training Configuration:")
    print(f"{'='*60}")
    print(f"Model: yolov8{model_size}-pose")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {imgsz}")
    print(f"Batch Size: {batch}")
    print(f"Device: {device}")
    print(f"Pretrained: {pretrained}")
    print(f"Output: {project_dir}/{name}")
    print(f"{'='*60}\n")

    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project_dir),
        name=name,
        patience=50,  # Early stopping patience
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        cache=False,  # Cache images for faster training (set to True if you have enough RAM)
        verbose=True,
        plots=True,  # Generate training plots
        mosaic=0.0,  # Disable mosaic augmentation
        exist_ok=True,
        pretrained=pretrained
    )

    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best model saved at: {project_dir}/{name}/weights/best.pt")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8-Pose for pitch detection')
    parser.add_argument('--data', type=str, required=True, help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size (n=nano, s=small, m=medium, l=large, x=xlarge)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--device', type=str, default='0', help='Device to train on (0 for GPU, cpu for CPU)')
    parser.add_argument('--name', type=str, default='pitch_detection', help='Run name')
    parser.add_argument('--no-pretrained', action='store_true', help='Train from scratch without pretrained weights')
    parser.add_argument('--update-yaml', action='store_true', help='Update data.yaml paths before training')

    args = parser.parse_args()

    # Update data.yaml if requested
    if args.update_yaml:
        dataset_path = Path(args.data).parent
        update_data_yaml(str(dataset_path))

    # Train the model
    train_pitch_detection(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        pretrained=not args.no_pretrained
    )


if __name__ == '__main__':
    main()
