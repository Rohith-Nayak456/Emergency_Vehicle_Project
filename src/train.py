"""
Training script for Emergency Vehicle Detection YOLO model.

This script can be run locally or adapted for Google Colab.
For Colab, see notebooks/model_training.ipynb.

Usage (local):
    python src/train.py

Usage (with custom settings):
    python src/train.py --epochs 50 --batch 8 --img 640
"""
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    DATA_YAML,
    MODEL_DIR,
    RESULTS_DIR,
    TRAIN_EPOCHS,
    TRAIN_BATCH_SIZE,
    TRAIN_IMG_SIZE,
    TRAIN_PATIENCE,
    RANDOM_SEED,
    YOLO_MODEL_VARIANT,
)


def train_model(
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = TRAIN_BATCH_SIZE,
    img_size: int = TRAIN_IMG_SIZE,
    patience: int = TRAIN_PATIENCE,
    seed: int = RANDOM_SEED,
    model_variant: str = YOLO_MODEL_VARIANT,
    data_yaml: str = str(DATA_YAML),
    project_dir: str = str(RESULTS_DIR / "training"),
    name: str = "emergency_vehicle_detector",
):
    """
    Train a YOLO model for emergency vehicle detection.

    Returns the path to the best model weights.
    """
    from ultralytics import YOLO

    print("=" * 60)
    print("EMERGENCY VEHICLE DETECTION - MODEL TRAINING")
    print("=" * 60)
    print(f"Model variant:   {model_variant}")
    print(f"Data YAML:       {data_yaml}")
    print(f"Epochs:          {epochs}")
    print(f"Batch size:      {batch_size}")
    print(f"Image size:      {img_size}")
    print(f"Patience:        {patience}")
    print(f"Seed:            {seed}")
    print(f"Output dir:      {project_dir}/{name}")
    print("=" * 60)

    # Load pretrained model
    print(f"\nLoading pretrained model: {model_variant}")
    model = YOLO(model_variant)

    # Start training
    start_time = time.time()

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=patience,
        seed=seed,
        project=project_dir,
        name=name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
        save=True,
        save_period=-1,  # Save only best
        plots=True,
        val=True,
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )

    elapsed = time.time() - start_time
    elapsed_min = elapsed / 60

    print(f"\n{'=' * 60}")
    print(f"Training completed in {elapsed_min:.1f} minutes")
    print(f"{'=' * 60}")

    # Copy best weights to models/
    best_path = Path(project_dir) / name / "weights" / "best.pt"
    dest_path = MODEL_DIR / "best.pt"

    if best_path.exists():
        import shutil
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, dest_path)
        print(f"Best weights copied to: {dest_path}")
    else:
        print(f"WARNING: Best weights not found at {best_path}")

    return str(dest_path)


def evaluate_model(
    model_path: str = None,
    data_yaml: str = str(DATA_YAML),
    img_size: int = TRAIN_IMG_SIZE,
):
    """Evaluate the trained model and print metrics."""
    from ultralytics import YOLO

    if model_path is None:
        model_path = str(MODEL_DIR / "best.pt")

    print(f"\nEvaluating model: {model_path}")
    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        verbose=True,
        plots=True,
        save_json=True,
    )

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@50:     {metrics.box.map50:.4f}")
    print(f"mAP@50-95:  {metrics.box.map:.4f}")

    # Per-class metrics
    class_names = model.names
    if hasattr(metrics.box, 'ap50') and metrics.box.ap50 is not None:
        print("\nPer-class AP@50:")
        for i, ap in enumerate(metrics.box.ap50):
            name = class_names.get(i, f"class_{i}")
            print(f"  {name:15s}: {ap:.4f}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Emergency Vehicle Detector")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch", type=int, default=TRAIN_BATCH_SIZE)
    parser.add_argument("--img", type=int, default=TRAIN_IMG_SIZE)
    parser.add_argument("--patience", type=int, default=TRAIN_PATIENCE)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--model", type=str, default=YOLO_MODEL_VARIANT)
    parser.add_argument("--evaluate-only", action="store_true")

    args = parser.parse_args()

    if args.evaluate_only:
        evaluate_model()
    else:
        best_path = train_model(
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img,
            patience=args.patience,
            seed=args.seed,
            model_variant=args.model,
        )
        print(f"\nBest model saved at: {best_path}")
        print("\nRunning evaluation...")
        evaluate_model(best_path)
