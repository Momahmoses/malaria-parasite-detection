"""
YOLOv8 fine-tuning on blood smear dataset.
Generates synthetic training config if real dataset not present.
"""
import os
import yaml
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_dataset_yaml(data_dir: str = "data/blood_smear") -> str:
    yaml_content = {
        "path": os.path.abspath(data_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {
            0: "healthy_rbc",
            1: "ring_stage",
            2: "trophozoite",
            3: "schizont",
            4: "gametocyte",
        },
        "nc": 5,
    }
    yaml_path = f"{data_dir}/dataset.yaml"
    os.makedirs(data_dir, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f)
    logger.info(f"Dataset YAML created: {yaml_path}")
    return yaml_path


def generate_synthetic_annotations(data_dir: str = "data/blood_smear", n_images: int = 100):
    for split in ["train", "val"]:
        img_dir = Path(data_dir) / split / "images"
        label_dir = Path(data_dir) / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        n = n_images if split == "train" else n_images // 5
        for i in range(n):
            img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(img_dir / f"slide_{i:04d}.jpg")

            n_cells = np.random.randint(5, 30)
            with open(label_dir / f"slide_{i:04d}.txt", "w") as f:
                for _ in range(n_cells):
                    cls = np.random.choice([0, 0, 0, 1, 2, 3, 4], p=[0.7, 0.12, 0.08, 0.05, 0.05])
                    cx = np.random.uniform(0.05, 0.95)
                    cy = np.random.uniform(0.05, 0.95)
                    w = np.random.uniform(0.03, 0.08)
                    h = np.random.uniform(0.03, 0.08)
                    f.write(f"{cls} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n")

    logger.info(f"Generated synthetic annotations: {n_images} train, {n_images//5} val images")


def train(data_dir: str = "data/blood_smear", epochs: int = 50, img_size: int = 640,
          batch_size: int = 16, model_size: str = "n"):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return

    if not os.path.exists(f"{data_dir}/train"):
        logger.info("Generating synthetic training data...")
        generate_synthetic_annotations(data_dir, n_images=200)

    yaml_path = create_dataset_yaml(data_dir)

    model = YOLO(f"yolov8{model_size}.pt")
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project="models",
        name="malaria_detector",
        device=0 if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        patience=10,
        save=True,
        augment=True,
    )

    best_map = results.results_dict.get("metrics/mAP50", 0)
    logger.info(f"Training complete. Best mAP@50: {best_map:.4f}")

    model.export(format="onnx", imgsz=img_size)
    logger.info("Model exported to ONNX for edge deployment")
    return results


if __name__ == "__main__":
    train(epochs=30, model_size="n")
