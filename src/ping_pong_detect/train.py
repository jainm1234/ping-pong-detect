from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "ball" / "data.yaml"


def main():
    if not DATA.exists():
        print("No dataset at", DATA.parent)
        return
    model = YOLO(str(ROOT / "yolo11n.pt"))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.train(
        data=str(DATA),
        epochs=40,
        imgsz=960,
        batch=8,
        device=device,
        workers=2,
        project=str(ROOT / "runs"),
        name="ball",
        exist_ok=True,
    )


if __name__ == "__main__":
    main()
