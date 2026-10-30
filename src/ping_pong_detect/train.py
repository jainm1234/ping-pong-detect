from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main():
    if not (DATA / "data.yaml").exists():
        print("No dataset at", DATA)
        return
    if torch.cuda.is_available():
        device = 0
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    YOLO("yolo11s.pt").train(
        data=str(DATA / "data.yaml"),
        epochs=40,
        imgsz=1280,
        batch=8,
        device=device,
        workers=2,
        project=str(ROOT / "runs"),
        name="orange",
    )


if __name__ == "__main__":
    main()
