from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def main():
    if not (DATA / "data.yaml").exists():
        print("No dataset at", DATA)
        return
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    YOLO("yolo11s.pt").train(
        data=str(DATA / "data.yaml"),
        epochs=40,
        imgsz=1280,
        batch=8,
        device=device,
        workers=2,
        project=str(ROOT / "runs"),
        name="ball4",
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
