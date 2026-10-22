import shutil
import zipfile
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "data" / "ball.zip"
DATA = ROOT / "data" / "processed" / "ball"


def main():
    if not ZIP.exists():
        print("No dataset at", ZIP)
        return
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    zipfile.ZipFile(ZIP).extractall(DATA)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    YOLO("yolo11n.pt").train(
        data=str(DATA / "data.yaml"),
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
