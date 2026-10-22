# ping-pong-detect

Find an orange ping pong ball with YOLO and read its depth. Press `q` to quit.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo -E .venv/bin/python src/ping_pong_detect/webcam.py
```

`webcam.py` loads the fine-tuned weights at `runs/ball/weights/best.pt`. On macOS the Gemini camera has to be opened with `sudo`.

## Dataset

`data/processed/ball` has 100 images: 80 train and 20 val. Class 0 is the orange ping pong ball.

80 images are labeled balls. The other 20 are faces and other sports balls, and their label files are empty so the model learns those are not the ball.

## Fine-tune

```bash
python src/ping_pong_detect/train.py
```

This starts from `yolo11n.pt` and writes `runs/ball/weights/best.pt`.
