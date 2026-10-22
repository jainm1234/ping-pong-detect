# ping-pong-detect

Find the ping pong ball and its depth in millimeters. Press `q` to quit.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo -E .venv/bin/python src/ping_pong_detect/webcam.py
```

Needs a Gemini camera. On macOS, open it with `sudo`. Weights are `runs/ball/weights/best.pt`.

`data/ball.zip` is the dataset: 80 train, 20 val. Class 0 is the ball. `neg_` images have empty labels.

```bash
python src/ping_pong_detect/train.py
```
