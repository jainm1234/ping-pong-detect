# ping-pong-detect

Find the ping pong ball and its depth in millimeters. Press `q` to quit.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo -E .venv/bin/python src/ping_pong_detect/webcam.py
```

Needs a Gemini camera. On macOS, open it with `sudo`. Weights are `runs/ball4/weights/best.pt`.

`data/` is the dataset: 523 train, 131 val. Class 0 is the ball. Both splits contain `blurball`, `closeup`, `player`, `table`, and `white` images, with matching files under `labels/`. Frames with no ball have empty labels. A label box covers the whole ball and is at least 24 pixels wide.

```bash
python src/ping_pong_detect/train.py
```
