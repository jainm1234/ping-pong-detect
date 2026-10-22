# ping-pong-detect

Find the ping pong ball with YOLO and read its depth. Press `q` to quit.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo -E .venv/bin/python src/ping_pong_detect/webcam.py
```

`webcam.py` loads `runs/ball/weights/best.pt`. On macOS the Gemini camera has to be opened with `sudo`.
