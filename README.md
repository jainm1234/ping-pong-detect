# ping-pong-detect

Find the ping pong ball and its depth in millimeters. Press `q` to quit.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo -E .venv/bin/python src/ping_pong_detect/webcam.py
```

Needs a Gemini camera. On macOS, open it with `sudo`. Weights are `runs/ball4/weights/best.pt`.

`data/` is a synthetic dataset of orange ping pong balls: 800 train, 200 val, all 1280×720 under `images/<split>/orange` with matching files under `labels/`. Class 0 is the ball. Balls range from 12 to about 560 pixels wide and include motion blur, defocus, dim and colored lighting, and partly cropped balls. About 10% of frames have no ball, and frames with no ball have empty labels. Other round things such as basketballs, tennis balls, soccer balls, baseballs, beach balls, oranges, lights, bokeh, smiley faces, clocks, and real faces appear unlabeled. A label box covers the whole visible ball, including its motion streak.

Balls come from 19 photographed orange balls in `data/backgrounds/sprites` plus a renderer that varies shade, finish, lighting, seam, and logo. Backgrounds are about 400 Wikimedia Commons photos (credits in `data/backgrounds/credits.csv`) plus table photos from the earlier white-ball dataset. Validation uses backgrounds and photographed balls that never appear in training. Rejected backgrounds are listed in `data/backgrounds/rejected.txt`.

```bash
python src/ping_pong_detect/make_orange.py
python src/ping_pong_detect/train.py
```

`make_orange.py` writes sample sheets with boxes drawn to `runs/preview/`. Training saves to `runs/orange`; point `MODEL` in `webcam.py` there once it finishes.
