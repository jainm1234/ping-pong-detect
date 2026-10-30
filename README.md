# ping-pong-detect

Find the ping pong ball and its depth in millimeters. Press `q` to quit.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo -E .venv/bin/python src/ping_pong_detect/webcam.py
```

Needs a Gemini camera. On macOS, open it with `sudo`. Weights are `runs/orange/weights/best.pt`, created by `train.py`.

`data/` is a synthetic dataset of orange ping pong balls: 1040 train, 260 val, all 1280×720 under `images/<split>/<type>` with matching files under `labels/`. Class 0 is the ball. Balls range from 12 to about 560 pixels wide and include motion blur, defocus, dim and colored lighting, and partly cropped balls. About 10% of frames have no ball, and frames with no ball have empty labels. Other round things that are not balls appear unlabeled: lights, clocks, real faces, road signs, plates, wheels, and buttons. No other kinds of balls are used. A label box covers the whole visible ball, including its motion streak.

The `orange` folders (800 train, 200 val) place balls on general scenes. The `pov` folders (240 train, 60 val) show a table from where a player stands, using the 14 table scenes in `data/backgrounds/pov`. Their corners are in `pov.json`, which maps the real 1525 × 2740 mm table onto each photo, so a 40 mm ball gets the right size wherever it is: resting on the table, in the air over either side, or between the player and the table. The tables in `pov_06`, `pov_11`, and `pov_15` are only used for validation.

Balls come from 16 photographed orange balls in `data/backgrounds/sprites` (the 3 red ones on those sheets are skipped) plus a renderer that varies shade, finish, lighting, seam, and logo. Backgrounds are about 400 Wikimedia Commons photos (credits in `data/backgrounds/credits.csv`) plus table photos from the earlier white-ball dataset. Validation uses backgrounds and photographed balls that never appear in training. Rejected backgrounds are listed in `data/backgrounds/rejected.txt`.

```bash
python src/ping_pong_detect/make_orange.py
python src/ping_pong_detect/train.py
```

To train, only `data/images`, `data/labels`, and `data/data.yaml` are needed. `train.py` uses an NVIDIA GPU, then Apple GPU, then CPU, and downloads `yolo11s.pt` on first run. Rebuilding the dataset with `make_orange.py` uses only files in `data/backgrounds/`.

`make_orange.py` writes sample sheets with boxes drawn to `runs/preview/`. Training saves to `runs/orange`, which `webcam.py` reads.
