import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "data" / "blurball" / "csv" / "all_csv_annotations"
VIDEO_ROOT = ROOT / "data" / "blurball" / "videos"
OUT = ROOT / "data"
SIDE = 24
GAP_SECONDS = 2.0
TRAIN_VISIBLE = 400
TRAIN_EMPTY = 25
VAL_VISIBLE = 90
VAL_EMPTY = 10


def video_path(match, clip):
    folder = "videos_rallies" if match == "18" else "rallies_videos"
    return VIDEO_ROOT / match / folder / f"{clip}.mp4"


def load_clips():
    clips = {}
    for path in sorted(CSV_DIR.glob("*.csv")):
        match, _, clip = path.stem.split("_", 2)
        rows = []
        with path.open() as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    (
                        int(row["Frame"]),
                        int(float(row["Visibility"])),
                        float(row["X"]),
                        float(row["Y"]),
                    )
                )
        clips[(match, clip)] = rows
    return clips


def probe(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    width, height, rate = result.stdout.strip().split(",")
    num, den = rate.split("/")
    fps = float(num) / float(den)
    return int(width), int(height), fps


def eligible(frames, fps):
    chosen = []
    last_t = -1e9
    for frame in sorted(frames):
        timestamp = frame / fps
        if timestamp - last_t >= GAP_SECONDS:
            chosen.append(frame)
            last_t = timestamp
    return chosen


def take_even(items, count):
    if count >= len(items):
        return list(items)
    return [items[int(i * len(items) / count)] for i in range(count)]


def allocate(pools, total):
    keys = list(pools)
    quota = {key: 0 for key in keys}
    remaining = total
    while remaining:
        open_keys = [key for key in keys if quota[key] < len(pools[key])]
        if not open_keys:
            break
        for key in open_keys:
            if remaining == 0:
                break
            quota[key] += 1
            remaining -= 1
    return quota


def select_split(clips, probes, matches, visible_count, empty_count):
    by_match = {match: {"visible": {}, "empty": {}} for match in matches}
    for (match, clip), rows in clips.items():
        if match not in by_match:
            continue
        fps = probes[(match, clip)][2]
        visible = eligible([frame for frame, vis, _, _ in rows if vis == 1], fps)
        empty = eligible([frame for frame, vis, _, _ in rows if vis == 0], fps)
        if visible:
            by_match[match]["visible"][clip] = visible
        if empty:
            by_match[match]["empty"][clip] = empty

    selected = []
    for kind, count in (("visible", visible_count), ("empty", empty_count)):
        match_pools = {
            match: [item for clip_frames in info[kind].values() for item in clip_frames]
            for match, info in by_match.items()
        }
        match_quota = allocate(match_pools, count)
        for match, info in by_match.items():
            clip_quota = allocate(info[kind], match_quota[match])
            for clip, quota in clip_quota.items():
                for frame in take_even(info[kind][clip], quota):
                    selected.append((match, clip, frame, kind == "visible"))
    return selected


def yolo_line(x, y, width, height):
    x1 = max(0.0, x - SIDE / 2)
    y1 = max(0.0, y - SIDE / 2)
    x2 = min(float(width), x + SIDE / 2)
    y2 = min(float(height), y + SIDE / 2)
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w < 4 or box_h < 4:
        return None
    return (
        f"0 {((x1 + x2) / 2) / width:.6f} {((y1 + y2) / 2) / height:.6f} "
        f"{box_w / width:.6f} {box_h / height:.6f}\n"
    )


def extract_frames(video, frames, dest_dir):
    if dest_dir.exists():
        for old in dest_dir.glob("*"):
            old.unlink()
    dest_dir.mkdir(parents=True, exist_ok=True)
    select = "+".join(f"eq(n\\,{frame})" for frame in frames)
    pattern = dest_dir / "%05d.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            f"select='{select}'",
            "-fps_mode",
            "vfr",
            "-q:v",
            "3",
            str(pattern),
        ],
        check=True,
    )
    written = sorted(dest_dir.glob("*.jpg"))
    if len(written) != len(frames):
        raise RuntimeError(f"{video} produced {len(written)} frames, expected {len(frames)}")
    return written


def build():
    clips = load_clips()
    probes = {}
    for match, clip in clips:
        probes[(match, clip)] = probe(video_path(match, clip))
    train_matches = [f"{i:02d}" for i in range(22)]
    val_matches = [f"{i:02d}" for i in range(22, 26)]
    train = select_split(clips, probes, train_matches, TRAIN_VISIBLE, TRAIN_EMPTY)
    val = select_split(clips, probes, val_matches, VAL_VISIBLE, VAL_EMPTY)
    print(f"selected train {len(train)} val {len(val)}")

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            folder = OUT / kind / split / "blurball"
            folder.mkdir(parents=True, exist_ok=True)
            for old in folder.glob("*"):
                old.unlink()

    grouped = {}
    for split, rows in (("train", train), ("val", val)):
        for match, clip, frame, visible in rows:
            grouped.setdefault((match, clip), []).append((split, frame, visible))

    points = {}
    for (match, clip), rows in clips.items():
        points[(match, clip)] = {frame: (x, y) for frame, _, x, y in rows}

    for (match, clip), items in grouped.items():
        items.sort(key=lambda item: item[1])
        frames = [frame for _, frame, _ in items]
        tmp = ROOT / "data" / "blurball" / "tmp" / f"{match}_{clip}"
        written = extract_frames(video_path(match, clip), frames, tmp)
        width, height, _ = probes[(match, clip)]
        for (split, frame, visible), image in zip(items, written):
            name = f"bb_{match}_{clip}_{frame:05d}.jpg"
            target = OUT / "images" / split / "blurball" / name
            image.replace(target)
            label = OUT / "labels" / split / "blurball" / f"{Path(name).stem}.txt"
            if not visible:
                label.write_text("")
                continue
            x, y = points[(match, clip)][frame]
            line = yolo_line(x, y, width, height)
            if line is None:
                raise RuntimeError(f"ball outside frame {name} at {x},{y}")
            label.write_text(line)
        tmp.rmdir()

    (OUT / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: ball\n")
    train_images = list((OUT / "images" / "train" / "blurball").glob("*.jpg"))
    val_images = list((OUT / "images" / "val" / "blurball").glob("*.jpg"))
    print(f"wrote train {len(train_images)} val {len(val_images)}")


if __name__ == "__main__":
    build()
