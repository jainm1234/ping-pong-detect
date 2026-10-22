from pathlib import Path

import cv2
import numpy as np
from pyorbbecsdk import (
    AlignFilter,
    Config,
    OBFormat,
    OBFrameAggregateOutputMode,
    OBSensorType,
    OBStreamType,
    Pipeline,
)
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "runs" / "ball" / "weights" / "best.pt"


def main():
    if not MODEL.exists():
        print("No weights at", MODEL)
        return
    try:
        pipeline = Pipeline()
    except RuntimeError as error:
        if "uvc_open failed" in str(error):
            print("sudo -E .venv/bin/python src/ping_pong_detect/webcam.py")
            return
        raise

    config = Config()
    color = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color = color.get_video_stream_profile(1280, 720, OBFormat.MJPG, 30)
    depth = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    config.enable_stream(color)
    config.enable_stream(depth.get_default_video_stream_profile())
    config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.COLOR_FRAME_REQUIRE)
    pipeline.start(config)
    align = AlignFilter(OBStreamType.COLOR_STREAM)
    model = YOLO(str(MODEL))
    cv2.namedWindow("ball", cv2.WINDOW_NORMAL)
    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            aligned = align.process(frames)
            if aligned is None:
                continue
            frames = aligned.as_frame_set()
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if color is None or depth is None:
                continue
            frame = cv2.imdecode(np.asanyarray(color.get_data()), cv2.IMREAD_COLOR)
            raw = np.frombuffer(depth.get_data(), dtype=np.uint16)
            raw = raw.reshape(depth.get_height(), depth.get_width()).astype(np.float32)
            raw *= depth.get_depth_scale()
            result = model.predict(frame, conf=0.25, verbose=False)[0]
            boxes = result.boxes
            if boxes is not None and len(boxes):
                best = int(boxes.conf.argmax())
                x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[best].tolist())
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                radius = max(x2 - x1, y2 - y1) // 2
                cv2.circle(frame, center, max(radius, 1), (0, 255, 0), 2)
                values = raw[max(y1, 0) : y2, max(x1, 0) : x2]
                values = values[values > 0]
                if values.size:
                    near = np.percentile(values, 10)
                    values = values[values <= near * 1.15]
                    cv2.putText(
                        frame,
                        f"{int(np.median(values))} mm",
                        (x1, max(y1 - 8, 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            cv2.imshow("ball", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
