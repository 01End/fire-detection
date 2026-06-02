# FireWatch — Camera Fire Detection with Per-Floor Localization

Detects fire/smoke from camera video (recorded **or** live RTSP/USB) and reports **which
floor** the fire is on. Built for a company deployment with a few cameras each covering
several floors. Every component uses a **commercial-friendly, license-free** stack
(PyTorch/torchvision BSD, OpenCV/Streamlit Apache) — no AGPL / Ultralytics, no paid licenses.

## How it works

```
FrameSource → FireDetector → FloorMapper → Debouncer → EventSink(s)
```

1. **Sources** read frames from video files, image folders, RTSP, or USB webcams behind a
   common interface.
2. **Detector** — a torchvision detection model trained on a public fire/smoke dataset —
   returns fire/smoke boxes (auto CPU/GPU).
3. **Floor mapping** maps each detection's bottom-center point to a floor using per-camera
   polygon zones.
4. **Debouncer** confirms sustained detections (N-of-M frames + cooldown) to cut false alarms.
5. **Event sinks** log the event, save an annotated snapshot, feed the dashboard, and (later)
   forward to the company's central system.

## Layout

| Path | Purpose |
|------|---------|
| `src/firewatch/sources/`   | Frame sources (file / RTSP / webcam) + factory |
| `src/firewatch/detection/` | Model builder + device-agnostic detector |
| `src/firewatch/floors/`    | Polygon zones + bbox→floor mapping |
| `src/firewatch/events/`    | DetectionEvent, debounce, pluggable sinks |
| `src/firewatch/pipeline.py`| Orchestrator wiring it all together |
| `tools/label_zones.py`     | Interactive tool to draw floor zones on a snapshot |
| `training/`                | Download, dataset, train, eval, export |
| `dashboard/app.py`         | Streamlit feeds + alerts panel |
| `configs/cameras/`         | Per-camera source + floor-zone config |

## Quick start

```bash
pip install -r requirements.txt
pip install -e .                       # exposes the `firewatch` command

# 1. Define floor zones for a camera (draws on a snapshot)
firewatch setup-zones --config configs/cameras/cam1.yaml

# 2. Run detection on that camera
firewatch run --config configs/cameras/cam1.yaml

# 3. Watch the dashboard
streamlit run dashboard/app.py
```

Training the detector is documented in `training/` (run after verifying the dataset license).

## License

Code: MIT (see project owner). All dependencies are permissively licensed for commercial use.
