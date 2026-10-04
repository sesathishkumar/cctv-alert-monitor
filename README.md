# Watchtower — CCTV Alert Monitor

A local prototype for watching video and raising reviewable movement alerts inside a configured area. It provides a monitoring dashboard, video uploads, a synthetic demo, evidence images, persistent alert history, acknowledgment, and an optional browser alert tone.

## Run

Requires Python 3.12. From `/workspace/cctv-alert-monitor`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

Open the dashboard on port 5000 on the local machine. The development server binds to loopback. Start with **Synthetic demo** to see a moving rectangle trigger alerts, or choose **Video file** and upload MP4, AVI, MOV or MKV footage up to 200 MB. Decoding depends on the video codec.

Set the zone using frame percentages. The yellow rectangle marks it. Movement must persist for the configured duration before alerting; cooldown limits repeated alerts. Settings apply when starting or restarting monitoring. Uploaded videos play at their reported frame rate and stop at the end.

## Detection and evidence

OpenCV MOG2 background subtraction detects foreground movement, removes shadows and small noise, and filters movement to the configured zone. Twenty frames are used for background calibration. This is **motion detection**, not person recognition, intent assessment, fall detection, or unattended-object detection. Light changes, animals, camera shake, and some scene transitions can generate false alerts; objects that stop moving can disappear into the background model. Clips beginning with an event may be missed during calibration. Use a fixed camera and tune with representative footage.

Alerts record capture time, playback time, input label and an annotated JPEG in `data/evidence/`. The SQLite database survives restarts. Acknowledgment marks an alert as seen; it does not classify the event as safe or resolve it. Video uploads remain in `data/uploads/`. No automatic deletion or retention policy is configured. Evidence and uploads stay on this machine; no external alert messages are sent. The browser sound is optional and requires enabling it in the dashboard.

## Verification

```sh
.venv/bin/python -m pip install pytest==8.4.2
.venv/bin/python -m pytest -q
```

Checks cover detection duration, cooldown, movement outside the zone, invalid settings, actual uploaded-video decoding, end of playback, persisted alerts, evidence decoding, acknowledgment, and unreadable video errors.

## Next development milestones

1. Connect actual RTSP camera feeds, with reconnect handling and offline alerts.
2. Add person detection and zone-entry rules using a validated detection model.
3. Evaluate falls and abandoned objects as separate event detectors using labeled footage.
4. Add operator accounts, access control, retention rules, multi-camera support and configured notification channels before deployment beyond a local prototype.

This version processes one demo or uploaded video at a time. It has no RTSP input or authentication, and is not ready for an operational police/security deployment. Human review is required for alerts. Measure false alarms and missed events on footage from the intended cameras before relying on it.

## Gun and knife detection

Weapon mode supports separate **possible gun** and **possible knife** alerts, annotated bounding boxes, model confidence scores, per-category duration and cooldown, evidence and acknowledgment. It runs independently of motion, so a stationary visible weapon can be detected. A weapon's bounding-box center must lie inside the configured zone. This identifies visible objects; it does not associate a weapon with a particular person or establish that someone is carrying it. Violence detection is not implemented.

A compatible pretrained model is now configured from the GitHub source supplied by the user. Its embedded classes are `guns` and `knife`. Install the pinned, checksum-verified weights after cloning:

```sh
.venv/bin/python scripts/download_weapon_model.py
```

Then launch `app.py` normally; the bundled manifest and local model paths are used automatically. The weights are excluded from Git. Their model metadata declares AGPL-3.0, while upstream's LICENSE is GPL-3.0 and its README incorrectly claims MIT. Read `models/NOTICE.md` before deployment or redistribution.

For Windows PowerShell with Python 3.12 installed, from the extracted project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_weapon_model.py
.\.venv\Scripts\python.exe app.py
```

Choose **Possible guns + knives** in the dashboard and upload your video. Detection still requires reviewing each alert. The synthetic demo contains a moving rectangle, not weapons, so it is useful for motion mode only.

Actual OpenCV inference has detected both categories in upstream demonstration/training images. Those checks are not independent accuracy evaluation on CCTV footage. Fixture-based parser and alert tests remain separate from real-model checks. If weights are absent, weapon mode is disabled until downloaded.

Supply a legally usable YOLOv8 **detection** model trained with both guns and knives, exported as a static square-input ONNX model without embedded NMS. The required output is `[1, 4 + class_count, prediction_count]`, with center-x, center-y, width, height in input-image pixels and per-class scores. Models with objectness channels, end-to-end NMS, or other output layouts need a different adapter. OpenCV must support the model's exported operators.

Create a manifest with the exact class-index order from that model's training/export metadata. For a model with class 0 handgun and class 1 knife:

```json
{
  "format": "yolov8-detection-onnx",
  "input_size": 640,
  "sha256": "REPLACE_WITH_EXPECTED_MODEL_SHA256",
  "classes": [
    {"label": "handgun", "category": "gun"},
    {"label": "knife", "category": "knife"}
  ]
}
```

Use the checksum supplied with the trusted model or record the checksum of your own verified export. Class order cannot be guessed from this example. Multiple gun types can map to `gun`; non-weapon classes can map to `ignore`. A generic COCO model has no gun class and is insufficient for this workflow.

Start with your actual paths:

```sh
WEAPON_MODEL_PATH=/path/to/weapons.onnx \
WEAPON_MANIFEST_PATH=/path/to/weapons.json \
.venv/bin/python app.py
```

Select **Possible guns + knives** and adjust the confidence threshold. Configuration and checksum verification do not prove inference readiness: model loading and output validation occur during monitoring, with errors shown in the dashboard. Evaluate footage with guns, knives, similar-looking tools and no weapons; measure false alarms and missed events before deployment. A score is not a calibrated probability of a weapon or danger. Small or obscured objects can be missed, and tools or toy weapons may be confused with weapons.
