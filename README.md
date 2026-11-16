# YOLOv8 on Caltech256 + Unity Drone Bridge

Fine-tuning YOLOv8 on **Caltech256** (256 object categories, ~30 000 images) and wiring the resulting
detector into two real consumers: a live IP-camera feed with an on-screen HUD, and a Flask service that
accepts video frames from a Unity client and returns detections as JSON.

University coursework project, cleaned up for publication. Code comments are in Russian (original),
README in English.

## Pipeline

```
Caltech256 (256 folders of JPEGs)
        │  src/prepare_caltech256.py
        ▼
YOLO dataset  images/{train,val} + labels/{train,val} + caltech256.yaml
        │  src/train_yolo.py  (ultralytics)
        ▼
runs/detect/.../weights/best.pt
        │
        ├── src/detect_realtime.py   IP camera → boxes + HUD → saved frames
        └── src/unity_server.py      Unity frames → JSON detections
```

**Dataset conversion.** Caltech256 is a *classification* dataset, so each image is turned into a
detection sample by writing a single full-frame bounding box labelled with the class index of its
folder, then splitting into train/val with `sklearn.model_selection.train_test_split` and emitting the
`caltech256.yaml` descriptor YOLO expects. It is a pragmatic way to reuse a classification dataset for
a detector, and a good exercise in dataset plumbing.

**Training.** `src/train_yolo.py` runs Ultralytics YOLOv8 over the full converted dataset
(`KMP_DUPLICATE_LIB_OK` is set for the Windows OpenMP/OpenCV clash). `labs/quick_train.py` is the
reduced-epoch variant used while iterating.

**Live inference.** `src/detect_realtime.py` polls an MJPEG-ish endpoint (`http://<ip>:8080/shot.jpg`),
runs the trained model, draws boxes with class and confidence, keeps a HUD counter and saves frames to
`detected_objects/`. The camera IP is a placeholder in the published version — set your own.

**Unity bridge.** `src/unity_server.py` exposes a Flask endpoint that receives a frame from the Unity
client, runs YOLO and returns `{"detections": [{class, confidence, bbox}]}`. This is the "simulator
sends its camera view, gets detections back" pattern used for drone perception prototyping.

**Algorithm scheme.** `tools/flowchart.py` renders the detection loop as a Graphviz diagram
(`docs/drone_scheme.dot` is the exported DOT source) — the same scheme that goes into the course report.

## Layout

```
caltech256-yolo/
├─ src/
│  ├─ prepare_caltech256.py   # classification dataset -> YOLO detection dataset
│  ├─ train_yolo.py           # full training run
│  ├─ detect_realtime.py      # IP-camera inference + HUD + frame capture
│  └─ unity_server.py         # Flask service for the Unity client
├─ labs/
│  └─ quick_train.py          # short training run for iteration
├─ tools/
│  └─ flowchart.py            # Graphviz algorithm diagram
├─ dataset/
│  ├─ caltech256.yaml         # YOLO dataset descriptor
│  ├─ classes.txt             # 256 class names
│  └─ objects_sample.txt      # sample detection output
├─ docs/
│  └─ drone_scheme.dot        # exported flowchart source
└─ requirements.txt
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

1. Download Caltech256 (~1.2 GB) and unpack it so that `256_ObjectCategories/256_ObjectCategories/`
   contains the 256 numbered class folders.
2. Convert to YOLO format:

```bash
cd src
python prepare_caltech256.py      # writes caltech256_yolo_full/
```

3. Train:

```bash
python train_yolo.py
```

4. Run inference on a live camera:

```bash
python detect_realtime.py         # edit the `url` and model_path constants first
```

5. Or start the Unity bridge:

```bash
python unity_server.py            # Flask on the default port
```

## Notes and limitations

* Dataset images and trained weights are **not** versioned (`.gitignore`), and `runs/` is training
  output — only the code and the dataset descriptor live here.
* Full-frame bounding boxes mean the detector learns "what dominates the frame", not precise
  localisation; it is a fine bootstrap for the Unity demo, not a production detector.
* Camera endpoints and model paths are constants at the top of each script, not CLI arguments — kept
  as-is from the course version.
* The Unity client itself is not part of this repository; the Flask contract is the integration point.

## Stack

Python 3.10+ · Ultralytics YOLOv8 · PyTorch · OpenCV · Flask · scikit-learn · PyYAML · Graphviz
