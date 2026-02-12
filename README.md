# 🕳️ Pothole Detection using YOLOv8

A real-time pothole detection system built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and trained on the [Roboflow Pothole Detection Dataset](https://universe.roboflow.com/roboflow-universe-projects/pothole-detection). Detects potholes in images, videos, and live webcam feeds.

---

## 📁 Project Structure

```
pothole_project/
│
├── pothole-detection-1/         # Downloaded dataset (auto-generated)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
│
├── runs/                        # Training outputs (auto-generated)
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt      # ← Your trained model
│
├── download_dataset.py          # Step 1: Download dataset from Roboflow
├── train_pothole.py             # Step 2: Train YOLOv8 model
├── test_pothole.py              # Step 3: Test on images
├── webcam_pothole.py            # Step 4: Real-time webcam detection
├── venv/                        # Virtual environment
└── README.md
```

---

## ⚙️ Requirements

- Python 3.8+
- pip
- A [Roboflow](https://roboflow.com) account (free) for dataset download

### Python Libraries

```
ultralytics
roboflow
opencv-python
matplotlib
```

---

## 🚀 Getting Started

### Step 1 — Set Up Project Folder

```bash
mkdir D:\pothole_project
cd D:\pothole_project
```

### Step 2 — Create & Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

Your terminal should now show `(venv)`.

### Step 3 — Install Required Libraries

```bash
pip install ultralytics roboflow opencv-python matplotlib
```

Verify the installation:

```bash
yolo checks
```

No errors means you're ready to go.

---

## 📦 Dataset Download

This project uses the **Pothole Detection** dataset from Roboflow Universe (free, pre-labeled, YOLO-formatted).

### Get Your Roboflow API Key

1. Sign up at [https://roboflow.com](https://roboflow.com) (free)
2. Go to your Dashboard → copy your **API Key**

### Create `download_dataset.py`

```python
from roboflow import Roboflow

# Paste your API key here
rf = Roboflow(api_key="PASTE_YOUR_API_KEY_HERE")

project = rf.workspace("roboflow-universe-projects").project("pothole-detection")
dataset = project.version(1).download("yolov8")

print("Dataset downloaded successfully")
```

### Run the Downloader

```bash
python download_dataset.py
```

After download, verify your folder structure:

- `pothole-detection-1/train/images/` — road images with potholes ✅
- `pothole-detection-1/train/labels/` — `.txt` annotation files ✅

---

## 🏋️ Training the Model

### Create `train_pothole.py`

```python
from ultralytics import YOLO

# Load base YOLOv8 nano model (fast training)
model = YOLO("yolov8n.pt")

# Train the model
model.train(
    data="pothole-detection-1/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8
)
```

### Run Training

```bash
python train_pothole.py
```

### What to Expect

You'll see live training progress like:

```
Epoch 1/20   loss: 0.5   mAP: 0.78
Epoch 2/20   loss: 0.4   mAP: 0.81
...
```

| Hardware | Estimated Training Time |
|----------|------------------------|
| CPU      | 30 – 60 minutes        |
| GPU      | 5 – 10 minutes         |

### Trained Model Location

After training, your model is saved at:

```
runs/detect/train/weights/best.pt
```

> ⚠️ **Keep this file safe** — it is your trained pothole detector.

---

## 🖼️ Testing on Images

### Create `test_pothole.py`

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("pothole-detection-1/test/images")
results[0].show()
```

### Run

```bash
python test_pothole.py
```

A window will pop up showing the detected potholes with bounding boxes.

---

## 📹 Real-Time Webcam Detection

### Create `webcam_pothole.py`

```python
from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    frame = results[0].plot()
    cv2.imshow("Pothole Detection", frame)

    if cv2.waitKey(1) == 27:  # Press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
```

### Run

```bash
python webcam_pothole.py
```

Point your webcam at a road or pothole image and watch detection in real time. Press **ESC** to exit.

---

## 📊 Model Details

| Property     | Value                  |
|--------------|------------------------|
| Architecture | YOLOv8 Nano (yolov8n)  |
| Dataset      | Roboflow Pothole v1    |
| Epochs       | 20                     |
| Image Size   | 640×640                |
| Batch Size   | 8                      |
| Output       | `best.pt`              |

---

## 🛠️ Troubleshooting

**`yolo checks` fails after install**
→ Make sure your virtual environment is activated (`(venv)` should appear in terminal).

**Dataset download fails**
→ Double-check your Roboflow API key and internet connection.

**Webcam not opening**
→ Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` if you have multiple cameras.

**Training too slow**
→ Reduce `epochs` to `10` or `batch` to `4` for faster (but less accurate) results. Install CUDA and PyTorch with GPU support for a major speed boost.

**`best.pt` not found after training**
→ Check the `runs/detect/` folder — if you've trained multiple times, the latest run may be in `train2/`, `train3/`, etc.

---

## 🔧 Customization

- **Improve accuracy** — Increase `epochs` to `50–100` and use `yolov8s.pt` (small) instead of nano.
- **Use your own images** — Replace the Roboflow dataset with your own labeled images using the same folder structure.
- **Run on video files** — Replace `cv2.VideoCapture(0)` with `cv2.VideoCapture("path/to/video.mp4")`.
- **Save detection output** — Add `results[0].save(filename="output.jpg")` to save annotated images.

---

## 📄 License

This project is for educational and research purposes. Dataset provided by [Roboflow Universe](https://universe.roboflow.com) under their respective terms.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com) for dataset hosting and management
- [OpenCV](https://opencv.org) for video capture and display
