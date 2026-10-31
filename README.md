# Vision Mate: AI-Powered Assistive Vision System

A wearable, camera-based assistant for visually impaired and deaf/mute users. It looks at the world through a camera, works out what it is seeing (objects, printed text, traffic signs and lights, familiar faces, sign-language gestures) and tells the user through **spoken audio feedback**. It is built to run in real time on a small edge device (Raspberry Pi / NVIDIA Jetson) rather than on a workstation.

> Related: patent application *"Assistive Navigation Apparatus and Method Thereof"* (Indian Patent App. No. 202641105831).

<!-- Add a demo GIF, screenshot, or Loom link here. Reviewers look for this first. -->

---

## What it does

| Capability | What the user hears | How it works |
|---|---|---|
| **Object detection** | "person detected, chair detected" | YOLOv8n on every 3rd frame, confidence > 0.5, speaks only when the set of objects changes |
| **Text reading (OCR)** | The text on a page or sign, read aloud | Crop, 2x upscale, grayscale, bilateral filter, Otsu threshold, then Tesseract (`--oem 3 --psm 6`) |
| **Traffic sign recognition** | "Stop", "Speed limit 50 km/h", etc. | CNN trained on the 43-class GTSRB dataset, confidence threshold and speech cooldown |
| **Traffic light state** | "You must stop, because the traffic light is red" | Small CNN classifying red / yellow / green / off |
| **Face recognition** | The name of a known person | `face_recognition` (dlib) encodings of photos in `Module-4/images/` |
| **Sign-language to speech** | Spoken letters, digits, words and phrases | Two-hand landmark tracking (cvzone), then angle and distance features, then Random Forest classifiers |
| **Scene captioning** | A spoken sentence describing the scene | InceptionV3 image features with an LSTM caption decoder |

### Design choices for real-time use on edge hardware
- **Frame skipping.** Heavy models run on every 3rd to 15th frame instead of every frame, depending on the module.
- **Mode switching.** Only one pipeline is active at a time (keys `1`, `2`, `4`, `7` in `main.py`), so the device is not running every model at once.
- **Speech cooldowns and locks.** A cooldown between utterances and a thread lock stop the audio from overlapping or repeating the same message.
- **Lightweight audio playback** with `mpg123` instead of a heavier audio library.

### Reported results
| Metric | Result |
|---|---|
| Traffic detection and OCR-based navigation accuracy | 96.5% |
| Sign-language gesture and facial recognition accuracy | 97% |
| Processing speed gain from frame skipping and mode switching | +40% |
| Real-time inference latency | < 200 ms |

---

## Repository structure

```
.
├── main.py                  # Mode-switching launcher (keys 1 / 2 / 4 / 7)
├── detection.py             # Standalone YOLOv8 object detection + speech
├── traffic_sign.py          # Standalone live traffic-sign classifier + speech
├── test.py                  # Minimal YOLOv8 + gTTS test script
├── yolov8n.pt               # YOLOv8 nano weights
├── traffic_model.keras      # Traffic-sign model used by traffic_sign.py
├── Instructions.txt         # dlib / face_recognition / TensorFlow install notes
├── commands for xavier      # Setup notes for NVIDIA Jetson Xavier (JetPack 5.0)
├── Module-1/                # voice.py: text-to-speech helper (gTTS)
├── Module-2/                # OCR.py: camera OCR with preprocessing + Tesseract
├── Module-3/                # Image_Captioning.py / .ipynb: InceptionV3 + LSTM captioning
├── Module-4/                # reco.py, simple_facerec.py, images/: face recognition
├── Module-5/                # Training.py, classification.py: traffic-sign CNN (GTSRB, 43 classes)
├── Module-6/                # traffic_light.py: traffic-light state classifier
└── Module-7/                # sign.py, models/: sign-language recognition (A-Z, 0-9, words)
```

---

## Tech stack

Python, OpenCV, YOLOv8 (Ultralytics), TensorFlow / Keras, scikit-learn (Random Forest), `face_recognition` (dlib), cvzone hand tracking, Tesseract OCR (`pytesseract`), gTTS, `mpg123`.

---

## Setup

Tested on Linux (Ubuntu 20.04) on edge hardware. See `commands for xavier` for the full Jetson Xavier / JetPack 5.0 walkthrough.

```bash
# 1. System packages
sudo apt-get install python3-pip python3-venv mpg123 tesseract-ocr \
    build-essential cmake libopenblas-dev liblapack-dev

# 2. Virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Python packages
pip install opencv-python numpy ultralytics tensorflow keras scikit-learn joblib \
    pandas pytesseract gTTS playsound face_recognition cvzone mediapipe

# 4. Folder used by Module-1's voice helper
mkdir -p Output
```

`face_recognition` needs `dlib`, which compiles from source on most edge devices. Follow `Instructions.txt` if `pip install face_recognition` fails.

> `requirements.txt` is a full `pip freeze` from the development machine (it includes OS-level packages), so use it as a version reference rather than installing it directly.

---

## Usage

Run from the repository root. Each script opens the camera and speaks results aloud. Press `q` to quit (`Esc` in `main.py`).

```bash
python detection.py          # YOLOv8 object detection with spoken output
python traffic_sign.py       # traffic-sign recognition
python Module-2/OCR.py       # read printed text aloud
python Module-4/reco.py      # recognise known faces
python Module-7/sign.py      # sign-language to speech
python Module-3/Image_Captioning.py   # scene captioning
```

**Adding known faces:** drop a clear, front-facing photo into `Module-4/images/` named after the person (for example `alex.jpeg`). The filename becomes the name that is spoken.

---

## Known limitations

- `main.py` is an in-progress unified launcher. It imports `ocr`, `recognise` and `detect_sign`, which do not exist in the modules yet, and Modules 3, 5 and 6 are not wired into it. Run the modules individually as shown above.
- Text-to-speech uses gTTS, which needs an internet connection.
- OCR is tuned for printed text on paper, not handwriting or stylised signage.
- Face recognition covers only the people whose photos are in `Module-4/images/`.

## Future work
- Finish wiring all modules into the mode-switching launcher
- Offline text-to-speech
- Obstacle distance estimation for navigation
- A compact wearable enclosure and battery management

---

## Author

**Saksham Pundir**, B.Tech, Electronics & Communication Engineering, VIT Chennai
[LinkedIn](https://linkedin.com/in/saksham-pundir-491953245) · [GitHub](https://github.com/saksham0333)
