# VisionMate: Edge-Computed Wearable Assistive Vision System

[![DOI](https://zenodo.org/badge/1209215688.svg)](https://doi.org/10.5281/zenodo.24335718)

A wearable prototype for visually impaired people, built on a **Raspberry Pi 5**. It runs object detection, text reading (OCR), face recognition, traffic sign and traffic light classification, scene description (image captioning) and Indian Sign Language (ISL) recognition **on the device**, so no camera image leaves it, and speaks the result to the user. A separate **Arduino Uno R3** subsystem sends an SOS text message with GPS coordinates when the user presses a button, even if the Raspberry Pi has failed.

This repository holds the code for the paper *"An edge-computed wearable system with on-device vision inference for text reading, object detection and sign language recognition for visually impaired people"* (Pundir, Kumar, Ralph A R and Pradhan, School of Electronics Engineering, VIT Chennai): the mode launcher and frame-decimation logic, the recognition modules, the Arduino firmware, and the training and evaluation scripts.

> Related: patent application *"Assistive Navigation Apparatus and Method Thereof"* (Indian Patent App. No. 202641105831).

---

## Modes

`main.py` runs exactly one mode at a time. The mode numbers match Figure 5 of the paper.

| Mode | What it does | What the user hears | Module | Frames processed |
|---|---|---|---|---|
| 0 | Idle: no module running, no model in memory | "Idle mode" | `main.py` | none |
| 1 | Scene description: InceptionV3 encoder + LSTM decoder, beam search (width 5, up to 74 tokens) | A sentence describing the scene | `Module-3/Image_Captioning.py` | on request (press `c` in the camera window) |
| 2 | Text reading: crop, 2x upscale, grayscale, bilateral filter, Otsu threshold, Tesseract LSTM (`--oem 3 --psm 6`) | The printed text, read aloud | `Module-2/OCR.py` | 1 in 15 (N = 15) |
| 3 | Object detection: YOLOv8n, confidence > 0.5, speaks only when the set of objects changes | "person detected, chair detected" | `detection.py` | 1 in 3 (N = 3) |
| 4 | Face recognition: dlib 128-d face encodings, tolerance 0.5, 4 s cooldown per person | The name of a known person | `Module-4/reco.py` | 1 in 5 (N = 5) |
| 5 | Traffic sign: CNN trained on the 43-class GTSRB dataset (30 x 30 input, 165,355 parameters). When it sees the "Traffic signals" sign it also runs the traffic light classifier | "Stop", "Speed limit 50 km/h", "Traffic signals. Traffic light appears red" | `traffic_sign.py` | every frame |
| 6 | Traffic light: small CNN (29,700 parameters) classifying red / yellow / green / off | "Traffic light appears green" | `Module-6/traffic_light.py` | 1 in 5 (N = 5) |
| 7 | Sign language: MediaPipe hand landmarks (up to two hands) turned into an 893-value geometric feature vector, classified by random forests for characters, numbers and two-handed ISL words | The spoken letters, numbers or sentence | `Module-7/sign.py` | every frame |

### How the device stays within its thermal limits
- **Single-model residency.** Only one mode runs at a time. When the user changes mode, `main.py` stops the running module, which releases its model and the camera, before it starts the next one. Peak memory is therefore set by the largest single model, not by the sum of all of them.
- **Frame decimation.** Each module passes only every N-th camera frame to its model (`frame_count % N`). N is set per module, as in the table above.
- **Mode-change cooldown.** A mode change within 3 seconds of the previous one is ignored, so repeated key presses do not restart a module or repeat the announcement.
- **Speech cooldowns.** Within each mode, a cooldown and a thread lock stop the same result being spoken repeatedly or two messages overlapping.

### Sign language features (Section 4.1)
`Module-7/features.py` builds the 893-value vector: 861 pairwise distances between the 42 landmarks of the two hands (distances involving an absent hand are zero), 30 joint angles taken only along each finger (15 per hand), and 2 hand-presence flags. The models shipped in the original release were trained on an older 899-value vector; `sign.py` detects them and still runs them, and prints a warning until they are retrained (see below).

### Emergency SOS subsystem
The SOS path runs on a separate Arduino Uno R3 with a NEO-6M GPS receiver and a SIM800C GSM module. When the user presses the SOS button, the Arduino sends an SMS with the latest GPS fix to a preset contact. It shares only the 5 V power rail with the Raspberry Pi, so it still works if the Pi has crashed or shut down. The SIM800C works only on 2G networks, so it needs a SIM on an operator that still runs 2G (Airtel was used in testing). Firmware, wiring and upload steps: [`Arduino/README.md`](Arduino/README.md).

### Results
Thermal behaviour, per-module latency and memory, and recognition accuracy on held-out data are reported in the paper (Tables 2 to 4). The scripts in `evaluation/` reproduce them; see [Reproducing the results](#reproducing-the-results).

---

## Repository structure

```
.
├── main.py                  # Mode launcher: modes 0-7, one module at a time, optional thermal log
├── detection.py             # Mode 3: YOLOv8n object detection + speech
├── traffic_sign.py          # Mode 5: live traffic-sign classifier (calls the traffic light classifier)
├── traffic_model.keras      # Traffic-sign model of Section 4.5 (saved without optimizer state)
├── yolov8n.pt               # YOLOv8 nano weights
├── test.py                  # Minimal YOLOv8 + gTTS test script
├── requirements.txt         # Python packages for the Raspberry Pi 5
├── requirements-dev-freeze.txt  # Full pip freeze of the original development machine (reference only)
├── Instructions.txt         # dlib / face_recognition / TensorFlow install notes
├── commands for xavier      # Older setup notes for NVIDIA Jetson Xavier (JetPack 5.0)
├── Module-1/                # voice.py: text-to-speech helper (gTTS + mpg123)
├── Module-2/                # Mode 2: OCR.py, camera OCR with preprocessing + Tesseract
├── Module-3/                # Mode 1: Image_Captioning.py / .ipynb, InceptionV3 + LSTM captioning
├── Module-4/                # Mode 4: reco.py, simple_facerec.py, images/ (face recognition)
├── Module-5/                # Training.py (trains the traffic-sign CNN on GTSRB), classification.py (classify an image file)
├── Module-6/                # Mode 6: traffic_light.py (live mode + used by traffic_sign.py), train_traffic_light.py
├── Module-7/                # Mode 7: sign.py, features.py, collect_dataset.py, convert_legacy_dataset.py,
│                            #         train_models.py, models/ (A-Z, 0-9, words)
├── Arduino/                 # sos_firmware/sos_firmware.ino and wiring guide (SOS subsystem)
└── evaluation/              # scripts that reproduce Tables 2-4 and Figures 11-12
```

`traffic_model.h5` holds the same weights as `traffic_model.keras` in the older file format. `traffic_classifier.keras` is a different training run of the same network and is not used by the code.

---

## Hardware

- Raspberry Pi 5 (8 GB), Raspberry Pi OS 64-bit, no heatsink or fan
- USB webcam
- Wired earphones (audio played with `mpg123`)
- Arduino Uno R3 + NEO-6M GPS + SIM800C GSM module + SOS push-button (emergency subsystem)
- External power banks

## Tech stack

Python 3, OpenCV, YOLOv8 (Ultralytics), TensorFlow / Keras, scikit-learn (random forest), `face_recognition` (dlib), MediaPipe / cvzone hand tracking, Tesseract OCR (`pytesseract`), gTTS, `mpg123`. Arduino: TinyGPSPlus, SoftwareSerial.

---

## Setup

On the Raspberry Pi 5 (Raspberry Pi OS 64-bit):

```bash
# 1. System packages
sudo apt-get install python3-pip python3-venv mpg123 tesseract-ocr \
    build-essential cmake libopenblas-dev liblapack-dev

# 2. Virtual environment
python3 -m venv venv && source venv/bin/activate

# 3. Python packages
pip install -r requirements.txt
```

`face_recognition` needs `dlib`, which compiles from source on most edge devices. Follow `Instructions.txt` if `pip install face_recognition` fails.

The captioning and traffic light models were saved with older Keras versions. Current TensorFlow (Keras 3) cannot rebuild them from their stored configuration, so `Image_Captioning.py` and `traffic_light.py` rebuild the same networks in code and load the saved weights; this works on any TensorFlow version from 2.15. The first run of scene description downloads the InceptionV3 ImageNet weights (about 92 MB).

Speech uses gTTS, so the Pi needs an internet connection (a mobile network was used in testing).

---

## Usage

Run from the repository root:

```bash
python main.py
```

Type a mode number (0-7) and press Enter. Type `q` to quit. Press `q` in a module's camera window to close that module.

Each module can also be run on its own:

```bash
python Module-3/Image_Captioning.py   # mode 1: scene description (press c)
python Module-2/OCR.py                # mode 2: read printed text aloud
python detection.py                   # mode 3: object detection
python Module-4/reco.py               # mode 4: recognise known faces
python traffic_sign.py                # mode 5: traffic-sign recognition
python Module-6/traffic_light.py      # mode 6: traffic light state
python Module-7/sign.py               # mode 7: sign language to speech
python Module-5/classification.py photo.jpg   # classify a traffic sign in an image file
```

**Adding known faces:** put a clear, front-facing photo in `Module-4/images/`, named after the person (for example `alex.jpeg`). The file name is the name that is spoken.

---

## Training

### Traffic sign classifier (Sections 4.5, 5.1, 5.4)
Download GTSRB (the Kaggle copy with `Train/<ClassId>/`, `Test.csv` and `Test/`), then:

```bash
python Module-5/Training.py --data /path/to/gtsrb --split random   # random 80:20 split
python Module-5/Training.py --data /path/to/gtsrb --split track    # whole tracks held out
```

Adam, categorical cross-entropy, batch size 32, 15 epochs. The model is saved to `traffic_model.keras` without optimizer state, and the training curves (Figure 10) to `results/traffic_sign/`.

### Traffic light classifier (Section 4.6)
Put the self-collected photographs in `<data>/train/{red,yellow,green,off}/` (and held-out ones in `<data>/test/...`), then:

```bash
python Module-6/train_traffic_light.py --data /path/to/traffic_lights
```

### Sign language models (Sections 4.1, 5.2)
The models must be retrained on the 893-value features. Either convert the recordings used for the original models (no need to record again; the conversion is exact):

```bash
python Module-7/convert_legacy_dataset.py old_words.csv --mode words --label-col label --signer S1 --session 1
```

or record new samples (only the feature vector is stored, never the image):

```bash
python Module-7/collect_dataset.py --mode words --label Sorry --signer S1 --session 1 --samples 112
```

Both write to `data/sign_dataset.csv`. Then train and evaluate:

```bash
python Module-7/train_models.py
```

This runs leave-one-signer-out cross-validation for each model, writes per-class precision and recall (Supplementary Table S1) to `results/sign_language/`, and saves the three forests (100 trees each) where `sign.py` loads them. The previous models are kept in `Module-7/models/legacy/`.

---

## Reproducing the results

All scripts write to `results/<name>/`.

| Paper | Command |
|---|---|
| Table 2, Figure 12 (thermal) | Record five runs per condition with `python main.py --object-n 1 --log-temperature logs/baseline_run1.csv` and `python main.py --log-temperature logs/decimated_run1.csv` (select mode 3, run 30 minutes, let the board cool between runs), then `python evaluation/thermal_summary.py --baseline "logs/baseline_run*.csv" --decimated "logs/decimated_run*.csv" --ambient 27` |
| Table 3 (size, memory, load time, latency) | `python evaluation/profile_modules.py --source test_video.mp4` (500 frames and 50 utterances per module by default; needs an internet connection for gTTS) |
| Table 4, Figure 11 (traffic sign) | `python evaluation/eval_traffic_sign.py --data /path/to/gtsrb` (official test set, 12,630 images) |
| Table 4 (traffic light) | `python evaluation/eval_traffic_light.py --data /path/to/traffic_lights` |
| Table 4 (OCR CER/WER) | `python evaluation/eval_ocr.py --images ocr_test --manifest ocr_test/manifest.csv` (each image needs a `.txt` transcription) |
| Table 4 (face FAR/FRR) | `python evaluation/eval_face.py --manifest face_trials/manifest.csv` |
| Table 4 (captioning BLEU/CIDEr) | `python evaluation/eval_captioning.py --karpathy dataset_flickr30k.json --images flickr30k-images` |
| Table 4, Supplementary Table S1 (sign language) | `python Module-7/train_models.py` |

`python evaluation/capture_images.py --out <folder> --prefix <name>` saves camera frames for the OCR, face and traffic light test sets (space to save, q to quit). Each script's `--help` describes the expected file layout.

---

## Privacy

- No camera image, video frame or face embedding is sent off the device. In scene-description mode the current frame is written to the SD card as a temporary file and deleted as soon as its caption has been generated.
- gTTS sends each text string to be spoken (object labels, recognised names, OCR text, captions) to Google's online text-to-speech service. The audio file is deleted after it is played.
- The face database (`Module-4/images/`) stays on the device.
- The sign language dataset stores only feature vectors, from which the source frames cannot be reconstructed.

## Known limitations

- Speech needs an internet connection, because gTTS uses an online service. Without a connection the modules keep running but nothing is spoken.
- Not yet tested with visually impaired users; results so far are bench and field tests by the project team.
- Traffic light phrases describe the light ("Traffic light appears green"). They are not crossing instructions: the model cannot tell a pedestrian signal from a vehicle signal or check which way the signal faces, so it must not be used on its own as a cue to cross.
- Traffic light mode classifies the centre of the camera view. There is no separate light detector, so the light needs to be near the centre of the frame.
- The traffic sign classifier is trained on German signs (GTSRB), not Indian signs.
- Sign language is classified one frame at a time, so signs defined by movement cannot be recognised. It was validated only on signers from the project team.
- Face recognition was tested with two enrolled people, indoors, at up to 2.5 m.
- Thermal results come from one Raspberry Pi 5 at one ambient temperature (27 °C).
- Power comes from external power banks, which limits wearing time.

## Future work

- An on-device speech engine (for example eSpeak NG or Piper) so speech works without a network
- A user study with visually impaired participants, reporting task-level success
- Retraining the traffic sign classifier on Indian signs, and pedestrian-signal classes for the traffic light module
- A larger ISL vocabulary, signers from outside the team, and a short temporal window for movement-based signs
- Thermal tests across a range of ambient temperatures
- An integrated power system and a compact wearable enclosure

---

## Authors

Saksham Pundir, Prashant Kumar, Lee Thatheyus Ralph A R and Sunil Kumar Pradhan (corresponding author), School of Electronics Engineering, Vellore Institute of Technology, Chennai.

**Saksham Pundir**, B.Tech, Electronics & Communication Engineering, VIT Chennai
[LinkedIn](https://linkedin.com/in/saksham-pundir-491953245) · [GitHub](https://github.com/saksham0333)
