"""Computational profile of each module on the Raspberry Pi 5 (Table 3, Section 6.2).

For each module, in its own process (so peak memory reflects one model, as
under single-model residency):

  size_mb         model files on disk
  load_s          time to import the module's libraries and load its model(s) in a
                  fresh process, which is what a mode switch in main.py costs
  frame_ms        per-frame processing time, mean +/- s.d. over --frames frames
  speech_ms       gTTS synthesis time (text -> MP3, includes the network round
                  trip), mean +/- s.d. over --utterances utterances
  end_to_end_ms   frame capture -> recognition -> speech synthesis -> start of
                  playback with mpg123, mean +/- s.d. over --utterances events
  peak_ram_mb     peak resident memory of the process

Use a recorded video for repeatable frames (or a camera index):

    python evaluation/profile_modules.py --source test_video.mp4                # all modules
    python evaluation/profile_modules.py --source 0 --modules object face       # some modules, live camera
    python evaluation/profile_modules.py --source test_video.mp4 --modules caption --frames 50

Modules: object ocr face sign_traffic light caption sign
Results: results/profile/<module>.json and results/profile/table3.csv
Speech needs an internet connection (gTTS).
"""

import argparse
import glob
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

from common import ROOT, add_path, mean_sd, out_dir

MODULES = ['object', 'ocr', 'face', 'sign_traffic', 'light', 'caption', 'sign']
LABELS = {
    'object': 'Object detection (YOLOv8n)', 'ocr': 'OCR (Tesseract LSTM)', 'face': 'Face recognition (dlib)',
    'sign_traffic': 'Traffic sign (CNN)', 'light': 'Traffic light (CNN)', 'caption': 'Scene description (CNN-LSTM)',
    'sign': 'Sign language (MediaPipe + RF)',
}


# ---------------------------------------------------------------- adapters
class Adapter:
    files = []
    extra = {}

    def load(self):
        raise NotImplementedError

    def process(self, frame):
        """Run the module on one frame and return the text it would speak (or '')."""
        raise NotImplementedError


class ObjectAdapter(Adapter):
    def load(self):
        from ultralytics import YOLO
        self.files = [os.path.join(ROOT, 'yolov8n.pt')]
        self.model = YOLO(self.files[0])
        self.model.to('cpu')
        self.model(np.zeros((480, 640, 3), np.uint8), verbose=False)   # warm-up

    def process(self, frame):
        frame = cv2.resize(frame, (640, 480))
        labels = set()
        for r in self.model(frame, verbose=False):
            for box in r.boxes:
                if float(box.conf[0]) > 0.5:
                    labels.add(self.model.names[int(box.cls[0])])
        return ', '.join(f"{l} detected" for l in sorted(labels))


class OcrAdapter(Adapter):
    def load(self):
        add_path('Module-2')
        import pytesseract
        from OCR import extract_text
        self.extract_text = extract_text
        self.extract_text(np.full((480, 640, 3), 255, np.uint8))   # starts Tesseract once
        tessdata = glob.glob('/usr/share/tesseract-ocr/*/tessdata/eng.traineddata') + \
            glob.glob('/usr/share/tessdata/eng.traineddata')
        self.files = tessdata[:1]
        self.extra = {'tesseract_version': str(pytesseract.get_tesseract_version())}

    def process(self, frame):
        return ' '.join(self.extract_text(frame))


class FaceAdapter(Adapter):
    def load(self):
        add_path('Module-4')
        import face_recognition_models
        from simple_facerec import SimpleFacerec
        self.sfr = SimpleFacerec()
        self.sfr.load_encoding_images(os.path.join(ROOT, 'Module-4', 'images'))
        # face_recognition uses the HOG detector (no file), the 5-point landmark model and the ResNet encoder
        self.files = [face_recognition_models.pose_predictor_five_point_model_location(),
                      face_recognition_models.face_recognition_model_location()]

    def process(self, frame):
        _, names = self.sfr.detect_known_faces(frame)
        return ', '.join(n for n in names if n != 'Unknown')


class TrafficSignAdapter(Adapter):
    def load(self):
        add_path()
        import traffic_sign
        self.ts = traffic_sign
        self.model = traffic_sign.load_sign_model()
        self.size = traffic_sign.input_size(self.model)
        self.files = [traffic_sign.MODEL_PATH]
        traffic_sign.classify(self.model, np.zeros((64, 64, 3), np.uint8), self.size)

    def process(self, frame):
        _, label, conf = self.ts.classify(self.model, frame, self.size)
        return label if conf > self.ts.CONFIDENCE_THRESHOLD else ''


class LightAdapter(Adapter):
    def load(self):
        add_path('Module-6')
        import traffic_light
        self.tl = traffic_light
        self.files = [traffic_light.MODEL_PATH]
        traffic_light.classify_frame(np.zeros((64, 64, 3), np.uint8))

    def process(self, frame):
        h, w = frame.shape[:2]
        idx, conf = self.tl.classify_frame(frame[h // 4: 3 * h // 4, w // 3: 2 * w // 3])
        return self.tl.PHRASES[self.tl.STATES[idx]] if conf > 0.6 else ''


class CaptionAdapter(Adapter):
    def load(self):
        add_path('Module-3')
        import Image_Captioning as ic
        self.ic = ic
        keras_cache = os.path.expanduser('~/.keras/models')
        self.files = [ic.MODEL_PATH, ic.W2I_PATH, ic.I2W_PATH] + glob.glob(os.path.join(keras_cache, 'inception_v3*'))

    def process(self, frame):
        return self.ic.caption_frame(frame)


class SignAdapter(Adapter):
    def load(self):
        add_path('Module-7')
        from cvzone.HandTrackingModule import HandDetector
        import sign
        self.sign = sign
        self.models = sign.load_models()
        self.model = self.models['words']
        self.detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
        self.files = list(sign.MODEL_PATHS.values())
        try:
            import mediapipe
            mp_dir = os.path.dirname(mediapipe.__file__)
            self.files += glob.glob(os.path.join(mp_dir, 'modules', 'hand_landmark', '*.tflite'))
            self.files += glob.glob(os.path.join(mp_dir, 'modules', 'palm_detection', '*.tflite'))
        except ImportError:
            pass
        self.landmark_ms, self.classify_ms = [], []

    def process(self, frame):
        t0 = time.perf_counter()
        hands, _ = self.detector.findHands(frame, draw=False)
        t1 = time.perf_counter()
        self.landmark_ms.append(1000 * (t1 - t0))
        if not hands:
            return ''
        left, right = self.sign.hand_coords(hands)
        label, prob = self.model.predict(left, right)
        self.classify_ms.append(1000 * (time.perf_counter() - t1))
        return label if prob >= self.sign.CONFIDENCE_THRESHOLD else ''


ADAPTERS = {'object': ObjectAdapter, 'ocr': OcrAdapter, 'face': FaceAdapter, 'sign_traffic': TrafficSignAdapter,
            'light': LightAdapter, 'caption': CaptionAdapter, 'sign': SignAdapter}


# ---------------------------------------------------------------- measurement
class FrameSource:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {source}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok and not str(self.source).isdigit():      # loop a video file
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("No frame")
        return frame


def synthesise(text):
    """gTTS text -> MP3 file; returns (path, ms)."""
    from gtts import gTTS
    fd, path = tempfile.mkstemp(suffix='.mp3')
    os.close(fd)
    t = time.perf_counter()
    gTTS(text=text, lang='en').save(path)
    return path, 1000 * (time.perf_counter() - t)


def start_playback(path, play):
    """Start mpg123 and return the process (or None)."""
    if play and shutil.which('mpg123'):
        return subprocess.Popen(['mpg123', '-q', path])
    return None


def run_child(module, args):
    adapter = ADAPTERS[module]()
    t = time.perf_counter()
    adapter.load()
    load_s = time.perf_counter() - t
    src = FrameSource(args.source)

    frame_ms, texts = [], []
    for _ in range(args.frames):
        frame = src.read()
        t = time.perf_counter()
        text = adapter.process(frame)
        frame_ms.append(1000 * (time.perf_counter() - t))
        if text:
            texts.append(text)

    speech_ms, e2e_ms = [], []
    fallback = 'This is a test of the speech output'
    for k in range(args.utterances):
        t_capture = time.perf_counter()
        frame = src.read()
        text = adapter.process(frame) or (texts[k % len(texts)] if texts else fallback)
        path, s_ms = synthesise(text)
        proc = start_playback(path, not args.no_playback)
        e2e_ms.append(1000 * (time.perf_counter() - t_capture))
        speech_ms.append(s_ms)
        if proc:
            proc.wait()
        os.remove(path)

    size_mb = sum(os.path.getsize(f) for f in adapter.files if f and os.path.exists(f)) / 1e6
    out = {
        'module': module, 'label': LABELS[module], 'source': str(args.source),
        'files': [os.path.relpath(f, ROOT) if f.startswith(ROOT) else f for f in adapter.files],
        'size_mb': round(size_mb, 2),
        'load_s': round(load_s, 2),
        'frames': len(frame_ms), 'frame_ms': mean_sd(frame_ms),
        'utterances': len(speech_ms), 'speech_ms': mean_sd(speech_ms), 'end_to_end_ms': mean_sd(e2e_ms),
        'peak_ram_mb': round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        'frames_with_output': len(texts), **adapter.extra,
    }
    if module == 'sign':
        out['landmark_ms'] = mean_sd(adapter.landmark_ms)
        out['features_and_classification_ms'] = mean_sd(adapter.classify_ms)
    json.dump(out, open(os.path.join(out_dir('profile'), f'{module}.json'), 'w'), indent=2)
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', default='0', help='video file or camera index')
    ap.add_argument('--modules', nargs='+', default=MODULES, choices=MODULES)
    ap.add_argument('--frames', type=int, default=500)
    ap.add_argument('--utterances', type=int, default=50)
    ap.add_argument('--no-playback', action='store_true', help='do not play the audio (end-to-end still ends at playback start)')
    ap.add_argument('--child', help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        run_child(args.child, args)
        return

    res = out_dir('profile')
    rows = []
    for module in args.modules:
        print(f"\n=== {LABELS[module]} ===")
        cmd = [sys.executable, os.path.abspath(__file__), '--child', module, '--source', str(args.source),
               '--frames', str(args.frames), '--utterances', str(args.utterances)]
        if args.no_playback:
            cmd.append('--no-playback')
        if subprocess.run(cmd, cwd=ROOT).returncode != 0:
            print(f"{module}: failed (see the error above)")
            continue
        r = json.load(open(os.path.join(res, f'{module}.json')))
        f = lambda m: f"{m[0]:,.0f} ± {m[1]:,.0f}"
        rows.append({'Module (model)': r['label'], 'Size on disk (MB)': r['size_mb'], 'Peak RAM (MB)': r['peak_ram_mb'],
                     'Load time on switch (s)': r['load_s'], 'Per-frame processing (ms)': f(r['frame_ms']),
                     'Speech synthesis, gTTS (ms)': f(r['speech_ms']), 'End-to-end response (ms)': f(r['end_to_end_ms'])})
    if rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(os.path.join(res, 'table3.csv'), index=False)
        print("\n" + pd.DataFrame(rows).to_string(index=False))
        print("Wrote", os.path.join(res, 'table3.csv'))


if __name__ == "__main__":
    main()
