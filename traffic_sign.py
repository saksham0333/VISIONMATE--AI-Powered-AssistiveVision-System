"""Mode 5: traffic sign recognition (Section 4.5 of the paper).

CNN trained on the 43-class GTSRB dataset (30 x 30 x 3 input, three
convolutional blocks, 165,355 parameters). When the classifier sees the
"Traffic signals" warning sign, the traffic light state classifier (Module-6)
is run on the same frame and its description is added to the announcement.

Run from the repository root:  python traffic_sign.py
"""

import os
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np
import tensorflow as tf
from gtts import gTTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Traffic light state classifier (Module-6). It is called when the sign
# classifier sees the "Traffic signals" warning sign.
sys.path.append(os.path.join(BASE_DIR, "Module-6"))
try:
    from traffic_light import classify_frame as classify_light, STATES as LIGHT_STATES, PHRASES as LIGHT_PHRASES
except Exception as e:
    classify_light = None
    print("Traffic light classifier not loaded:", e)

# =========================
# CONFIG
# =========================

MODEL_PATH = os.path.join(BASE_DIR, "traffic_model.keras")   # the network of Section 4.5
CONFIDENCE_THRESHOLD = 0.50
COOLDOWN = 4   # seconds before the same sign is announced again

# GTSRB 43 class labels (ClassId 0-42)
CLASS_NAMES = [
    "Speed limit 20 km/h", "Speed limit 30 km/h", "Speed limit 50 km/h",
    "Speed limit 60 km/h", "Speed limit 70 km/h", "Speed limit 80 km/h",
    "End of speed limit 80 km/h", "Speed limit 100 km/h",
    "Speed limit 120 km/h", "No passing",
    "No passing for vehicles over 3.5 tons",
    "Right-of-way at intersection", "Priority road",
    "Yield", "Stop", "No vehicles",
    "Vehicles over 3.5 tons prohibited",
    "No entry", "General caution",
    "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals",
    "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow",
    "Wild animals crossing", "End of all speed and passing limits",
    "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right",
    "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory",
    "End of no passing", "End of no passing for vehicles over 3.5 tons"
]
TRAFFIC_SIGNALS_CLASS = 26


# =========================
# MODEL
# =========================

def load_sign_model(path=MODEL_PATH):
    model = tf.keras.models.load_model(path, compile=False)
    return model


def input_size(model):
    """Input height/width the model was built for (30 for the network of Section 4.5)."""
    return int(model.input_shape[1])


def preprocess(frame, size):
    """BGR frame -> (1, size, size, 3) float array scaled to [0, 1].

    The same preprocessing is used for training (Module-5/Training.py) and
    evaluation (evaluation/eval_traffic_sign.py).
    """
    img = cv2.resize(frame, (size, size))
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def classify(model, frame, size=None):
    """Return (class_index, label, confidence) for one BGR frame."""
    size = size or input_size(model)
    predictions = model.predict(preprocess(frame, size), verbose=0)
    class_index = int(np.argmax(predictions))
    return class_index, CLASS_NAMES[class_index], float(np.max(predictions))


# =========================
# SPEECH
# =========================

speech_lock = threading.Lock()


def speak(text):
    with speech_lock:
        try:
            tts = gTTS(text=text, lang="en")
            file_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
            tts.save(file_path)
            subprocess.run(["mpg123", "-q", file_path], check=False)
            os.remove(file_path)
        except Exception as e:
            print("Speech error:", e)


# =========================
# MAIN LOOP
# =========================

def main():
    model = load_sign_model()
    size = input_size(model)
    print("Model Loaded Successfully")
    print("Input Shape:", model.input_shape)

    cap = cv2.VideoCapture(0)
    last_spoken = ""
    last_time = 0
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        class_index, label, confidence = classify(model, frame, size)
        current_time = time.time()

        if confidence > CONFIDENCE_THRESHOLD:
            if label != last_spoken or (current_time - last_time > COOLDOWN):
                print(f"Detected: {label} ({confidence:.2f})")

                text = label
                if class_index == TRAFFIC_SIGNALS_CLASS and classify_light is not None:
                    light_index, light_conf = classify_light(frame)
                    text = f"{label}. {LIGHT_PHRASES[LIGHT_STATES[light_index]]}"
                    print(f"Traffic light: {LIGHT_STATES[light_index]} ({light_conf:.2f})")

                threading.Thread(target=speak, args=(text,), daemon=True).start()
                last_spoken = label
                last_time = current_time

        cv2.putText(frame, f"{label} ({confidence:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Live GTSRB Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
