import os
import sys
import threading
import time

import cv2
import h5py
import numpy as np
from tensorflow.keras import layers, models

# Paths are built from this file's location, so the module works whether it is
# run directly, imported by Module-5, or started by main.py.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..', 'Module-1'))
from voice import voice

MODEL_PATH = os.path.join(BASE_DIR, 'model.h5')


def build_model():
    """The traffic light network of Section 4.6 (29,700 parameters)."""
    return models.Sequential([
        layers.Input((32, 32, 3)),
        layers.Conv2D(32, (3, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Activation('relu'),
        layers.Flatten(),
        layers.Dense(4),
        layers.Activation('softmax'),
    ])


def load_light_model(path=MODEL_PATH):
    """Load the model on any TensorFlow version.

    model.h5 was saved with an old Keras version that current TensorFlow
    (Keras 3) cannot rebuild from its stored configuration. If the normal
    loader fails, the same network is built here and the saved weights are
    copied in, layer by layer.
    """
    try:
        return models.load_model(path, compile=False)
    except Exception:
        model = build_model()
        weights = []
        with h5py.File(path, 'r') as f:
            group = f['model_weights'] if 'model_weights' in f else f
            for name in group.attrs['layer_names']:
                name = name.decode() if isinstance(name, bytes) else name
                layer = group[name]
                for wname in layer.attrs['weight_names']:
                    wname = wname.decode() if isinstance(wname, bytes) else wname
                    weights.append(np.array(layer[wname]))
        model.set_weights(weights)
        return model


model = load_light_model()

STATES = ['red', 'yellow', 'green', 'off']

# The phrases describe what the camera sees. They are not crossing
# instructions, because the model cannot tell a pedestrian signal from a
# vehicle signal or check which way the signal head faces.
PHRASES = {
    'red': 'Traffic light appears red',
    'yellow': 'Traffic light appears yellow',
    'green': 'Traffic light appears green',
    'off': 'Traffic light appears to be off',
}


def classify_frame(frame):
    """Return (state index, confidence) for a BGR image array."""
    img_resized = cv2.resize(frame, (32, 32), interpolation=cv2.INTER_LINEAR)
    img_ = np.expand_dims(np.array(img_resized), axis=0)
    probs = model.predict(img_, verbose=0)[0]
    return int(np.argmax(probs)), float(np.max(probs))


def test_an_image(file_path):
    img = cv2.imread(file_path)
    predicted_state, _ = classify_frame(img)
    return predicted_state


def traffic_lig(file_path):
    state = STATES[test_an_image(file_path)]
    print(state)
    voice(PHRASES[state])


def live_traffic_light(every_n=5, confidence_threshold=0.6, cooldown=4):
    """Traffic light mode: classify the centre of the camera view and speak
    the light state when it changes (or after the cooldown)."""
    cap = cv2.VideoCapture(0)
    frame_count = 0
    last_state = None
    last_time = 0

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % every_n == 0:
            h, w = frame.shape[:2]
            centre = frame[h // 4: 3 * h // 4, w // 3: 2 * w // 3]
            idx, conf = classify_frame(centre)
            state = STATES[idx]
            now = time.time()
            if conf > confidence_threshold and (state != last_state or now - last_time > cooldown):
                print(f"{state} ({conf:.2f})")
                threading.Thread(target=voice, args=(PHRASES[state],), daemon=True).start()
                last_state = state
                last_time = now

        cv2.imshow("Traffic light", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_traffic_light()
