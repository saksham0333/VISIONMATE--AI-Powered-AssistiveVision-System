import cv2
import numpy as np
import tensorflow as tf
import time
import threading
import tempfile
import os
from gtts import gTTS

# =========================
# CONFIG
# =========================

MODEL_PATH = "traffic_model.keras"   # your trained model
IMG_SIZE = 32
CONFIDENCE_THRESHOLD = 0.50
COOLDOWN = 4

# =========================
# LOAD MODEL
# =========================

model = tf.keras.models.load_model(MODEL_PATH)
print("Model Loaded Successfully")
print("Input Shape:", model.input_shape)

# =========================
# GTSRB 43 CLASS LABELS
# =========================

class_names = [
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

# =========================
# SPEECH FUNCTION
# =========================

speech_lock = threading.Lock()

def speak(text):
    with speech_lock:
        try:
            tts = gTTS(text=text, lang="en")
            file_path = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ).name
            tts.save(file_path)
            os.system(f"mpg123 {file_path} > /dev/null 2>&1")
            os.remove(file_path)
        except Exception as e:
            print("Speech error:", e)

# =========================
# PREPROCESS
# =========================

def preprocess(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# MAIN LOOP
# =========================

cap = cv2.VideoCapture(0)

last_spoken = ""
last_time = 0

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed = preprocess(frame)

    predictions = model.predict(processed, verbose=0)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)

    label = class_names[class_index]

    current_time = time.time()

    if confidence > CONFIDENCE_THRESHOLD:
        if label != last_spoken or (current_time - last_time > COOLDOWN):
            print(f"Detected: {label} ({confidence:.2f})")

            threading.Thread(
                target=speak,
                args=(label,),
                daemon=True
            ).start()

            last_spoken = label
            last_time = current_time

    # Display
    cv2.putText(frame,
                f"{label} ({confidence:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2)

    cv2.imshow("Live GTSRB Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
