import cv2
import numpy as np
import threading
import tempfile
import os
from gtts import gTTS
from ultralytics import YOLO
import torch

# ---------------- DEVICE SETUP ---------------- #

device = "cpu" 
print(f"Using device: {device}")

# Load YOLO model
model = YOLO("yolov8n.pt")
model.to(device)

# Prevent overlapping speech
speech_lock = threading.Lock()
last_spoken_objects = None


# ---------------- TEXT TO SPEECH ---------------- #

def text_to_speech(text):
    global speech_lock

    with speech_lock:
        tts = gTTS(text=text, lang='en')
        file_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        tts.save(file_path)

        # Use mpg123 instead of pygame (lighter)
        os.system(f"mpg123 {file_path} > /dev/null 2>&1")
        os.remove(file_path)


# ---------------- OBJECT DETECTION ---------------- #

def object_detection():
    global last_spoken_objects

    cap = cv2.VideoCapture(0)  
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        frame_count += 1
        if frame_count % 3 != 0:
            continue

        frame = cv2.resize(frame, (640, 480))

        results = model(frame, verbose=False)

        detected_objects = set()

        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]

                if conf > 0.5:
                    detected_objects.add(label)

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 2)

        detected_objects_tuple = tuple(sorted(detected_objects))

        if detected_objects_tuple and detected_objects_tuple != last_spoken_objects:
            last_spoken_objects = detected_objects_tuple

            speech_message = ", ".join(
                [f"{obj} detected" for obj in detected_objects_tuple]
            )

            threading.Thread(
                target=text_to_speech,
                args=(speech_message,),
                daemon=True
            ).start()

        cv2.imshow("YOLOv8 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    object_detection()
