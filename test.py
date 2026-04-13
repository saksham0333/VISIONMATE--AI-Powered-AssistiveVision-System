from ultralytics import YOLO
import cv2
from gtts import gTTS
import os
import time

# Load lightweight YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

last_spoken = ""
last_time = 0
cooldown = 3  # seconds before speaking again

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, imgsz=416, conf=0.5)

    annotated = results[0].plot()

    if len(results[0].boxes) > 0:
        class_id = int(results[0].boxes[0].cls[0])
        label = model.names[class_id]

        current_time = time.time()

        # Speak only if different object or cooldown passed
        if label != last_spoken or (current_time - last_time) > cooldown:
            print("Detected:", label)

            tts = gTTS(text=label, lang='en')
            tts.save("temp.mp3")
            os.system("mpg123 temp.mp3")

            last_spoken = label
            last_time = current_time

    cv2.imshow("YOLO USB Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
