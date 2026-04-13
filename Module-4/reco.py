import cv2
import os
import sys
import time
import threading
import tempfile
from gtts import gTTS
from simple_facerec import SimpleFacerec

# ==============================
# PATH SETUP
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

images_path = os.path.join(BASE_DIR, "images")

# ==============================
# TEXT TO SPEECH (Lightweight)
# ==============================

speech_lock = threading.Lock()

def speak(text):
    with speech_lock:
        try:
            tts = gTTS(text=text, lang='en')
            file_path = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ).name

            tts.save(file_path)

            # Use mpg123 for lightweight playback
            os.system(f"mpg123 {file_path} > /dev/null 2>&1")

            os.remove(file_path)

        except Exception as e:
            print("Speech Error:", e)

# ==============================
# LOAD KNOWN FACES
# ==============================

sfr = SimpleFacerec()
sfr.load_encoding_images(images_path)

# ==============================
# LIVE FACE RECOGNITION
# ==============================

def live_face_recognition():

    cap = cv2.VideoCapture(0)

    last_spoken_name = ""
    last_spoken_time = 0
    cooldown = 4  # seconds between speaking same name

    frame_count = 0

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        frame_count += 1

        # Process every 5th frame (reduce CPU)
        if frame_count % 5 != 0:
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        face_locations, names = sfr.detect_known_faces(frame)

        current_time = time.time()

        for face_loc, name in zip(face_locations, names):

            top, right, bottom, left = face_loc

            # Draw bounding box
            cv2.rectangle(frame,
                          (left, top),
                          (right, bottom),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        name,
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0), 2)

            if name != "Unknown":
                if name != last_spoken_name or \
                   (current_time - last_spoken_time > cooldown):

                    print("Recognized:", name)

                    threading.Thread(
                        target=speak,
                        args=(name,),
                        daemon=True
                    ).start()

                    last_spoken_name = name
                    last_spoken_time = current_time

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    live_face_recognition()
