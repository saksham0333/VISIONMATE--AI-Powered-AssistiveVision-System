import cv2
import pytesseract
import numpy as np
import threading
import tempfile
import os
import time
from gtts import gTTS

# ===============================
# TEXT TO SPEECH (Lightweight)
# ===============================

speech_lock = threading.Lock()

def speak(text):
    with speech_lock:
        try:
            tts = gTTS(text=text, lang='en')
            file_path = tempfile.NamedTemporaryFile(
                suffix=".mp3", delete=False
            ).name

            tts.save(file_path)

            # Use mpg123 for fast lightweight playback
            os.system(f"mpg123 {file_path} > /dev/null 2>&1")

            os.remove(file_path)

        except Exception as e:
            print("Speech Error:", e)


# ===============================
# OCR PREPROCESSING (PRINTED PAPER)
# ===============================

def extract_text(frame):

    # ---- Step 1: Crop center (removes background noise) ----
    h, w, _ = frame.shape
    frame = frame[int(h*0.2):int(h*0.85),
                  int(w*0.1):int(w*0.9)]

    # ---- Step 2: Resize for better OCR accuracy ----
    frame = cv2.resize(frame, None, fx=2.0, fy=2.0)

    # ---- Step 3: Convert to grayscale ----
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---- Step 4: Bilateral filter (keeps text edges sharp) ----
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # ---- Step 5: OTSU threshold (best for printed text) ----
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ---- Step 6: Slight dilation to strengthen letters ----
    kernel = np.ones((1, 1), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # ---- Step 7: Tesseract config ----
    custom_config = r'--oem 3 --psm 6'

    text = pytesseract.image_to_string(
        thresh,
        config=custom_config
    )

    # ---- Step 8: Clean text ----
    lines = text.split('\n')
    cleaned = []

    for line in lines:
        line = line.strip()
        if len(line) > 4:
            cleaned.append(line)

    return cleaned


# ===============================
# MAIN CAMERA OCR FUNCTION
# ===============================

def ocr_camera():

    cap = cv2.VideoCapture(0)
    frame_count = 0
    last_spoken_time = 0
    cooldown = 5   # seconds between readings

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        frame_count += 1

        # Process every 15th frame (reduces CPU usage)
        if frame_count % 15 == 0:

            current_time = time.time()

            if current_time - last_spoken_time > cooldown:

                text_lines = extract_text(frame)

                if text_lines:
                    sentence = " ".join(text_lines)
                    print("\nDetected Text:\n", sentence)

                    threading.Thread(
                        target=speak,
                        args=(sentence,),
                        daemon=True
                    ).start()

                    last_spoken_time = current_time

        cv2.imshow("Printed Paper OCR", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    ocr_camera()
