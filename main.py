import cv2
import sys
import time
import os

# Set Qt platform to xcb to avoid wayland warning
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Add module paths
sys.path.append('Module-1')
from voice import voice
sys.path.append('Module-2')
from OCR import ocr
sys.path.append('Module-4')
from reco import recognise
sys.path.append('Module-7')
from sign import detect_sign  # Update this with your actual function name

mode = 1
last_action_time = 0
cooldown = 3  # seconds

def cam():
    global mode, last_action_time
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process every 5th frame only
        if frame_count % 5 != 0:
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) == 27:
                break
            continue
        
        current_time = time.time()
        
        # Prevent continuous speech
        if current_time - last_action_time > cooldown:
            try:
                if mode == 1:
                    # gTTS mode - just announce something
                    voice("Camera is active")
                elif mode == 2:
                    # OCR mode
                    ocr(frame)
                elif mode == 4:
                    # Face recognition mode
                    recognise(frame)
                elif mode == 7:
                    # Sign language detection mode
                    detect_sign(frame)
            except Exception as e:
                print(f"Error in mode {mode}: {e}")
            
            last_action_time = current_time
        
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        
        if key == ord('1'):
            voice("Voice Mode Activated")
            mode = 1
        elif key == ord('2'):
            voice("OCR Mode Activated")
            mode = 2
        elif key == ord('4'):
            voice("Facial Recognition Mode Activated")
            mode = 4
        elif key == ord('7'):
            voice("Sign Language Detection Mode Activated")
            mode = 7
        elif key == 27:
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    cam()