"""Mode 7: Indian Sign Language recognition (Section 4.1 of the paper).

MediaPipe hand landmarks (cvzone wrapper, up to two hands) -> 893-value
geometric feature vector (features.py) -> one of three random forests
(characters, numbers, words) -> sentence, spoken when the hands leave the frame.

State logic (unchanged from the tested prototype):
  - a prediction is accepted only if its maximum class probability is >= 0.40
  - a 1.5 s debounce follows each accepted sign and each mode switch
  - 5 consecutive frames with no hand insert a word space (character/number mode)
  - 30 consecutive frames with no hand end the utterance: the sentence is spoken

The models must be trained on the same feature vector they are given. Models
retrained with train_models.py use the 893-value vector. The models shipped in
the original release expect the older 899-value vector; they are detected
automatically (n_features_in_ == 899) and fed that vector, with a warning, until
they are retrained.
"""

import os
import re
import subprocess
import sys
import tempfile
import time
import warnings

import cv2
import joblib
import numpy as np
import pandas as pd
from cvzone.HandTrackingModule import HandDetector
from gtts import gTTS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from features import (NUM_FEATURES, LEGACY_NUM_FEATURES, extract_features,  # noqa: E402
                      legacy_features)

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATHS = {
    'character': os.path.join(MODEL_DIR, "characters", "RFC_MODEL_3_A_Z_modes.pkl"),
    'numbers': os.path.join(MODEL_DIR, "numbers", "RFC_MODEL_2_0_9_modes.pkl"),
    'words': os.path.join(MODEL_DIR, "words", "RFC_MODEL_WORDS_SET_1.pkl"),
}

# Integer labels used by the models in the original release.
LABEL_MAPS = {
    'character': {
        0: 'A', 1: 'B', 2: 'back_space', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J',
        11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'numbers', 16: 'O', 17: 'P', 18: 'Q', 19: 'R', 20: 'S',
        21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'words', 26: 'X', 27: 'Y', 28: 'Z'
    },
    'numbers': {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'back_space',
        11: 'character', 12: 'words'
    },
    'words': {
        0: 'back_space', 1: 'character', 2: 'Hello-Hi-Bye', 3: 'Help', 4: 'How', 5: 'I-Love-You', 6: 'Nice',
        7: 'No', 8: 'numbers', 9: 'Please', 10: 'Sorry', 11: 'Thankyou', 12: 'What-is-your-name', 13: 'Who',
        14: 'Yes'
    }
}

CONFIDENCE_THRESHOLD = 0.40
DEBOUNCE_S = 1.5
SPACE_AFTER_NO_HAND_FRAMES = 5
END_OF_UTTERANCE_FRAMES = 30
MAX_RECOGNITION_FRAMES = 3
MODE_SWITCH_GESTURES = {'character': ['numbers', 'words'],
                        'numbers': ['character', 'words'],
                        'words': ['character', 'numbers']}


# ----------------------------------------------------------------------
# Speech
# ----------------------------------------------------------------------

def text_to_audio(text, lang='en'):
    """Speak text with gTTS + mpg123; the MP3 file is deleted after playback."""
    try:
        path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        gTTS(text=text, lang=lang).save(path)
        subprocess.run(["mpg123", "-q", path], check=False)
        os.remove(path)
    except Exception as e:
        print("Speech error:", e)


# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------

class ModeModel:
    """A random forest plus the feature builder and label decoding it needs."""

    def __init__(self, mode, path):
        self.mode = mode
        self.clf = joblib.load(path)
        n = getattr(self.clf, 'n_features_in_', None)
        if n == NUM_FEATURES:
            self.legacy = False
        elif n == LEGACY_NUM_FEATURES:
            self.legacy = True
            warnings.warn(f"{os.path.basename(path)} was trained on the old 899-value features. "
                          "It still works, but retrain it with train_models.py to use the 893-value "
                          "features described in the paper.")
        else:
            raise ValueError(f"{path}: model expects {n} features, not {NUM_FEATURES}")
        self.columns = getattr(self.clf, 'feature_names_in_', None)

    def features(self, left, right):
        return legacy_features(left, right) if self.legacy else extract_features(left, right)

    def decode(self, label):
        if isinstance(label, (str, np.str_)):
            return str(label)
        return LABEL_MAPS[self.mode].get(int(label), 'Unknown')

    def predict(self, left, right):
        """Return (label, probability) for one frame."""
        x = self.features(left, right).reshape(1, -1)
        if self.columns is not None:
            x = pd.DataFrame(x, columns=self.columns)
        proba = self.clf.predict_proba(x)[0]
        k = int(np.argmax(proba))
        return self.decode(self.clf.classes_[k]), float(proba[k])


def load_models():
    models = {mode: ModeModel(mode, path) for mode, path in MODEL_PATHS.items()}
    print("Models Loaded Successfully")
    return models


def hand_coords(hands):
    """Return (left, right) as lists of 21 (x, y) points, or None if absent."""
    left = right = None
    for hand in hands or []:
        pts = [(lm[0], lm[1]) for lm in hand["lmList"]]
        if hand["type"] == "Left":
            left = pts
        elif hand["type"] == "Right":
            right = pts
    return left, right


# ----------------------------------------------------------------------
# Sentence helpers
# ----------------------------------------------------------------------

def remove_numbers_attached_to_words(sentence):
    return ' '.join(re.sub(r'\d', '', w) if re.search(r'\d', w) else w for w in sentence.split())


def replace_multiple_spaces_with_single(sentence):
    return re.sub(r'\s+', ' ', sentence).strip()


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main():
    models = load_models()
    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1,
                            detectionCon=0.5, minTrackCon=0.5)
    cap = cv2.VideoCapture(0)

    sentence = ""
    previous_gesture = ""
    no_hand_frames = 0
    recognition_count = 0
    backspace_frame_counter = 0
    mode_switch_time = None
    last_recognition_time = None
    out_of_frame_frames = 0
    current_mode = 'character'
    word_selection_mode = False
    word_list = []
    recognition_frame_count = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detector.findHands(img, draw=True)
        left, right = hand_coords(hands)

        if hands:
            no_hand_frames = 0
            out_of_frame_frames = 0
        else:
            no_hand_frames += 1
            out_of_frame_frames += 1
            if no_hand_frames >= SPACE_AFTER_NO_HAND_FRAMES and current_mode in ['character', 'numbers']:
                sentence += ' '
                no_hand_frames = 0
            sentence = replace_multiple_spaces_with_single(sentence)
            if out_of_frame_frames >= END_OF_UTTERANCE_FRAMES:
                if sentence.strip():
                    sentence = ' '.join(sentence.split())
                    print("Final Sentence:", sentence)
                    text_to_audio(sentence)
                    sentence = ""
                    cv2.putText(img, "Sentence Printed!", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                out_of_frame_frames = 0

        predicted_gesture = "No Gesture Detected"
        if hands:
            now = time.time()
            if mode_switch_time and (now - mode_switch_time < DEBOUNCE_S):
                predicted_gesture = "Waiting..."
            elif last_recognition_time and (now - last_recognition_time < DEBOUNCE_S):
                predicted_gesture = "Waiting..."
            else:
                gesture, max_prob = models[current_mode].predict(left, right)
                if max_prob >= CONFIDENCE_THRESHOLD:
                    predicted_gesture = gesture
                    last_recognition_time = now

                    # Mode switching
                    if predicted_gesture in MODE_SWITCH_GESTURES[current_mode]:
                        current_mode = predicted_gesture
                        previous_gesture = ""
                        recognition_count = 0
                        mode_switch_time = now
                        recognition_frame_count = 0
                        continue

                    # Backspace
                    if predicted_gesture == 'back_space':
                        backspace_frame_counter += 1
                        if backspace_frame_counter == 1:
                            if current_mode == 'words':
                                words = sentence.strip().split(' ')
                                if words:
                                    sentence = ' '.join(words[:-1]) + ' '
                            elif sentence:
                                sentence = sentence[:-1]
                        elif backspace_frame_counter > 1:
                            backspace_frame_counter = 0
                    else:
                        backspace_frame_counter = 0

                    if predicted_gesture != previous_gesture:
                        recognition_count = 1
                        previous_gesture = predicted_gesture
                        recognition_frame_count = 1
                    else:
                        recognition_count += 1
                        recognition_frame_count += 1
                        if recognition_frame_count >= MAX_RECOGNITION_FRAMES and current_mode in ['character', 'numbers']:
                            if predicted_gesture not in ['back_space', 'character', 'numbers', 'words']:
                                sentence += predicted_gesture

                    if recognition_count == 1:
                        if current_mode == 'words':
                            if predicted_gesture not in ['back_space', 'character', 'numbers']:
                                if '-' in predicted_gesture:
                                    # A multi-word sign opens word selection with a number sign
                                    word_list = predicted_gesture.split('-')
                                    word_selection_mode = True
                                    current_mode = 'numbers'
                                else:
                                    sentence += f' {predicted_gesture} '
                        elif predicted_gesture not in ['back_space', 'character', 'numbers', 'words']:
                            sentence += predicted_gesture
                    else:
                        predicted_gesture = "No Gesture Detected"

        cv2.putText(img, f"Gesture: {predicted_gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(img, f"Mode: {current_mode}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
        cv2.putText(img, f"Sentence: {sentence}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        if word_selection_mode:
            for i, word in enumerate(word_list):
                cv2.putText(img, f"{i}: {word}", (10, 190 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 40, 0), 2)
            cv2.putText(img, f"{len(word_list)}: All words", (10, 190 + len(word_list) * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 40, 0), 2)
            if predicted_gesture.isdigit():
                selected_index = int(predicted_gesture)
                if selected_index < len(word_list):
                    sentence += f' {word_list[selected_index]} '
                    x = sentence.split(' ')
                    if len(x) >= 3:
                        x.pop(-3)
                    sentence = remove_numbers_attached_to_words(' '.join(x))
                    word_selection_mode = False
                    current_mode = 'words'
                elif selected_index == len(word_list):
                    sentence += ' '.join(word_list) + ' '
                    sentence = remove_numbers_attached_to_words(sentence)
                    word_selection_mode = False
                    current_mode = 'words'
                else:
                    cv2.putText(img, "Select appropriate word", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Hand Gesture Recognition", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if sentence.strip():
        sentence = ' '.join(sentence.replace('-', ' ').split())
        print("Final Sentence:", sentence)
        text_to_audio(sentence)


if __name__ == "__main__":
    main()
