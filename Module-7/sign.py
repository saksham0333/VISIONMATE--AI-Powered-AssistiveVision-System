import os
import joblib
import cv2
import pandas as pd
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import math
import time
from gtts import gTTS
from IPython.display import display, Audio
import string
import random
import re

def calculate_angle(A, B, C):
    """Calculate the angle between points A, B, and C.
    
    The points are given as (x, y) coordinates.
    """
    BA = (A[0] - B[0], A[1] - B[1])
    BC = (C[0] - B[0], C[1] - B[1])
    magnitude_BA = math.sqrt(BA[0] ** 2 + BA[1] ** 2)
    magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)
    
    if magnitude_BA == 0 or magnitude_BC == 0:
        return 0.0

    cosine_angle = (BA[0] * BC[0] + BA[1] * BC[1]) / (magnitude_BA * magnitude_BC)
    cosine_angle = max(-1.0, min(1.0, cosine_angle))
    angle = math.degrees(math.acos(cosine_angle))
    return angle

def generate_random_filename():
    # Generate a random string of 8 characters
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(8))

def text_to_audio(text, lang='en'):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    AUDIO_DIR = os.path.join(BASE_DIR, "audio")
    os.makedirs(AUDIO_DIR, exist_ok=True)

    filename = generate_random_filename() + '.mp3'
    file_path = os.path.join(AUDIO_DIR, filename)

    # Generate speech
    tts = gTTS(text=text, lang=lang)
    tts.save(file_path)

    # Play audio (Raspberry Pi compatible)
    os.system(f"mpg123 {file_path} > /dev/null 2>&1")

    # Delete after playing
    os.remove(file_path)

# Load the trained models
# Get current file directory (Module-7 or wherever this file is)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust if models folder is inside Module-7
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_paths = {
    'character': os.path.join(MODEL_DIR, "characters", "RFC_MODEL_3_A_Z_modes.pkl"),
    'numbers': os.path.join(MODEL_DIR, "numbers", "RFC_MODEL_2_0_9_modes.pkl"),
    'words': os.path.join(MODEL_DIR, "words", "RFC_MODEL_WORDS_SET_1.pkl")
}

# Load models
models = {mode: joblib.load(path) for mode, path in model_paths.items()}

print("Models Loaded Successfully")

# Initialize the HandDetector
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

# Define the list of key points (landmarks)
num_landmarks = 21

# Dictionary mapping numerical labels to text keys for each mode
label_maps = {
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

# Function to calculate distances and angles between key points
def calculate_features(coords):
    distances = []
    for i in range(len(coords)):
        for j in range(len(coords)):
            if i < j:
                distance = np.linalg.norm(np.array(coords[i]) - np.array(coords[j]))
                distances.append(distance)
    return distances 

# Define the confidence threshold
confidence_threshold = 0.40

# Start the video capture
cap = cv2.VideoCapture(0)

# Variables to store sentence construction
sentence = ""
previous_gesture = ""
frame_count = 0
no_hand_frames = 0
recognition_count = 0
backspace_frame_counter = 0
mode_switch_time = None  # Track the time of the last mode switch
last_recognition_time = None  # Track the time of the last gesture recognition
out_of_frame_frames = 0  # Track consecutive frames with no hands

# Define mode variables
current_mode = 'character'  # Initial mode
mode_switch_gestures = {'character': ['numbers', 'words'], 'numbers': ['character', 'words'], 'words': ['character', 'numbers']}

# Variables for word selection mode
word_selection_mode = False
word_list = []
current_word_selection_index = -1

# Maximum number of frames to allow repeated recognition
max_recognition_frames = 3
recognition_frame_count = 0

def remove_numbers_attached_to_words(sentence):
    words = sentence.split()
    new_words = []
    for word in words:
        # Use regular expression to check if there are digits attached to the word
        if re.search(r'\d', word):
            # Remove digits that are directly attached to the word
            new_word = re.sub(r'\d', '', word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def replace_multiple_spaces_with_single(sentence):
    # Use regular expression to replace multiple spaces with a single space
    return re.sub(r'\s+', ' ', sentence).strip()

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    if not success:
        break
    
    # Find hands in the image
    hands, img = detector.findHands(img, draw=True)

    # Initialize lists for storing key points coordinates
    left_hand_coords = [(0, 0)] * num_landmarks
    right_hand_coords = [(0, 0)] * num_landmarks

    # Check if any hands are detected
    if hands:
        # Reset no_hand_frames and out_of_frame_frames counts
        no_hand_frames = 0
        out_of_frame_frames = 0

        # Loop through each detected hand
        for hand in hands:
            lmList = hand["lmList"]  # List of 21 landmarks for the current hand
            handType = hand["type"]  # Type of the current hand ("Left" or "Right")

            if handType == "Left":
                left_hand_coords = [(lm[0], lm[1]) for lm in lmList]
            elif handType == "Right":
                right_hand_coords = [(lm[0], lm[1]) for lm in lmList]
    else:
        # Increment no_hand_frames count and out_of_frame_frames count
        no_hand_frames += 1
        out_of_frame_frames += 1

        # Add space if no hand detected for 10 frames in character or number mode
        if no_hand_frames >= 5 and current_mode in ['character', 'numbers']:
            sentence += ' '
            no_hand_frames = 0
        
        sentence = replace_multiple_spaces_with_single(sentence)

        # Check if out_of_frame_frames reached 30 (no hands for 30 frames)
        if out_of_frame_frames >= 30:
            # Print the current sentence
            if sentence.strip():
                sentence = ' '.join(sentence.split())  # Remove extra spaces
                print("Final Sentence:", sentence)
                # Convert the sentence to audio and play it
                text_to_audio(sentence)
                # Reset sentence and show a message on the screen
                sentence = ""
                cv2.putText(img, "Sentence Printed!", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            out_of_frame_frames = 0

    left_hand_angles = [0.0] * (num_landmarks - 2)
    right_hand_angles = [0.0] * (num_landmarks - 2)
    
    if left_hand_coords[0] != (0, 0):  # If left hand is detected
        for i in range(1, num_landmarks - 1):
            A = left_hand_coords[i - 1]
            B = left_hand_coords[i]
            C = left_hand_coords[i + 1]
            angle = calculate_angle(A, B, C)
            left_hand_angles[i - 1] = angle

    if right_hand_coords[0] != (0, 0):  # If right hand is detected
        for i in range(1, num_landmarks - 1):
            A = right_hand_coords[i - 1]
            B = right_hand_coords[i]
            C = right_hand_coords[i + 1]
            angle = calculate_angle(A, B, C)
            right_hand_angles[i - 1] = angle

    # Combine the coordinates into a single list
    angles = left_hand_angles + right_hand_angles
    combined_coords = left_hand_coords + right_hand_coords
    
    # Calculate distances and angles between key points
    features = calculate_features(combined_coords)
    features = features + angles

    # Convert features to DataFrame for prediction
    features_df = pd.DataFrame([features])

    # Make a prediction if features are available
    if not features_df.empty and hands:
        current_time = time.time()
        
        # If a mode switch has happened recently, delay gesture recognition for 1 second
        if mode_switch_time and (current_time - mode_switch_time < 1.5):
            predicted_gesture = "Waiting..."
        elif last_recognition_time and (current_time - last_recognition_time < 1.5):
            predicted_gesture = "Waiting..."
        else:
            clf = models[current_mode]
            label_map = label_maps[current_mode]

            probabilities = clf.predict_proba(features_df)
            max_prob = np.max(probabilities)
            if max_prob >= confidence_threshold:
                prediction = clf.predict(features_df)
                predicted_gesture = label_map.get(prediction[0], 'Unknown')
                last_recognition_time = current_time  # Update last recognition time

                # Handle mode switching
                if predicted_gesture in mode_switch_gestures[current_mode]:
                    current_mode = predicted_gesture
                    previous_gesture = ""
                    recognition_count = 0
                    mode_switch_time = current_time  # Record the time of the mode switch
                    recognition_frame_count = 0  # Reset recognition frame count for new mode
                    continue  # Skip the rest

                # Handle backspace gesture
                if predicted_gesture == 'back_space':
                    backspace_frame_counter += 1
                    if backspace_frame_counter == 1:
                        if current_mode == 'words':
                            words = sentence.strip().split(' ')
                            if words:
                                sentence = ' '.join(words[:-1]) + ' '
                        else:
                            if sentence:
                                sentence = sentence[:-1]
                    elif backspace_frame_counter > 1:
                        backspace_frame_counter = 0
                else:
                    backspace_frame_counter = 0

                # Add to sentence if the gesture is different from the previous one
                if predicted_gesture != previous_gesture:
                    recognition_count = 1  # Reset the recognition count for the new gesture
                    previous_gesture = predicted_gesture
                    recognition_frame_count = 1  # Reset recognition frame count
                else:
                    recognition_count += 1
                    recognition_frame_count += 1

                    # Check if we have recognized the same gesture for enough frames
                    if recognition_frame_count >= max_recognition_frames and current_mode in ['character', 'numbers']:
                        if predicted_gesture not in ['back_space', 'character', 'numbers', 'words']:
                            sentence += predicted_gesture

                # Handle adding to the sentence based on the mode
                if recognition_count == 1:
                    if current_mode == 'words':
                        if predicted_gesture not in ['back_space', 'character', 'numbers']:
                            if '-' in predicted_gesture:
                                # Split the gesture into words and switch to number mode
                                word_list = predicted_gesture.split('-')
                                word_selection_mode = True
                                current_mode = 'numbers'
                                current_word_selection_index = -1
                            else:
                                sentence += f' {predicted_gesture} '
                    else:
                        if predicted_gesture not in ['back_space', 'character', 'numbers', 'words']:
                            sentence += predicted_gesture
                else:
                    predicted_gesture = "No Gesture Detected"
            else:
                predicted_gesture = "No Gesture Detected"
    else:
        predicted_gesture = "No Gesture Detected"

    # Display the predicted gesture on the image
    cv2.putText(img, f"Gesture: {predicted_gesture}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(img, f"Mode: {current_mode}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    cv2.putText(img, f"Sentence: {sentence}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Display word selection mode options
    if word_selection_mode:
        for i, word in enumerate(word_list):
            cv2.putText(img, f"{i}: {word}", (10, 190 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 40, 0), 2)
        cv2.putText(img, f"{len(word_list)}: All words", (10, 190 + len(word_list)*30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 40, 0), 2)

        if predicted_gesture.isdigit():
            selected_index = int(predicted_gesture)
            if selected_index < len(word_list):
                sentence += f' {word_list[selected_index]} '
                x = sentence.split(' ')
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

    # Display a message if the sentence is printed
    if out_of_frame_frames >= 30 and sentence.strip() == "":
        cv2.putText(img, "Sentence Printed!", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the image with predictions
    cv2.imshow("Hand Gesture Recognition", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
cap.release()
cv2.destroyAllWindows()

# Print and play the final constructed sentence if not already printed
if sentence.strip():
    sentence = sentence.replace('-', ' ')
    sentence = ' '.join(sentence.split())  # Remove extra spaces
    text_to_audio(sentence)
    print("Final Sentence:", sentence)