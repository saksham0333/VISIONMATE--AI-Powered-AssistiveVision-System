#!/usr/bin/env python
# coding: utf-8

import os
import time
import pickle
import random
import collections

import numpy as np
import cv2
from PIL import Image

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import (
    InceptionV3,
    preprocess_input
)
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.utils import to_categorical

from gtts import gTTS
from playsound import playsound

# --------------------------------------------------
# PATH SETUP
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "Model_Weights", "model_8.h5")
W2I_PATH = os.path.join(BASE_DIR, "Storage", "word_to_idx.pkl")
I2W_PATH = os.path.join(BASE_DIR, "Storage", "idx_to_word.pkl")

# --------------------------------------------------
# LOAD CAPTION MODEL
# --------------------------------------------------

model = load_model(MODEL_PATH, compile=False)

# --------------------------------------------------
# LOAD INCEPTION FEATURE EXTRACTOR
# --------------------------------------------------

model_temp = InceptionV3(weights='imagenet', input_shape=(299, 299, 3))
model_inception = Model(
    model_temp.input,
    model_temp.layers[-2].output
)

# --------------------------------------------------
# LOAD DICTIONARIES
# --------------------------------------------------

with open(W2I_PATH, 'rb') as w2i:
    word_to_idx = pickle.load(w2i)

with open(I2W_PATH, 'rb') as i2w:
    idx_to_word = pickle.load(i2w)

# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def encode_image(img_path):
    img = preprocess_image(img_path)
    feature_vector = model_inception.predict(img, verbose=0)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector

# --------------------------------------------------
# BEAM SEARCH
# --------------------------------------------------

def beam_search(image_features, beam_index=5):
    start = [word_to_idx["startseq"]]
    max_length = 74

    start_word = [[start, 0.0]]

    while len(start_word[0][0]) < max_length:
        temp = []

        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_length)
            preds = model.predict([image_features, np.array(par_caps)], verbose=0)

            word_preds = np.argsort(preds[0])[-beam_index:]

            for w in word_preds:
                next_cap = s[0][:] + [w]
                prob = s[1] + preds[0][w]
                temp.append([next_cap, prob])

        start_word = sorted(temp, key=lambda l: l[1])[-beam_index:]

    final_caption = start_word[-1][0]
    words = []

    for idx in final_caption:
        word = idx_to_word[idx]
        if word == "endseq":
            break
        words.append(word)

    return " ".join(words[1:])

# --------------------------------------------------
# CAPTION FUNCTION
# --------------------------------------------------

def caption_this_image(img_path):
    enc = encode_image(img_path)
    caption = beam_search(enc)
    return caption

# --------------------------------------------------
# VOICE FUNCTION
# --------------------------------------------------

def voice(text):
    language = 'en'
    tts = gTTS(text=text, lang=language, slow=False)

    filename = f"output_{int(time.time())}.mp3"
    tts.save(filename)

    playsound(filename)
    os.remove(filename)

# --------------------------------------------------
# STREAM CAPTION FUNCTION
# --------------------------------------------------

def output_caption_stream(frame, count):
    img_path = os.path.join(BASE_DIR, f"frame_{count}.jpg")
    cv2.imwrite(img_path, frame)

    caption = caption_this_image(img_path)
    print("Caption:", caption)

    voice(caption)

    os.remove(img_path)

# --------------------------------------------------
# OPTIONAL: TEST WITH CAMERA
# --------------------------------------------------

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1)

        if key == ord('c'):  # Press 'c' to caption
            output_caption_stream(frame, count)
            count += 1

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
