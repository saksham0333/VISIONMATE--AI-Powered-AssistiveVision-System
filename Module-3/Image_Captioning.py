#!/usr/bin/env python
# coding: utf-8
"""Mode 1: scene description (Section 4.7 of the paper).

InceptionV3 (ImageNet, penultimate layer, 2048-d) encodes the frame; an LSTM
decoder (vocabulary 5,121 tokens, 200-d embedding, 256 units, up to 74 tokens)
generates the caption by beam search with a beam width of 5. Captioning runs
only when the user asks for it (press c in the camera window).

The current frame is written to a temporary image file next to this script and
deleted as soon as its caption has been generated (Section 5.5).
"""

import os
import pickle
import subprocess
import tempfile

import cv2
import h5py
import numpy as np
from gtts import gTTS
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model, load_model

try:
    from tensorflow.keras.utils import load_img, img_to_array, pad_sequences
except ImportError:  # older TensorFlow
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Model_Weights", "model_8.h5")
W2I_PATH = os.path.join(BASE_DIR, "Storage", "word_to_idx.pkl")
I2W_PATH = os.path.join(BASE_DIR, "Storage", "idx_to_word.pkl")

MAX_LENGTH = 74        # caption length limit (tokens)
VOCAB_SIZE = 5121      # 5,120 words + padding index 0
EMBEDDING_DIM = 200
BEAM_WIDTH = 5

# --------------------------------------------------
# CAPTION MODEL
# --------------------------------------------------


def build_caption_model():
    """The decoder of Section 4.7, with the layer names used in model_8.h5."""
    image_in = layers.Input(shape=(2048,), name="input_2")
    x = layers.Dropout(0.5, name="dropout")(image_in)
    x = layers.Dense(256, activation="relu", name="dense")(x)

    caption_in = layers.Input(shape=(MAX_LENGTH,), name="input_3")
    y = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True, name="embedding")(caption_in)
    y = layers.Dropout(0.5, name="dropout_1")(y)
    y = layers.LSTM(256, name="lstm")(y)

    z = layers.add([x, y], name="add")
    z = layers.Dense(256, activation="relu", name="dense_1")(z)
    out = layers.Dense(VOCAB_SIZE, activation="softmax", name="dense_2")(z)
    return Model(inputs=[image_in, caption_in], outputs=out)


def load_caption_model(path=MODEL_PATH):
    """Load model_8.h5 on any TensorFlow version.

    The file was saved with Keras 2.8. Current TensorFlow (Keras 3) cannot
    rebuild it from its stored configuration, so if the normal loader fails
    the same network is built here and the saved weights are copied in by
    layer name.
    """
    try:
        return load_model(path, compile=False)
    except Exception:
        model = build_caption_model()
        with h5py.File(path, "r") as f:
            group = f["model_weights"]
            for layer in model.layers:
                if layer.name not in group:
                    continue
                g = group[layer.name]
                names = [n.decode() if isinstance(n, bytes) else n for n in g.attrs["weight_names"]]
                if names:
                    layer.set_weights([np.array(g[n]) for n in names])
        return model


model = load_caption_model()

# Image encoder: InceptionV3 without its classification layer
model_temp = InceptionV3(weights="imagenet", input_shape=(299, 299, 3))
model_inception = Model(model_temp.input, model_temp.layers[-2].output)

with open(W2I_PATH, "rb") as f:
    word_to_idx = pickle.load(f)
with open(I2W_PATH, "rb") as f:
    idx_to_word = pickle.load(f)

START = word_to_idx["startseq"]
END = word_to_idx["endseq"]

# --------------------------------------------------
# ENCODER
# --------------------------------------------------


def preprocess_image(img_path):
    img = load_img(img_path, target_size=(299, 299))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return preprocess_input(x)


def encode_image(img_path):
    feature_vector = model_inception.predict(preprocess_image(img_path), verbose=0)
    return feature_vector.reshape(1, feature_vector.shape[1])

# --------------------------------------------------
# BEAM SEARCH
# --------------------------------------------------


def beam_search(image_features, beam_width=BEAM_WIDTH, max_length=MAX_LENGTH):
    """Beam search over the decoder.

    Each beam keeps its summed log-probability. A beam is finished when it
    emits the end token or reaches the length limit. The search stops when
    beam_width captions have finished (or no beams remain), and the finished
    caption with the highest length-normalised log-probability is returned.
    All live beams are scored in one batched call per step.
    """
    beams = [([START], 0.0)]
    finished = []

    while beams and len(finished) < beam_width:
        seqs = pad_sequences([seq for seq, _ in beams], maxlen=max_length)
        feats = np.repeat(image_features, len(beams), axis=0)
        preds = model.predict([feats, seqs], verbose=0)

        candidates = []
        for (seq, score), p in zip(beams, preds):
            for w in np.argsort(p)[-beam_width:]:
                candidates.append((seq + [int(w)], score + float(np.log(p[w] + 1e-12))))
        candidates.sort(key=lambda c: c[1], reverse=True)

        beams = []
        for seq, score in candidates[:beam_width]:
            if seq[-1] == END or len(seq) >= max_length:
                finished.append((seq, score))
            else:
                beams.append((seq, score))

    pool = finished or beams
    best_seq, _ = max(pool, key=lambda c: c[1] / len(c[0]))

    words = []
    for idx in best_seq[1:]:
        if idx == END:
            break
        words.append(idx_to_word[idx])
    return " ".join(words)


def caption_this_image(img_path):
    return beam_search(encode_image(img_path))


def caption_frame(frame):
    """Caption a camera frame (BGR). The frame is saved as a temporary file and deleted."""
    fd, img_path = tempfile.mkstemp(suffix=".jpg", dir=BASE_DIR)
    os.close(fd)
    try:
        cv2.imwrite(img_path, frame)
        return caption_this_image(img_path)
    finally:
        os.remove(img_path)

# --------------------------------------------------
# SPEECH (gTTS + mpg123, file deleted after playback)
# --------------------------------------------------


def voice(text):
    try:
        path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
        gTTS(text=text, lang="en", slow=False).save(path)
        subprocess.run(["mpg123", "-q", path], check=False)
        os.remove(path)
    except Exception as e:
        print("Speech error:", e)


def output_caption_stream(frame):
    caption = caption_frame(frame)
    print("Caption:", caption)
    voice(caption)

# --------------------------------------------------
# CAMERA LOOP
# --------------------------------------------------


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    print("Press 'c' to describe the scene, 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Camera", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            output_caption_stream(frame)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
