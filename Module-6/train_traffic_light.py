"""Train the traffic light state classifier of Section 4.6.

Network: 32 x 32 x 3 input; one 32-filter 3 x 3 convolution; 2 x 2 max pooling;
ReLU; dense layer with four outputs (red, yellow, green, off); softmax.
29,700 parameters. Adam (learning rate 0.001), categorical cross-entropy.
Images are fed as raw 0-255 BGR pixels, exactly as traffic_light.classify_frame
does at run time.

Data layout (self-collected photographs, Section 5.1):

    <data>/train/red/*.jpg   <data>/train/yellow/*.jpg   <data>/train/green/*.jpg   <data>/train/off/*.jpg
    <data>/test/red/...      (held-out images, used by evaluation/eval_traffic_light.py)

    python Module-6/train_traffic_light.py --data /path/to/traffic_lights

The trained model replaces Module-6/model.h5 (and Module-5/model.h5, which the
file-based classifier uses); the previous files are kept as model_previous.h5.
"""

import argparse
import os
import shutil
import sys

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
STATES = ['red', 'yellow', 'green', 'off']   # class index order used by traffic_light.py


def build_model():
    from tensorflow.keras import layers, models
    return models.Sequential([
        layers.Input((32, 32, 3)),
        layers.Conv2D(32, (3, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Activation('relu'),
        layers.Flatten(),
        layers.Dense(4),
        layers.Activation('softmax'),
    ])


def load_folder(folder):
    X, y = [], []
    for k, state in enumerate(STATES):
        d = os.path.join(folder, state)
        if not os.path.isdir(d):
            sys.exit(f"Missing folder {d}")
        for name in sorted(os.listdir(d)):
            img = cv2.imread(os.path.join(d, name))
            if img is None:
                continue
            X.append(cv2.resize(img, (32, 32), interpolation=cv2.INTER_LINEAR))
            y.append(k)
    return np.array(X, dtype='float32'), np.array(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder containing train/<state>/ images')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-save', action='store_true')
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    tf.keras.utils.set_random_seed(args.seed)

    X, y = load_folder(os.path.join(args.data, 'train'))
    print(f"Training images: {len(X)} " + ", ".join(f"{s}={int((y == k).sum())}" for k, s in enumerate(STATES)))

    model = build_model()
    print(f"Parameters: {model.count_params():,}")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, to_categorical(y, 4), batch_size=args.batch_size, epochs=args.epochs, verbose=2)

    if not args.no_save:
        for folder in (BASE_DIR, os.path.join(ROOT, 'Module-5')):
            path = os.path.join(folder, 'model.h5')
            if os.path.exists(path):
                shutil.copy2(path, os.path.join(folder, 'model_previous.h5'))
            model.save(path)
            print("Saved", path)


if __name__ == "__main__":
    main()
