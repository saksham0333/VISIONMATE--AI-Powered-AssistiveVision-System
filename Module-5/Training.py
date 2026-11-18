"""Train the traffic sign classifier of Section 4.5 on GTSRB (Sections 5.1 and 5.4).

Network: 30 x 30 x 3 input; three convolutional blocks (32, 64, 128 filters,
3 x 3, ReLU, batch normalisation, 2 x 2 max pooling); flatten; dense 128 ReLU;
dropout 0.5; softmax over 43 classes. 165,355 parameters.
Training: Adam, categorical cross-entropy, batch size 32, 15 epochs.

Data: the Kaggle copy of GTSRB, with Train/<ClassId>/*.png. File names are
<class>_<track>_<frame>.png; frames of one track show the same physical sign.

    # random 80:20 split of the training set (the 96.5% figure in the paper)
    python Module-5/Training.py --data /path/to/gtsrb --split random

    # track-wise split: whole tracks are held out, so no sign is in both sets
    python Module-5/Training.py --data /path/to/gtsrb --split track

The model is saved without optimizer state to traffic_model.keras in the
repository root (use --out to change), and the training curves (Figure 10)
to results/traffic_sign/. Evaluate on the official test set with
evaluation/eval_traffic_sign.py.
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

NUM_CLASSES = 43
IMG_SIZE = 30


def build_model():
    from tensorflow.keras import layers, models
    return models.Sequential([
        layers.Input((IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'), layers.BatchNormalization(), layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])


def load_train_set(train_dir):
    """Images as uint8 (BGR, 30 x 30), labels, and a track id per image."""
    X, y, tracks = [], [], []
    for class_id in range(NUM_CLASSES):
        folder = os.path.join(train_dir, str(class_id))
        if not os.path.isdir(folder):
            sys.exit(f"Missing folder {folder}")
        for name in sorted(os.listdir(folder)):
            img = cv2.imread(os.path.join(folder, name))
            if img is None:
                continue
            X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
            y.append(class_id)
            parts = os.path.splitext(name)[0].split('_')
            tracks.append(f"{class_id}_{parts[1]}" if len(parts) >= 3 else f"{class_id}_{name}")
    return np.array(X), np.array(y), np.array(tracks)


def save_without_optimizer(model, path):
    """Save the trained weights in a fresh, uncompiled copy (no optimizer state)."""
    clean = build_model()
    clean.set_weights(model.get_weights())
    clean.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='GTSRB root folder (contains Train/)')
    ap.add_argument('--split', choices=['random', 'track'], default='random')
    ap.add_argument('--val-fraction', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=os.path.join(ROOT, 'traffic_model.keras'))
    ap.add_argument('--results', default=os.path.join(ROOT, 'results', 'traffic_sign'))
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    from tensorflow.keras.utils import to_categorical

    tf.keras.utils.set_random_seed(args.seed)
    X, y, tracks = load_train_set(os.path.join(args.data, 'Train'))
    print(f"Loaded {len(X)} training images from {len(set(tracks))} tracks")

    idx = np.arange(len(X))
    if args.split == 'random':
        tr, va = train_test_split(idx, test_size=args.val_fraction, random_state=args.seed)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
        tr, va = next(gss.split(idx, y, groups=tracks))
        assert not set(tracks[tr]) & set(tracks[va])

    X = X.astype('float32') / 255.0   # same scaling as traffic_sign.preprocess
    Y = to_categorical(y, NUM_CLASSES)

    model = build_model()
    print(f"Parameters: {model.count_params():,}")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    hist = model.fit(X[tr], Y[tr], batch_size=args.batch_size, epochs=args.epochs,
                     validation_data=(X[va], Y[va]), verbose=2)

    save_without_optimizer(model, args.out)
    print(f"Saved {args.out} ({os.path.getsize(args.out) / 1e6:.2f} MB)")

    os.makedirs(args.results, exist_ok=True)
    h = hist.history
    with open(os.path.join(args.results, f'training_history_{args.split}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'accuracy', 'val_accuracy', 'loss', 'val_loss'])
        for e in range(len(h['loss'])):
            w.writerow([e + 1, h['accuracy'][e], h['val_accuracy'][e], h['loss'][e], h['val_loss'][e]])
    print(f"Final validation accuracy ({args.split} split): {100 * h['val_accuracy'][-1]:.1f}%")

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        ep = range(1, len(h['loss']) + 1)
        ax[0].plot(ep, h['accuracy'], label='training'); ax[0].plot(ep, h['val_accuracy'], label='validation')
        ax[0].set_title('(a) Accuracy'); ax[0].set_xlabel('Epoch'); ax[0].legend()
        ax[1].plot(ep, h['loss'], label='training'); ax[1].plot(ep, h['val_loss'], label='validation')
        ax[1].set_title('(b) Categorical cross-entropy loss'); ax[1].set_xlabel('Epoch'); ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.results, f'figure10_training_curves_{args.split}.png'), dpi=220)
    except Exception as e:
        print("Plot skipped:", e)


if __name__ == "__main__":
    main()
