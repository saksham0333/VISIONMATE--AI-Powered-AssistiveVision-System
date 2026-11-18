"""Traffic sign accuracy on the official GTSRB test set (Table 4, Figure 11).

Uses the deployed model (traffic_model.keras) with exactly the preprocessing of
live traffic sign mode (traffic_sign.preprocess). The official test set
(12,630 images) is used neither for training nor for model selection.

Data: the Kaggle copy of GTSRB, with Test.csv (columns include ClassId, Path)
and the Test/ folder next to it.

    python evaluation/eval_traffic_sign.py --data /path/to/gtsrb

Outputs (results/traffic_sign/):
    summary.json                    accuracy, macro F1, n
    per_class.csv                   precision, recall, F1 and n for the 43 classes
    confusion_matrix_43.csv         raw counts
    confusion_supercategories.csv   row-normalised %, six GTSRB super-categories
    figure11_confusion.png          Figure 11
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd

from common import add_path, fmt, out_dir, save_json  # noqa: F401

add_path()
from traffic_sign import CLASS_NAMES, MODEL_PATH, input_size, load_sign_model, preprocess  # noqa: E402

# The six super-categories of the GTSRB benchmark (Stallkamp et al., 2012), by ClassId.
SUPERCATEGORIES = [
    ('Speed limit', [0, 1, 2, 3, 4, 5, 7, 8]),
    ('Other prohibitory', [9, 10, 15, 16]),
    ('Derestriction', [6, 32, 41, 42]),
    ('Mandatory', [33, 34, 35, 36, 37, 38, 39, 40]),
    ('Danger / warning', [11] + list(range(18, 32))),
    ('Unique', [12, 13, 14, 17]),
]


def supercategory_matrix(y_true, y_pred):
    to_super = {}
    for k, (_, ids) in enumerate(SUPERCATEGORIES):
        for i in ids:
            to_super[i] = k
    n = len(SUPERCATEGORIES)
    m = np.zeros((n, n))
    for t, p in zip(y_true, y_pred):
        m[to_super[t], to_super[p]] += 1
    return 100 * m / m.sum(axis=1, keepdims=True)


def plot_figure11(m, accuracy, macro_f1, n, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [f"{name}\n({len(ids)})" for name, ids in SUPERCATEGORIES]
    fig, ax = plt.subplots(figsize=(6.5, 5.7))
    im = ax.imshow(m, cmap='Blues', vmin=0, vmax=100)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{m[i, j]:.1f}", ha='center', va='center', fontsize=7.6,
                    color='white' if m[i, j] > 50 else '#222222', fontweight='bold' if i == j else 'normal')
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=38, ha='right', rotation_mode='anchor', fontsize=6.6)
    ax.set_yticklabels(labels, fontsize=6.6)
    ax.set_xlabel('Predicted super-category'); ax.set_ylabel('True super-category')
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label('Row-normalised prediction share (%)', fontsize=7.2)
    ax.set_title('Traffic sign classifier — confusion over GTSRB super-categories\n'
                 f'Official GTSRB test set, n = {n:,}; 43-class accuracy {100 * accuracy:.1f}%, macro F1 {macro_f1:.2f}',
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    print("Wrote", path)


def main():
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='GTSRB root folder containing Test.csv')
    ap.add_argument('--model', default=MODEL_PATH)
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()

    test = pd.read_csv(os.path.join(args.data, 'Test.csv'))
    model = load_sign_model(args.model)
    size = input_size(model)

    y_true, y_pred = [], []
    for start in range(0, len(test), args.batch_size):
        rows = test.iloc[start:start + args.batch_size]
        batch = []
        for path in rows['Path']:
            img = cv2.imread(os.path.join(args.data, path))
            if img is None:
                raise FileNotFoundError(os.path.join(args.data, path))
            batch.append(preprocess(img, size)[0])
        pred = model.predict(np.stack(batch), verbose=0).argmax(axis=1)
        y_true += rows['ClassId'].astype(int).tolist()
        y_pred += pred.tolist()
        print(f"\r{len(y_pred)}/{len(test)}", end='')
    print()

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(43), zero_division=0)

    res = out_dir('traffic_sign')
    pd.DataFrame({'class_id': range(43), 'name': CLASS_NAMES, 'precision': p.round(4), 'recall': r.round(4),
                  'f1': f1.round(4), 'n': sup}).to_csv(os.path.join(res, 'per_class.csv'), index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=range(43))).to_csv(os.path.join(res, 'confusion_matrix_43.csv'))
    m = supercategory_matrix(y_true, y_pred)
    pd.DataFrame(m.round(2), index=[s for s, _ in SUPERCATEGORIES],
                 columns=[s for s, _ in SUPERCATEGORIES]).to_csv(os.path.join(res, 'confusion_supercategories.csv'))
    save_json({'test_set': 'official GTSRB test set', 'n': int(len(y_true)), 'accuracy': round(float(acc), 4),
               'macro_f1': round(float(macro_f1), 4), 'model': os.path.basename(args.model)},
              os.path.join(res, 'summary.json'))
    plot_figure11(m, acc, macro_f1, len(y_true), os.path.join(res, 'figure11_confusion.png'))
    print(f"Accuracy {100 * acc:.1f}%  macro F1 {macro_f1:.2f}  (n = {len(y_true)})")


if __name__ == "__main__":
    main()
