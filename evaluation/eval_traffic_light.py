"""Traffic light state accuracy on held-out images (Table 4).

Held-out images (not used for training) in one folder per state:

    <data>/test/red/*.jpg  <data>/test/yellow/*.jpg  <data>/test/green/*.jpg  <data>/test/off/*.jpg

    python evaluation/eval_traffic_light.py --data /path/to/traffic_lights

The images are classified whole with Module-6/traffic_light.classify_frame,
the same function live traffic light mode and traffic sign mode use.
Outputs go to results/traffic_light/.
"""

import argparse
import os

import cv2
import numpy as np
import pandas as pd

from common import add_path, list_images, out_dir, save_json

add_path('Module-6')
from traffic_light import STATES, classify_frame  # noqa: E402


def main():
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder containing test/<state>/ images')
    ap.add_argument('--split', default='test')
    args = ap.parse_args()

    y_true, y_pred = [], []
    for k, state in enumerate(STATES):
        folder = os.path.join(args.data, args.split, state)
        if not os.path.isdir(folder):
            print("Missing", folder)
            continue
        for path in list_images(folder):
            img = cv2.imread(path)
            if img is None:
                continue
            idx, _ = classify_frame(img)
            y_true.append(k)
            y_pred.append(idx)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    acc = float((y_true == y_pred).mean())
    p, r, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(4), zero_division=0)
    res = out_dir('traffic_light')
    pd.DataFrame({'state': STATES, 'precision': p.round(4), 'recall': r.round(4), 'f1': f1.round(4), 'n': sup}) \
        .to_csv(os.path.join(res, 'per_class.csv'), index=False)
    pd.DataFrame(confusion_matrix(y_true, y_pred, labels=range(4)), index=STATES, columns=STATES) \
        .to_csv(os.path.join(res, 'confusion_matrix.csv'))
    save_json({'n': int(len(y_true)), 'accuracy': round(acc, 4)}, os.path.join(res, 'summary.json'))
    print(f"Accuracy {100 * acc:.1f}% on {len(y_true)} held-out images")


if __name__ == "__main__":
    main()
