"""Train the three sign language random forests and evaluate them (Sections 4.1, 5.2, 6.3).

For each mode (character, numbers, words) in the dataset CSV:

  1. Leave-one-signer-out cross-validation: one fold per signer; the forest is
     trained on all other signers and tested on the held-out signer, so no
     signer is in both the training and the test fold. Fold accuracies are
     reported as mean +/- s.d. (Table 4).
  2. Per-class precision and recall from the pooled out-of-fold predictions
     (Supplementary Table S1), plus the confusion matrix.
  3. A final forest (100 trees) trained on all data and saved where sign.py
     loads it. The previous model file is kept in models/legacy/.

    python Module-7/train_models.py                       # uses data/sign_dataset.csv
    python Module-7/train_models.py --data a.csv b.csv    # several files
    python Module-7/train_models.py --no-save              # evaluate only

Results go to results/sign_language/.
"""

import argparse
import json
import os
import shutil
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import LeaveOneGroupOut

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)
from features import NUM_FEATURES, feature_names  # noqa: E402

MODEL_PATHS = {
    'character': os.path.join(BASE_DIR, "models", "characters", "RFC_MODEL_3_A_Z_modes.pkl"),
    'numbers': os.path.join(BASE_DIR, "models", "numbers", "RFC_MODEL_2_0_9_modes.pkl"),
    'words': os.path.join(BASE_DIR, "models", "words", "RFC_MODEL_WORDS_SET_1.pkl"),
}
N_TREES = 100


def make_forest(seed):
    return RandomForestClassifier(n_estimators=N_TREES, random_state=seed, n_jobs=-1)


def evaluate_mode(mode, df, seed, out_dir):
    X = df[feature_names()].to_numpy(dtype=float)
    y = df['label'].astype(str).to_numpy()
    groups = df['signer'].astype(str).to_numpy()
    signers = sorted(set(groups))
    labels = sorted(set(y))
    result = {'mode': mode, 'n_samples': int(len(y)), 'n_classes': len(labels),
              'n_signers': len(signers), 'signers': signers}

    if len(signers) < 2:
        print(f"  {mode}: only one signer, leave-one-signer-out is not possible")
        result['loso'] = None
        return result

    fold_acc = []
    y_pred = np.empty_like(y)
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        clf = make_forest(seed).fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], y_pred[test_idx])
        fold_acc.append(acc)
        print(f"  {mode}: held-out signer {groups[test_idx][0]}: accuracy {acc:.3f} (n={len(test_idx)})")

    p, r, f1, support = precision_recall_fscore_support(y, y_pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame({'mode': mode, 'class': labels, 'precision': p.round(3), 'recall': r.round(3),
                              'f1': f1.round(3), 'n': support})
    per_class.to_csv(os.path.join(out_dir, f'{mode}_per_class.csv'), index=False)
    cm = pd.DataFrame(confusion_matrix(y, y_pred, labels=labels), index=labels, columns=labels)
    cm.to_csv(os.path.join(out_dir, f'{mode}_confusion_matrix.csv'))

    result['loso'] = {
        'fold_accuracy': [round(a, 4) for a in fold_acc],
        'accuracy_mean': round(float(np.mean(fold_acc)), 4),
        'accuracy_sd': round(float(np.std(fold_acc, ddof=1)), 4) if len(fold_acc) > 1 else 0.0,
        'pooled_accuracy': round(float(accuracy_score(y, y_pred)), 4),
        'lowest_recall_classes': per_class.nsmallest(3, 'recall')[['class', 'recall']].values.tolist(),
    }
    print(f"  {mode}: LOSO accuracy {100 * result['loso']['accuracy_mean']:.1f} "
          f"+/- {100 * result['loso']['accuracy_sd']:.1f}% over {len(fold_acc)} signers")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', nargs='+', default=[os.path.join(ROOT, 'data', 'sign_dataset.csv')])
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'sign_language'))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-save', action='store_true', help='evaluate only, do not replace the models')
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p) for p in args.data], ignore_index=True)
    missing = [c for c in ['signer', 'mode', 'label'] + feature_names() if c not in df.columns]
    if missing:
        sys.exit(f"Dataset is missing {len(missing)} columns (first: {missing[:3]}). "
                 "Record with collect_dataset.py or convert with convert_legacy_dataset.py.")
    os.makedirs(args.out, exist_ok=True)

    summary = []
    for mode in ['character', 'numbers', 'words']:
        part = df[df['mode'] == mode]
        if part.empty:
            continue
        print(f"\n== {mode}: {len(part)} vectors, {part['label'].nunique()} classes, "
              f"{part['signer'].nunique()} signers, {part['session'].nunique() if 'session' in part else '?'} sessions")
        summary.append(evaluate_mode(mode, part, args.seed, args.out))

        if not args.no_save:
            path = MODEL_PATHS[mode]
            if os.path.exists(path):
                legacy_dir = os.path.join(BASE_DIR, 'models', 'legacy')
                os.makedirs(legacy_dir, exist_ok=True)
                backup = os.path.join(legacy_dir, os.path.basename(path))
                if not os.path.exists(backup):
                    shutil.copy2(path, backup)
            X = part[feature_names()].to_numpy(dtype=float)
            clf = make_forest(args.seed).fit(X, part['label'].astype(str).to_numpy())
            assert clf.n_features_in_ == NUM_FEATURES
            joblib.dump(clf, path)
            print(f"  saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")

    pc = [os.path.join(args.out, f"{s['mode']}_per_class.csv") for s in summary if s.get('loso')]
    if pc:
        pd.concat([pd.read_csv(p) for p in pc]).to_csv(os.path.join(args.out, 'supplementary_table_S1.csv'), index=False)
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
