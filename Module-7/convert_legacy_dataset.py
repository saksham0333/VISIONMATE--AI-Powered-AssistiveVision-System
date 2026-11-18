"""Convert stored 899-value sign language vectors to the 893-value vector.

The models in the original release were trained on 899-value vectors (861
distances with (0, 0) placeholders for an absent hand + 38 angles at
consecutive landmark indices). Every distance needed for the new vector is
already in the old one, and the within-finger angles follow from those
distances, so the old recordings can be converted exactly without recording
again (see features.convert_legacy_vector).

The input CSV must contain the 899 feature columns, in their original order,
and a label column. Signer and session can come from columns or be given once
for the whole file.

Examples

    # old file with columns 0..898 and a 'label' column, all from signer S1 session 1
    python Module-7/convert_legacy_dataset.py old_words.csv --mode words --label-col label \
        --signer S1 --session 1

    # old file that already has signer/session columns
    python Module-7/convert_legacy_dataset.py old_chars.csv --mode character --label-col label \
        --signer-col signer --session-col session

    # integer labels (as in the original models) are mapped to names with sign.LABEL_MAPS
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from features import LEGACY_NUM_FEATURES, convert_legacy_vector, feature_names  # noqa: E402

LABEL_MAPS = {
    'character': {0: 'A', 1: 'B', 2: 'back_space', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I',
                  10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N', 15: 'numbers', 16: 'O', 17: 'P', 18: 'Q',
                  19: 'R', 20: 'S', 21: 'T', 22: 'U', 23: 'V', 24: 'W', 25: 'words', 26: 'X', 27: 'Y', 28: 'Z'},
    'numbers': {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
                10: 'back_space', 11: 'character', 12: 'words'},
    'words': {0: 'back_space', 1: 'character', 2: 'Hello-Hi-Bye', 3: 'Help', 4: 'How', 5: 'I-Love-You',
              6: 'Nice', 7: 'No', 8: 'numbers', 9: 'Please', 10: 'Sorry', 11: 'Thankyou',
              12: 'What-is-your-name', 13: 'Who', 14: 'Yes'},
}
DEFAULT_OUT = os.path.join(BASE_DIR, '..', 'data', 'sign_dataset.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input')
    ap.add_argument('--mode', required=True, choices=list(LABEL_MAPS))
    ap.add_argument('--label-col', required=True)
    ap.add_argument('--signer-col')
    ap.add_argument('--session-col')
    ap.add_argument('--signer', help='signer ID for every row (if there is no signer column)')
    ap.add_argument('--session', help='session for every row (if there is no session column)')
    ap.add_argument('--feature-cols', help='first:last column names of the 899 features (default: '
                                           'the 899 numeric columns other than the label/signer/session)')
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    meta = {c for c in (args.label_col, args.signer_col, args.session_col) if c}
    if args.feature_cols:
        first, last = args.feature_cols.split(':')
        cols = list(df.loc[:, first:last].columns)
    else:
        cols = [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols) != LEGACY_NUM_FEATURES:
        sys.exit(f"Found {len(cols)} feature columns, expected {LEGACY_NUM_FEATURES}. Use --feature-cols.")

    if not args.signer_col and not args.signer:
        sys.exit("Give --signer-col or --signer: leave-one-signer-out validation needs the signer of every row.")

    labels = df[args.label_col]
    if pd.api.types.is_integer_dtype(labels):
        labels = labels.map(LABEL_MAPS[args.mode])
        if labels.isna().any():
            sys.exit("Some integer labels are not in the label map for this mode.")

    X = np.vstack([convert_legacy_vector(v) for v in df[cols].to_numpy(dtype=float)])
    out = pd.DataFrame(X, columns=feature_names())
    out.insert(0, 'label', labels.astype(str).values)
    out.insert(0, 'mode', args.mode)
    out.insert(0, 'session', df[args.session_col].values if args.session_col else (args.session or ''))
    out.insert(0, 'signer', df[args.signer_col].values if args.signer_col else args.signer)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    exists = os.path.exists(args.out)
    out.to_csv(args.out, mode='a' if exists else 'w', header=not exists, index=False)
    print(f"Converted {len(out)} rows -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
