"""Record labelled sign language feature vectors from the camera (Section 5.2).

Only the 893-value feature vector is stored (features.py), never the image, so
the source frames cannot be reconstructed (Section 5.5).

Example: record 112 samples of the word sign "Sorry" from signer S1, session 1

    python Module-7/collect_dataset.py --mode words --label Sorry --signer S1 --session 1 --samples 112

Press r to start recording, q to quit. A frame is recorded only when at least
one hand is detected. Rows are appended to the CSV (default: data/sign_dataset.csv).

Label names must match the ones sign.py uses:
  character: A ... Z, back_space, numbers, words
  numbers:   0 ... 9, back_space, character, words
  words:     Hello-Hi-Bye, Help, How, I-Love-You, Nice, No, Please, Sorry,
             Thankyou, What-is-your-name, Who, Yes, back_space, character, numbers
"""

import argparse
import csv
import os
import sys

import cv2
from cvzone.HandTrackingModule import HandDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from features import extract_features, feature_names  # noqa: E402
from sign import LABEL_MAPS, hand_coords  # noqa: E402

DEFAULT_OUT = os.path.join(BASE_DIR, '..', 'data', 'sign_dataset.csv')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', required=True, choices=list(LABEL_MAPS))
    ap.add_argument('--label', required=True)
    ap.add_argument('--signer', required=True, help='signer ID, e.g. S1')
    ap.add_argument('--session', required=True, help='session number')
    ap.add_argument('--samples', type=int, default=112)
    ap.add_argument('--out', default=DEFAULT_OUT)
    args = ap.parse_args()

    valid = set(LABEL_MAPS[args.mode].values())
    if args.label not in valid:
        sys.exit(f"Unknown label '{args.label}' for mode {args.mode}. Valid: {sorted(valid)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    f = open(args.out, 'a', newline='')
    w = csv.writer(f)
    if new_file:
        w.writerow(['signer', 'session', 'mode', 'label'] + feature_names())

    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)
    cap = cv2.VideoCapture(0)
    recording = False
    count = 0
    print("Press r to start recording, q to quit")

    while count < args.samples:
        ok, img = cap.read()
        if not ok:
            break
        hands, img = detector.findHands(img, draw=True)
        if recording and hands:
            left, right = hand_coords(hands)
            w.writerow([args.signer, args.session, args.mode, args.label] + list(extract_features(left, right)))
            count += 1
        status = f"{args.mode}/{args.label}  {count}/{args.samples}  " + ("RECORDING" if recording else "press r")
        cv2.putText(img, status, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if recording else (0, 160, 0), 2)
        cv2.imshow("Collect sign data", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            recording = True
        elif key == ord('q'):
            break

    f.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"Recorded {count} samples to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
