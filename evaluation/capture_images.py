"""Save camera frames for the test sets (OCR pages and signboards, face trials, traffic lights).

    python evaluation/capture_images.py --out data/ocr_test --prefix page
    python evaluation/capture_images.py --out data/face_trials --prefix g_saksham_1.5m

Press space to save the current frame, q to quit. Frames are saved as
<prefix>_<number>.jpg. Only frames you choose to save are stored.
"""

import argparse
import os

import cv2


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True)
    ap.add_argument('--prefix', default='img')
    ap.add_argument('--camera', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    existing = [f for f in os.listdir(args.out) if f.startswith(args.prefix + '_')]
    count = len(existing)
    cap = cv2.VideoCapture(args.camera)
    print("space = save, q = quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        shown = frame.copy()
        cv2.putText(shown, f"{args.prefix}: {count} saved", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.imshow("Capture", shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            count += 1
            path = os.path.join(args.out, f"{args.prefix}_{count:03d}.jpg")
            cv2.imwrite(path, frame)
            print("Saved", path)
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
