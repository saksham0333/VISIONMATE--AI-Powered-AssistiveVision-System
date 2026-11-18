"""Classify a traffic sign in an image file.

Uses the same model and preprocessing as live traffic sign mode
(traffic_sign.py). If the sign is "Traffic signals", the traffic light
classifier is run on the same image and its description is printed and spoken.

    python Module-5/classification.py path/to/image.jpg
"""

import os
import sys

import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, BASE_DIR)

from traffic_sign import TRAFFIC_SIGNALS_CLASS, classify, load_sign_model  # noqa: E402
from traffic_light import traffic_lig  # noqa: E402

model = load_sign_model()


def classify_file(file_path):
    frame = cv2.imread(file_path)
    if frame is None:
        raise FileNotFoundError(file_path)
    class_index, label, confidence = classify(model, frame)
    print(f"{label} ({confidence:.2f})")
    if class_index == TRAFFIC_SIGNALS_CLASS:
        traffic_lig(file_path)
    return label


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    classify_file(sys.argv[1])
