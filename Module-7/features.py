"""Geometric feature vector for sign language recognition (Section 4.1 of the paper).

Each frame gives up to two hands of 21 MediaPipe landmarks. The left and right
hands are held in fixed slots (left = points 0-20, right = points 21-41).

The feature vector has 893 values:

  1. 861 pairwise Euclidean distances between the 42 landmark points,
     C(42, 2) = 861. When a hand is absent, every distance involving one of
     its landmarks is set to zero (no distances to placeholder coordinates).
  2. 30 joint angles, 15 per hand, each computed only from landmarks on the
     same finger: three joints per finger along the chain from the wrist
     (landmark 0) to the fingertip. The angles of an absent hand are zero.
  3. 2 hand-presence flags (left, right): 1 if the hand was detected, else 0.

Order: [distances (861)] + [left angles (15), right angles (15)] + [left flag, right flag]

The models in the original release were trained on an older 899-value vector
(861 distances computed with (0, 0) placeholders for an absent hand, plus 38
angles taken at consecutive landmark indices, which can span two fingers).
`legacy_features` reproduces that vector so the old models still run, and
`convert_legacy_vector` turns a stored 899-value vector into the 893-value one
without re-recording, because every needed distance is in the old vector.
"""

import math

import numpy as np

NUM_LANDMARKS = 21
NUM_POINTS = 2 * NUM_LANDMARKS                       # 42
NUM_DISTANCES = NUM_POINTS * (NUM_POINTS - 1) // 2   # 861

# (A, B, C) triplets: the angle is measured at B. Wrist = 0.
FINGER_JOINT_TRIPLETS = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),          # thumb
    (0, 5, 6), (5, 6, 7), (6, 7, 8),          # index
    (0, 9, 10), (9, 10, 11), (10, 11, 12),    # middle
    (0, 13, 14), (13, 14, 15), (14, 15, 16),  # ring
    (0, 17, 18), (17, 18, 19), (18, 19, 20),  # little
]
ANGLES_PER_HAND = len(FINGER_JOINT_TRIPLETS)         # 15
NUM_FEATURES = NUM_DISTANCES + 2 * ANGLES_PER_HAND + 2   # 893
LEGACY_NUM_FEATURES = NUM_DISTANCES + 2 * (NUM_LANDMARKS - 2)  # 899

# Index pairs (i, j), i < j, in the order used for the distance block.
PAIRS = [(i, j) for i in range(NUM_POINTS) for j in range(i + 1, NUM_POINTS)]
_PAIR_INDEX = {p: k for k, p in enumerate(PAIRS)}
_I = np.array([p[0] for p in PAIRS])
_J = np.array([p[1] for p in PAIRS])


def feature_names():
    names = [f"d_{i}_{j}" for i, j in PAIRS]
    for side in ("L", "R"):
        names += [f"ang_{side}_{a}_{b}_{c}" for a, b, c in FINGER_JOINT_TRIPLETS]
    names += ["present_L", "present_R"]
    return names


def joint_angle(a, b, c):
    """Angle ABC in degrees, from 2-D points (0 if a segment has zero length)."""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    nba = math.hypot(*ba)
    nbc = math.hypot(*bc)
    if nba == 0 or nbc == 0:
        return 0.0
    cos = (ba[0] * bc[0] + ba[1] * bc[1]) / (nba * nbc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _hand_angles(coords):
    if coords is None:
        return [0.0] * ANGLES_PER_HAND
    return [joint_angle(coords[a], coords[b], coords[c]) for a, b, c in FINGER_JOINT_TRIPLETS]


def extract_features(left, right):
    """Build the 893-value vector.

    left, right: list of 21 (x, y) landmark coordinates, or None if that hand
    was not detected.
    """
    pts = np.zeros((NUM_POINTS, 2), dtype=float)
    present = np.zeros(NUM_POINTS, dtype=bool)
    if left is not None:
        pts[:NUM_LANDMARKS] = np.asarray(left, dtype=float)[:, :2]
        present[:NUM_LANDMARKS] = True
    if right is not None:
        pts[NUM_LANDMARKS:] = np.asarray(right, dtype=float)[:, :2]
        present[NUM_LANDMARKS:] = True

    dist = np.linalg.norm(pts[_I] - pts[_J], axis=1)
    dist[~(present[_I] & present[_J])] = 0.0   # any absent landmark -> 0

    feats = np.concatenate([
        dist,
        _hand_angles(left),
        _hand_angles(right),
        [1.0 if left is not None else 0.0, 1.0 if right is not None else 0.0],
    ])
    return feats


# ----------------------------------------------------------------------
# Legacy 899-value vector (used by the models in the original release)
# ----------------------------------------------------------------------

def legacy_features(left, right):
    """Reproduce the old 899-value vector exactly as the original sign.py built it."""
    zero = [(0, 0)] * NUM_LANDMARKS
    lc = [tuple(p[:2]) for p in left] if left is not None else zero
    rc = [tuple(p[:2]) for p in right] if right is not None else zero
    combined = np.asarray(lc + rc, dtype=float)
    dist = list(np.linalg.norm(combined[_I] - combined[_J], axis=1))

    def consecutive(coords):
        angles = [0.0] * (NUM_LANDMARKS - 2)
        if coords[0] != (0, 0):
            for i in range(1, NUM_LANDMARKS - 1):
                angles[i - 1] = joint_angle(coords[i - 1], coords[i], coords[i + 1])
        return angles

    return np.array(dist + consecutive(lc) + consecutive(rc), dtype=float)


def _angle_from_distances(ab, bc, ac):
    """Angle at B (degrees) from the three side lengths of triangle ABC."""
    if ab == 0 or bc == 0:
        return 0.0
    cos = (ab * ab + bc * bc - ac * ac) / (2 * ab * bc)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def convert_legacy_vector(vec):
    """Convert one stored 899-value vector into the 893-value vector.

    A hand counts as absent when all 210 distances between its own landmarks
    are zero (all its points were the (0, 0) placeholder).
    """
    vec = np.asarray(vec, dtype=float)
    if vec.shape[0] != LEGACY_NUM_FEATURES:
        raise ValueError(f"expected {LEGACY_NUM_FEATURES} values, got {vec.shape[0]}")
    dist = vec[:NUM_DISTANCES].copy()

    def d(i, j):
        return dist[_PAIR_INDEX[(min(i, j), max(i, j))]]

    def hand_present(offset):
        return any(d(offset + i, offset + j) > 0
                   for i in range(NUM_LANDMARKS) for j in range(i + 1, NUM_LANDMARKS))

    left_present = hand_present(0)
    right_present = hand_present(NUM_LANDMARKS)
    present = np.array([left_present] * NUM_LANDMARKS + [right_present] * NUM_LANDMARKS)
    dist_new = dist.copy()
    dist_new[~(present[_I] & present[_J])] = 0.0

    def angles(offset, is_present):
        if not is_present:
            return [0.0] * ANGLES_PER_HAND
        return [_angle_from_distances(d(offset + a, offset + b), d(offset + b, offset + c),
                                      d(offset + a, offset + c))
                for a, b, c in FINGER_JOINT_TRIPLETS]

    return np.concatenate([dist_new, angles(0, left_present), angles(NUM_LANDMARKS, right_present),
                           [float(left_present), float(right_present)]])


assert NUM_FEATURES == 893 and LEGACY_NUM_FEATURES == 899
