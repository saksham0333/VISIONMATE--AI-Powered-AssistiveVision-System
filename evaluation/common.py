"""Shared helpers for the evaluation scripts."""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, 'results')

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.ppm')


def add_path(*parts):
    """Put a repository folder on the import path."""
    path = os.path.join(ROOT, *parts)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path


def mean_sd(values):
    """(mean, sample standard deviation) of a list; s.d. is 0 for a single value."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float('nan'), float('nan')
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


def fmt(mean, sd, digits=1, scale=1.0):
    return f"{mean * scale:.{digits}f} ± {sd * scale:.{digits}f}"


def levenshtein(a, b):
    """Edit distance between two sequences (characters or words)."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def list_images(folder):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXTS))


def out_dir(name):
    path = os.path.join(RESULTS, name)
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=float)
    print("Wrote", path)
