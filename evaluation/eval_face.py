"""Face recognition false accept and false reject rates (Sections 5.3 and 6.3, Table 4).

Protocol: genuine attempts are frames of an enrolled person; impostor attempts
are frames of people who are not enrolled. Record the attempts with
evaluation/capture_images.py and list them in a manifest CSV:

    image,identity,distance_m
    trials/g001.jpg,saksham,0.5
    trials/g040.jpg,prashant,2.5
    trials/i001.jpg,impostor,1.0

identity is the enrolled name (the file name in Module-4/images without its
extension) for a genuine attempt, or "impostor" for a person not enrolled.

    python evaluation/eval_face.py --manifest trials/manifest.csv

Each frame goes through the live pipeline (Module-4/simple_facerec: quarter
scale, HOG detector, 128-d dlib embedding, tolerance 0.5).

  - A genuine attempt is falsely rejected if the person's name is not returned
    (including when no face is detected).
  - An impostor attempt is falsely accepted if any enrolled name is returned.
FAR = false accepts / impostor attempts; FRR = false rejects / genuine attempts,
overall and per distance.
"""

import argparse
import os

import cv2
import pandas as pd

from common import add_path, out_dir, save_json

MODULE4 = add_path('Module-4')
from simple_facerec import SimpleFacerec  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--enrolled', default=os.path.join(MODULE4, 'images'))
    args = ap.parse_args()

    sfr = SimpleFacerec()
    sfr.load_encoding_images(args.enrolled)
    enrolled = set(sfr.known_face_names)

    trials = pd.read_csv(args.manifest)
    base = os.path.dirname(os.path.abspath(args.manifest))
    rows = []
    for _, t in trials.iterrows():
        path = t['image'] if os.path.isabs(t['image']) else os.path.join(base, t['image'])
        frame = cv2.imread(path)
        if frame is None:
            print("Cannot read", path)
            continue
        _, names = sfr.detect_known_faces(frame)
        returned = [n for n in names if n != 'Unknown']
        genuine = t['identity'] != 'impostor'
        if genuine and t['identity'] not in enrolled:
            print(f"Warning: {t['identity']} is not enrolled; treating {path} as genuine anyway")
        rows.append({
            'image': t['image'], 'identity': t['identity'], 'distance_m': t.get('distance_m'),
            'genuine': genuine, 'faces_detected': len(names), 'returned': ';'.join(returned),
            'false_reject': genuine and t['identity'] not in returned,
            'false_accept': (not genuine) and len(returned) > 0,
        })

    df = pd.DataFrame(rows)
    res = out_dir('face')
    df.to_csv(os.path.join(res, 'per_attempt.csv'), index=False)

    def rates(d):
        g, i = d[d['genuine']], d[~d['genuine']]
        return {'genuine_attempts': len(g), 'impostor_attempts': len(i),
                'FRR': round(float(g['false_reject'].mean()), 4) if len(g) else None,
                'FAR': round(float(i['false_accept'].mean()), 4) if len(i) else None}

    summary = {'overall': rates(df)}
    if 'distance_m' in df and df['distance_m'].notna().any():
        summary['by_distance_m'] = {str(k): rates(g) for k, g in df.groupby('distance_m')}
    save_json(summary, os.path.join(res, 'summary.json'))
    o = summary['overall']
    print(f"FAR {o['FAR']}  FRR {o['FRR']}  ({o['impostor_attempts']} impostor, {o['genuine_attempts']} genuine attempts)")


if __name__ == "__main__":
    main()
