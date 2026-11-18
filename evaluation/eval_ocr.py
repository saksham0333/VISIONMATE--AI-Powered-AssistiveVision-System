"""OCR character and word error rates against manual transcriptions (Section 6.3, Table 4).

Each test image needs a transcription file with the same name and a .txt
extension (e.g. page01.jpg + page01.txt) holding the text exactly as printed.
An optional manifest CSV adds a type (document / signboard) and a lighting
condition per image, so results can be broken down:

    image,type,lighting
    page01.jpg,document,daylight
    sign03.jpg,signboard,indoor

    python evaluation/eval_ocr.py --images /path/to/ocr_test --manifest /path/to/ocr_test/manifest.csv

The images go through exactly the live OCR pipeline (Module-2/OCR.extract_text:
crop, 2x upscale, grayscale, bilateral filter, Otsu threshold, dilation,
Tesseract --oem 3 --psm 6, lines shorter than five characters dropped).

CER = character edit distance / reference characters; WER = word edit distance /
reference words, per image, reported as mean +/- s.d. over images. Whitespace
is normalised; use --ignore-case to ignore letter case.
"""

import argparse
import os

import cv2
import pandas as pd

from common import add_path, fmt, levenshtein, list_images, mean_sd, out_dir, save_json

add_path('Module-2')
from OCR import extract_text  # noqa: E402


def normalise(text, ignore_case):
    text = ' '.join(text.split())
    return text.lower() if ignore_case else text


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--images', required=True)
    ap.add_argument('--manifest')
    ap.add_argument('--ignore-case', action='store_true')
    args = ap.parse_args()

    meta = pd.read_csv(args.manifest).set_index('image') if args.manifest else None
    rows = []
    for path in list_images(args.images):
        txt = os.path.splitext(path)[0] + '.txt'
        if not os.path.exists(txt):
            print("No transcription for", path)
            continue
        ref = normalise(open(txt, encoding='utf-8').read(), args.ignore_case)
        hyp = normalise(' '.join(extract_text(cv2.imread(path))), args.ignore_case)
        cer = levenshtein(hyp, ref) / max(1, len(ref))
        wer = levenshtein(hyp.split(), ref.split()) / max(1, len(ref.split()))
        row = {'image': os.path.basename(path), 'cer': cer, 'wer': wer, 'reference': ref, 'recognised': hyp}
        if meta is not None and row['image'] in meta.index:
            row.update(meta.loc[row['image']].to_dict())
        rows.append(row)
        print(f"{row['image']}: CER {100 * cer:.1f}%  WER {100 * wer:.1f}%")

    df = pd.DataFrame(rows)
    res = out_dir('ocr')
    df.to_csv(os.path.join(res, 'per_image.csv'), index=False)
    summary = {'n_images': len(df)}
    for metric in ('cer', 'wer'):
        m, s = mean_sd(df[metric])
        summary[metric] = {'mean': round(m, 4), 'sd': round(s, 4)}
    for col in ('type', 'lighting'):
        if col in df:
            summary[f'by_{col}'] = {str(k): {'n': len(g), 'cer_mean': round(g['cer'].mean(), 4),
                                             'wer_mean': round(g['wer'].mean(), 4)}
                                    for k, g in df.groupby(col)}
    save_json(summary, os.path.join(res, 'summary.json'))
    print(f"\nCER {fmt(*mean_sd(df['cer']), scale=100)}%   WER {fmt(*mean_sd(df['wer']), scale=100)}%   "
          f"over {len(df)} images")


if __name__ == "__main__":
    main()
