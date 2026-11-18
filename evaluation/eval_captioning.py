"""Caption quality on the Flickr30k test split: BLEU-1 to BLEU-4 and CIDEr (Section 4.7, Table 4).

Data: Flickr30k images and the Karpathy split file dataset_flickr30k.json
(29,000 train / 1,014 val / 1,000 test images, five reference captions each).

    python evaluation/eval_captioning.py --karpathy /path/to/dataset_flickr30k.json \
        --images /path/to/flickr30k-images

Captions are generated with the live pipeline (Module-3/Image_Captioning:
InceptionV3 encoder, LSTM decoder, beam search width 5, up to 74 tokens).
Generated captions are saved, so --resume skips images already captioned.

The metrics follow the standard COCO caption evaluation code (pycocoevalcap):
BLEU with the closest reference length for the brevity penalty, and CIDEr-D
(n = 1-4, sigma = 6, document frequencies from the test references, x10).
References are the lower-cased "tokens" in the Karpathy file.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

from common import add_path, out_dir, save_json


def ngrams(words, n):
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


# ------------------------------------------------------------------ BLEU
def corpus_bleu(hyps, refs, max_n=4):
    """hyps: list of token lists; refs: list of lists of token lists. Returns [BLEU-1..BLEU-max_n]."""
    match = [0] * max_n
    total = [0] * max_n
    hyp_len = ref_len = 0
    for h, rs in zip(hyps, refs):
        hyp_len += len(h)
        ref_len += min((abs(len(r) - len(h)), len(r)) for r in rs)[1]   # closest reference length
        for n in range(1, max_n + 1):
            hc = ngrams(h, n)
            max_ref = Counter()
            for r in rs:
                for g, c in ngrams(r, n).items():
                    max_ref[g] = max(max_ref[g], c)
            match[n - 1] += sum(min(c, max_ref[g]) for g, c in hc.items())
            total[n - 1] += max(len(h) - n + 1, 0)
    tiny = 1e-15
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, tiny))
    scores = []
    log_sum = 0.0
    for n in range(max_n):
        log_sum += math.log((match[n] + tiny) / (total[n] + tiny))
        scores.append(bp * math.exp(log_sum / (n + 1)))
    return scores


# ------------------------------------------------------------------ CIDEr-D
def cider_d(hyps, refs, n=4, sigma=6.0):
    def counts(words):
        c = Counter()
        for k in range(1, n + 1):
            c.update(ngrams(words, k))
        return c

    ref_counts = [[counts(r) for r in rs] for rs in refs]
    hyp_counts = [counts(h) for h in hyps]
    df = defaultdict(float)
    for rcs in ref_counts:
        for g in set(g for rc in rcs for g in rc):
            df[g] += 1
    ref_len = math.log(float(len(refs)))

    def vec(cnts):
        v = [defaultdict(float) for _ in range(n)]
        norm = [0.0] * n
        length = 0
        for g, tf in cnts.items():
            k = len(g) - 1
            v[k][g] = float(tf) * (ref_len - math.log(max(1.0, df[g])))
            norm[k] += v[k][g] ** 2
            if k == 1:          # as in the reference implementation
                length += tf
        return v, [math.sqrt(x) for x in norm], length

    def sim(vh, vr, nh, nr, lh, lr):
        delta = float(lh - lr)
        val = np.zeros(n)
        for k in range(n):
            for g in vh[k]:
                val[k] += min(vh[k][g], vr[k][g]) * vr[k][g]
            if nh[k] != 0 and nr[k] != 0:
                val[k] /= nh[k] * nr[k]
            val[k] *= math.e ** (-(delta ** 2) / (2 * sigma ** 2))
        return val

    scores = []
    for hc, rcs in zip(hyp_counts, ref_counts):
        vh, nh, lh = vec(hc)
        total = np.zeros(n)
        for rc in rcs:
            vr, nr, lr = vec(rc)
            total += sim(vh, vr, nh, nr, lh, lr)
        scores.append(float(np.mean(total)) / len(rcs) * 10.0)
    return float(np.mean(scores)), scores


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--karpathy', required=True, help='dataset_flickr30k.json')
    ap.add_argument('--images', required=True, help='folder with the Flickr30k images')
    ap.add_argument('--split', default='test')
    ap.add_argument('--limit', type=int, help='caption only the first N images (for a quick check)')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    data = json.load(open(args.karpathy))
    items = [im for im in data['images'] if im['split'] == args.split]
    if args.limit:
        items = items[:args.limit]
    res = out_dir('captioning')
    cap_path = os.path.join(res, f'captions_{args.split}.json')
    generated = json.load(open(cap_path)) if args.resume and os.path.exists(cap_path) else {}

    todo = [im for im in items if im['filename'] not in generated]
    if todo:
        add_path('Module-3')
        from Image_Captioning import caption_this_image  # loads the models
        for k, im in enumerate(todo, 1):
            generated[im['filename']] = caption_this_image(os.path.join(args.images, im['filename']))
            if k % 25 == 0 or k == len(todo):
                json.dump(generated, open(cap_path, 'w'), indent=1)
                print(f"{k}/{len(todo)} captioned")

    hyps = [generated[im['filename']].lower().split() for im in items]
    refs = [[[w.lower() for w in s['tokens']] for s in im['sentences']] for im in items]
    bleu = corpus_bleu(hyps, refs)
    cider, _ = cider_d(hyps, refs)
    summary = {'split': args.split, 'n_images': len(items),
               **{f'BLEU-{k + 1}': round(b, 4) for k, b in enumerate(bleu)}, 'CIDEr': round(cider, 4)}
    save_json(summary, os.path.join(res, 'summary.json'))
    print('  '.join(f"{k} {v}" for k, v in summary.items()))


if __name__ == "__main__":
    main()
