"""Summarise thermal runs into Table 2 and Figure 12 (Section 6.1).

Record each run with the launcher, select mode 3 (object detection), let it
run for the session length, then quit:

    python main.py --object-n 1 --log-temperature logs/baseline_run1.csv     # no decimation (N = 1)
    python main.py              --log-temperature logs/decimated_run1.csv    # decimation (N = 3)

Each run writes <name>.csv (temperature once per second, throttling flags at
the end) and <name>_frames.csv (time of each processed frame). Then:

    python evaluation/thermal_summary.py --baseline "logs/baseline_run*.csv" \
                                          --decimated "logs/decimated_run*.csv" --ambient 27

Per run, the session starts at the first row in object-detection mode and is
cut at --duration minutes. Reported per condition (mean +/- s.d. over runs):
  - steady-state temperature: mean over the last --steady minutes
  - time to first reach --threshold (80 C, the Pi 5 soft-throttle point)
  - processed frames per second over the steady window
  - runs in which throttling or frequency capping was flagged (vcgencmd get_throttled
    bits 17, 18 or 19: frequency capped, throttled or soft temperature limit since boot)
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

from common import fmt, mean_sd, out_dir, save_json

THROTTLE_BITS = {17: 'arm frequency capped', 18: 'throttled', 19: 'soft temperature limit'}


def parse_flags(value):
    try:
        flags = int(str(value).split('=')[-1], 16)
    except ValueError:
        return None, []
    return flags, [name for bit, name in THROTTLE_BITS.items() if flags & (1 << bit)]


def load_run(path, duration_min, steady_min, threshold):
    raw = pd.read_csv(path)
    flag_rows = raw[raw['elapsed_s'].astype(str) == 'get_throttled']
    flags, events = parse_flags(flag_rows['cpu_temp_c'].iloc[0]) if len(flag_rows) else (None, [])
    df = raw[raw['elapsed_s'].astype(str) != 'get_throttled'].copy()
    for c in ('elapsed_s', 'epoch_s', 'cpu_temp_c'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[df['mode'] == 'Object detection'].dropna(subset=['cpu_temp_c'])
    if df.empty:
        raise ValueError(f"{path}: no rows in object-detection mode")
    t0 = df['elapsed_s'].iloc[0]
    df['t_min'] = (df['elapsed_s'] - t0) / 60.0
    df = df[df['t_min'] <= duration_min]

    steady = df[df['t_min'] >= df['t_min'].max() - steady_min]
    above = df[df['cpu_temp_c'] >= threshold]
    run = {
        'file': os.path.basename(path),
        'session_min': round(float(df['t_min'].max()), 2),
        'start_temp_c': float(df['cpu_temp_c'].iloc[0]),
        'steady_temp_c': float(steady['cpu_temp_c'].mean()),
        'max_temp_c': float(df['cpu_temp_c'].max()),
        'time_to_threshold_min': float(above['t_min'].iloc[0]) if len(above) else None,
        'throttle_flags': hex(flags) if flags is not None else None,
        'throttled': bool(events) if flags is not None else None,
        'throttle_events': events,
        'fps_steady': None,
    }
    frames_path = os.path.splitext(path)[0] + '_frames.csv'
    if os.path.exists(frames_path) and 'epoch_s' in df and df['epoch_s'].notna().any():
        times = pd.read_csv(frames_path, header=None)[0].to_numpy(dtype=float)
        a, b = steady['epoch_s'].min(), steady['epoch_s'].max()
        if b > a:
            run['fps_steady'] = float(((times >= a) & (times <= b)).sum() / (b - a))
    return run, df[['t_min', 'cpu_temp_c']]


def summarise(runs):
    out = {'runs': len(runs)}
    for key in ('steady_temp_c', 'time_to_threshold_min', 'fps_steady', 'start_temp_c'):
        vals = [r[key] for r in runs if r[key] is not None]
        m, s = mean_sd(vals)
        out[key] = {'mean': round(m, 2), 'sd': round(s, 2), 'n': len(vals)}
    known = [r for r in runs if r['throttled'] is not None]
    out['throttled_runs'] = f"{sum(r['throttled'] for r in known)} of {len(known)}"
    return out


def plot_figure12(curves, summary, threshold, ambient, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colours = {'baseline': '#D4700A', 'decimated': '#0F8F7A'}
    labels = {'baseline': 'Every frame processed, N = 1', 'decimated': 'Frame decimation, N = 3'}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.5, 2.52), gridspec_kw={'width_ratios': [1.75, 1]})
    for cond, runs in curves.items():
        grid = np.arange(0, min(c['t_min'].max() for c in runs) + 1e-9, 1 / 60)
        stack = np.vstack([np.interp(grid, c['t_min'], c['cpu_temp_c']) for c in runs])
        m, s = stack.mean(axis=0), stack.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros(len(grid))
        ax.fill_between(grid, m - s, m + s, color=colours[cond], alpha=0.22, lw=0)
        ax.plot(grid, m, color=colours[cond], lw=1.3, label=labels[cond])
    ax.axhline(threshold, color='#C8406E', ls='--', lw=0.9)
    if ambient is not None:
        ax.axhline(ambient, color='#888888', ls=':', lw=0.8)
    n = max(len(r) for r in curves.values())
    ax.set_xlabel('Elapsed time under continuous inference (min)', fontsize=6.4)
    ax.set_ylabel('SoC temperature (°C)', fontsize=6.4)
    ax.tick_params(labelsize=5.8)
    ax.legend(fontsize=5.2, frameon=False, title=f'mean of n = {n} runs, band ± s.d.', title_fontsize=5)
    ax.set_title('(a)  Temperature trajectory, object-detection mode', loc='left', fontsize=6.6, fontweight='bold')
    conds = [c for c in ('baseline', 'decimated') if c in summary]
    means = [summary[c]['steady_temp_c']['mean'] for c in conds]
    sds = [summary[c]['steady_temp_c']['sd'] for c in conds]
    bx.bar(range(len(conds)), means, yerr=sds, capsize=4, color=[colours[c] for c in conds], width=0.5)
    for i, (mv, sv) in enumerate(zip(means, sds)):
        bx.text(i, mv + sv + 2, f"{mv:.1f} ± {sv:.1f} °C", ha='center', fontsize=5.8)
    bx.axhline(threshold, color='#C8406E', ls='--', lw=0.9)
    bx.set_xticks(range(len(conds)))
    bx.set_xticklabels(['No\ndecimation' if c == 'baseline' else 'With\ndecimation' for c in conds], fontsize=5.8)
    bx.set_ylim(0, max(threshold, max(means) + max(sds)) + 12)
    bx.set_title('(b)  Steady state', loc='left', fontsize=6.6, fontweight='bold')
    bx.tick_params(labelsize=5.8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    print("Wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--baseline', help='glob pattern for the N = 1 run logs')
    ap.add_argument('--decimated', help='glob pattern for the N = 3 run logs')
    ap.add_argument('--duration', type=float, default=30, help='session length in minutes')
    ap.add_argument('--steady', type=float, default=10, help='steady-state window in minutes (end of session)')
    ap.add_argument('--threshold', type=float, default=80.0)
    ap.add_argument('--ambient', type=float, help='ambient temperature, measured next to the device')
    args = ap.parse_args()

    summary, curves, all_runs = {}, {}, {}
    for cond, pattern in (('baseline', args.baseline), ('decimated', args.decimated)):
        if not pattern:
            continue
        files = sorted(f for f in glob.glob(pattern) if not f.endswith('_frames.csv'))
        if not files:
            print("No files match", pattern)
            continue
        loaded = [load_run(f, args.duration, args.steady, args.threshold) for f in files]
        all_runs[cond] = [r for r, _ in loaded]
        curves[cond] = [c for _, c in loaded]
        summary[cond] = summarise(all_runs[cond])

    res = out_dir('thermal')
    save_json({'settings': vars(args), 'summary': summary, 'runs': all_runs}, os.path.join(res, 'summary.json'))
    table = []
    for cond, s in summary.items():
        ttt = s['time_to_threshold_min']
        table.append({
            'condition': cond, 'runs': s['runs'],
            'steady_state_temp_c': fmt(s['steady_temp_c']['mean'], s['steady_temp_c']['sd']),
            f"time_to_{args.threshold:g}C_min": fmt(ttt['mean'], ttt['sd']) + f" ({ttt['n']} runs reached it)"
            if ttt['n'] else 'not reached',
            'throttling_runs': s['throttled_runs'],
            'processed_fps': fmt(s['fps_steady']['mean'], s['fps_steady']['sd'], 2) if s['fps_steady']['n'] else 'n/a',
        })
    pd.DataFrame(table).to_csv(os.path.join(res, 'table2.csv'), index=False)
    print(pd.DataFrame(table).to_string(index=False))
    if curves:
        plot_figure12(curves, summary, args.threshold, args.ambient, os.path.join(res, 'figure12_thermal.png'))


if __name__ == "__main__":
    main()
