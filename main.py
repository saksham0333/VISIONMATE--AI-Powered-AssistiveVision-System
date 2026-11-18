"""VisionMate mode launcher.

Runs exactly one recognition module at a time (single-model residency).
Each mode is started as its own process, and the module that is running is
stopped first, releasing its model and the camera, so only one model is in
memory at any time.

Run from the repository root:

    python main.py

Then type a mode number and press Enter:

    0  Idle (no module running, no model in memory)
    1  Scene description      Module-3/Image_Captioning.py
    2  Text reading (OCR)     Module-2/OCR.py
    3  Object detection       detection.py
    4  Face recognition       Module-4/reco.py
    5  Traffic sign           traffic_sign.py
    6  Traffic light          Module-6/traffic_light.py
    7  Sign language          Module-7/sign.py
    q  Quit

Options for the thermal test (Section 6.1 of the paper):

    python main.py --log-temperature thermal_log.csv
        logs the CPU temperature once per second (vcgencmd measure_temp) and
        reads the throttling flags (vcgencmd get_throttled) when you quit;
        object detection also writes the time of each processed frame to
        thermal_log_frames.csv. Summarise the runs with
        evaluation/thermal_summary.py.

    python main.py --object-n 1
        runs object detection on every frame (baseline, N = 1) instead of
        every third frame (N = 3).
"""

import argparse
import csv
import os
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Mode numbers follow Figure 5 of the paper.
MODES = {
    '1': ('Scene description', os.path.join('Module-3', 'Image_Captioning.py')),
    '2': ('Text reading', os.path.join('Module-2', 'OCR.py')),
    '3': ('Object detection', 'detection.py'),
    '4': ('Face recognition', os.path.join('Module-4', 'reco.py')),
    '5': ('Traffic sign', 'traffic_sign.py'),
    '6': ('Traffic light', os.path.join('Module-6', 'traffic_light.py')),
    '7': ('Sign language', os.path.join('Module-7', 'sign.py')),
}
IDLE = '0'

# Mode changes requested within this many seconds of the last change are
# ignored, so repeated key presses do not restart a module or repeat the
# announcement while the user settles on a mode.
MODE_CHANGE_COOLDOWN = 3

sys.path.append(os.path.join(BASE_DIR, 'Module-1'))
try:
    from voice import voice
except Exception as e:  # the launcher still works without speech
    print("Voice helper not available:", e)
    voice = None


def announce(text):
    print(text)
    if voice is not None:
        try:
            voice(text)
        except Exception as e:
            print("Speech error:", e)


def stop(process):
    """Stop the running module and wait until it has exited."""
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def start(key, env):
    name, script = MODES[key]
    process = subprocess.Popen([sys.executable, os.path.join(BASE_DIR, script)], cwd=BASE_DIR, env=env)
    announce(f"{name} mode activated")
    return process


# ----------------------------------------------------------------------
# Thermal logging (Raspberry Pi only)
# ----------------------------------------------------------------------

def vcgencmd(*args):
    try:
        return subprocess.run(['vcgencmd', *args], capture_output=True, text=True, timeout=2).stdout.strip()
    except Exception:
        return ''


def read_temperature():
    out = vcgencmd('measure_temp')  # e.g. temp=56.2'C
    try:
        return float(out.split('=')[1].split("'")[0])
    except (IndexError, ValueError):
        return None


class TemperatureLogger(threading.Thread):
    """Writes elapsed time, clock time, CPU temperature and current mode once per second."""

    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.mode = 'idle'
        self.running = True
        self.t0 = time.time()

    def run(self):
        with open(self.path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['elapsed_s', 'epoch_s', 'cpu_temp_c', 'mode'])
            while self.running:
                now = time.time()
                w.writerow([round(now - self.t0, 1), round(now, 3), read_temperature(), self.mode])
                f.flush()
                time.sleep(1)

    def finish(self):
        self.running = False
        flags = vcgencmd('get_throttled')  # e.g. throttled=0x0
        print("Throttling flags at end of run:", flags or 'not available')
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow(['get_throttled', '', flags, ''])


# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='VisionMate mode launcher')
    parser.add_argument('--log-temperature', metavar='CSV',
                        help='log CPU temperature once per second to this file')
    parser.add_argument('--object-n', type=int, default=3,
                        help='frame-decimation factor N for object detection (default 3; 1 = every frame)')
    args = parser.parse_args()

    env = dict(os.environ, VISIONMATE_DETECTION_N=str(max(1, args.object_n)))
    if args.log_temperature:
        # object detection also logs the time of every processed frame (Table 2 processing rate)
        env['VISIONMATE_FRAME_LOG'] = os.path.splitext(args.log_temperature)[0] + '_frames.csv'

    logger = None
    if args.log_temperature:
        logger = TemperatureLogger(args.log_temperature)
        logger.start()

    current = None
    last_change = 0.0

    print("VisionMate modes:")
    print(f"  {IDLE}  Idle (no module running)")
    for key, (name, _) in MODES.items():
        print(f"  {key}  {name}")
    print("  q  Quit")

    try:
        while True:
            choice = input("Mode: ").strip().lower()
            if choice == 'q':
                break
            if choice != IDLE and choice not in MODES:
                print("Unknown mode")
                continue
            if time.time() - last_change < MODE_CHANGE_COOLDOWN:
                print("Please wait a moment before changing mode again")
                continue
            stop(current)
            current = None
            if choice == IDLE:
                announce("Idle mode")
                if logger:
                    logger.mode = 'idle'
            else:
                current = start(choice, env)
                if logger:
                    logger.mode = MODES[choice][0]
            last_change = time.time()
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop(current)
        if logger:
            logger.finish()


if __name__ == "__main__":
    main()
