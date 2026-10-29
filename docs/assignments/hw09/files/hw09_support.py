"""Provided HW09 loading, interface, scoring and process helpers.

This module does not detect or track the animal. Do not submit or modify it.
The same file is bundled into the standalone Gradescope autograder.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

FPS = 10
TIMEOUT_SECONDS = 600
IOU_FULL_CREDIT = 0.20
CLIP_FULL_CREDIT = 0.90
POINTS_PER_CLIP = 2.5


class InterfaceError(ValueError):
    pass


def load_frames(directory):
    """Read the prepared JPEG sequence; no video decoder is required."""
    paths = sorted(Path(directory).glob('*.jpg'))
    if not paths:
        raise ValueError('No JPEG frames found.')
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.shape != (685, 1280, 3):
            raise ValueError('Missing, unreadable, or incorrectly sized prepared frame.')
        image.flags.writeable = False
        frames.append(image)
    return tuple(frames)


def load_references(path):
    with Path(path).open(newline='') as file:
        rows = list(csv.DictReader(file))
    return [{'index': int(r['input_frame_10fps']), 'time': float(r['time_s']),
             'box': np.array([float(r[k]) for k in ('x', 'y', 'width', 'height')])}
            for r in rows]


def check_output(boxes, count):
    """Whole-clip interface check, separate from per-frame localization."""
    if not isinstance(boxes, np.ndarray):
        raise InterfaceError('Return a NumPy array.')
    if boxes.shape != (count, 4):
        raise InterfaceError(f'Return shape ({count}, 4), one box per input frame.')
    if boxes.dtype.kind not in 'iuf':
        raise InterfaceError('Return a real numeric array, not bool, object, or complex.')
    return np.asarray(boxes, dtype=float)


def valid_box(box, image_shape):
    h, w = image_shape
    x, y, bw, bh = box
    return bool(np.isfinite(box).all() and bw > 0 and bh > 0
                and x >= 0 and y >= 0 and x+bw <= w+1e-6 and y+bh <= h+1e-6)


def intersection_over_union(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
    intersection = max(0.0, x2-x1)*max(0.0, y2-y1)
    union = a[2]*a[3]+b[2]*b[3]-intersection
    return float(intersection/union) if union > 0 else 0.0


def score_boxes(boxes, references, count, image_shape=(685, 1280)):
    boxes = check_output(boxes, count)
    details = []
    for ref in references:
        b = boxes[ref['index']]
        valid = valid_box(b, image_shape)
        overlap = intersection_over_union(b, ref['box']) if valid else 0.0
        credit = min(1.0, overlap/IOU_FULL_CREDIT)
        details.append({'index': ref['index'], 'time_s': ref['time'],
                        'valid': valid, 'iou': overlap, 'credit': credit})
    if not details:
        raise ValueError('Instructor reference set is empty.')
    mean = float(np.mean([r['credit'] for r in details]))
    return {'points': POINTS_PER_CLIP*min(1.0, mean/CLIP_FULL_CREDIT),
            'mean_credit': mean, 'full_credit_frames': sum(r['credit'] == 1 for r in details),
            'evaluated_frames': len(details), 'frames': details}


def _worker(job, sandbox):
    """Trusted bootstrap; the parent never imports submitted Python code."""
    if sandbox:
        if os.name != 'posix' or os.getuid() != 0:
            raise RuntimeError('Gradescope sandbox requires a root bootstrap on Linux.')
        # Private references and grader files remain owned/readable only by root.
        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)
    os.chdir(job)
    # Failures in supplied data are infrastructure errors, not student zeros.
    frames = load_frames(job/'frames')
    counts = {'lk': 0, 'farneback': 0, 'dis': 0}
    lk, farneback = cv2.calcOpticalFlowPyrLK, cv2.calcOpticalFlowFarneback
    dis_type = cv2.DISOpticalFlow

    def observe(frame, event, function):
        if event != 'c_return':
            return
        if function is lk:
            counts['lk'] += 1
        elif function is farneback:
            counts['farneback'] += 1
        elif (getattr(function, '__name__', '') == 'calc'
              and isinstance(getattr(function, '__self__', None), dis_type)):
            counts['dis'] += 1

    # Observe real successful native API calls without substituting their outputs.
    # A call proves execution, not meaningful use: source review is still needed.
    result = None
    try:
        sys.setprofile(observe)
        spec = importlib.util.spec_from_file_location('hw09_submission', job/'hw09.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        function = getattr(module, 'track_animal', None)
        if not callable(function):
            raise InterfaceError('Define track_animal(frames).')
        boxes = check_output(function(frames), len(frames))
        if not any(counts.values()):
            raise InterfaceError('No successful LK, Farneback, or DIS flow computation was observed.')
        # Invalid individual rows become missing detections; they do not erase
        # localization credit for the other frames.
        predictions = [b.tolist() if valid_box(b, frames[0].shape[:2]) else None for b in boxes]
        result = {'ok': True, 'prediction': predictions, 'method_calls': counts}
    except InterfaceError as error:
        result = {'ok': False, 'reason': str(error)}
    except BaseException as error:
        # Never return arbitrary exception text that could contain private data.
        result = {'ok': False, 'reason': 'Student code raised '+type(error).__name__+'.'}
    finally:
        sys.setprofile(None)
    (job/'output.json').write_text(json.dumps(result, allow_nan=False))


def _kill_tree(process):
    if os.name == 'posix':
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.poll() is None:
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if process.poll() is None:
            process.kill()


def run_submission(module_path, frames_dir, *, sandbox=False, timeout=TIMEOUT_SECONDS):
    """Run one clip in a new process with an elapsed-time deadline.

    The timeout override exists for instructor regression tests only. Deployed
    grading and the student evaluator use the disclosed 600-second default.
    Input copying is trusted preparation; all worker imports, input decoding,
    submitted code, and interface checks occur inside the elapsed-time budget.
    """
    module_path, frames_dir = Path(module_path), Path(frames_dir)
    if not module_path.is_file():
        return {'ok': False, 'reason': 'Submit hw09.py.'}
    paths = sorted(frames_dir.glob('*.jpg'))
    if not paths:
        raise RuntimeError('Instructor frame data is missing.')
    with tempfile.TemporaryDirectory(prefix='hw09_case_') as temp:
        job = Path(temp)
        shutil.copyfile(module_path, job/'hw09.py')
        (job/'frames').mkdir()
        for i, path in enumerate(paths):
            shutil.copyfile(path, job/'frames'/f'{i:06d}.jpg')
        if sandbox:
            os.chown(job, 65534, 65534)
            job.chmod(0o700)
            (job/'frames').chmod(0o755)
            (job/'hw09.py').chmod(0o644)
            for p in (job/'frames').iterdir():
                p.chmod(0o644)
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                   MPLCONFIGDIR=str(job/'mpl'))
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(job)]
        if sandbox:
            command.append('--sandbox')
        start = time.monotonic()
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   env=env, start_new_session=(os.name == 'posix'))
        try:
            process.wait(timeout=max(0, timeout-(time.monotonic()-start)))
        except subprocess.TimeoutExpired:
            _kill_tree(process)
            process.wait()
            return {'ok': False, 'reason': f'Time limit exceeded ({timeout:g} seconds per clip).',
                    'timeout': True, 'seconds': time.monotonic()-start}
        finally:
            _kill_tree(process)
        result_path = job/'output.json'
        if process.returncode != 0 or not result_path.is_file() or result_path.is_symlink():
            return {'ok': False, 'reason': 'Student process did not complete successfully.',
                    'seconds': time.monotonic()-start}
        try:
            result = json.loads(result_path.read_text())
            if not isinstance(result, dict) or type(result.get('ok')) is not bool:
                raise ValueError('Malformed worker output.')
            if result['ok']:
                rows = result['prediction']
                boxes = np.asarray([[np.nan]*4 if r is None else r for r in rows], dtype=float)
                check_output(boxes, len(paths))
                result['boxes'] = boxes
            result['seconds'] = time.monotonic()-start
            return result
        except Exception:
            return {'ok': False, 'reason': 'Student process returned invalid output.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--sandbox', action='store_true')
    args = parser.parse_args()
    _worker(args.worker.resolve(), args.sandbox)
