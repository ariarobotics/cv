"""Run the public raccoon clip and save scores and reference/prediction overlays."""
import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np

from hw09_support import (run_submission, load_references, score_boxes,
                          valid_box, TIMEOUT_SECONDS, POINTS_PER_CLIP)

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module', type=Path, default=ROOT/'hw09.py')
    parser.add_argument('--output-dir', type=Path, default=Path('results'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Running the public clip (up to {TIMEOUT_SECONDS} seconds).', flush=True)
    result = run_submission(args.module.resolve(), ROOT/'frames')
    if not result['ok']:
        summary = {'points': 0.0, 'max_points': POINTS_PER_CLIP, 'reason': result['reason']}
        print(f"Public score: 0 / {POINTS_PER_CLIP}. {result['reason']}")
    else:
        refs = load_references(ROOT/'public_references.csv')
        boxes = result['boxes']
        summary = score_boxes(boxes, refs, len(boxes))
        summary.update(method_calls=result['method_calls'], seconds=round(result['seconds'], 3))
        print(f"Public score: {summary['points']:.4f} / {POINTS_PER_CLIP}")
        print(f"Average frame credit: {100*summary['mean_credit']:.2f}% (90% earns full clip credit).")
        print('Observed flow calls:', result['method_calls'])
        np.save(args.output_dir/'predictions.npy', boxes)
        with (args.output_dir/'frame_scores.csv').open('w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(summary['frames'][0]))
            writer.writeheader()
            writer.writerows(summary['frames'])
        figures = []
        for ref, scored in zip(refs, summary['frames']):
            i = ref['index']
            image = cv2.imread(str(ROOT/'frames'/f'{i:06d}.jpg'))
            for b, color in [(ref['box'], (255, 255, 0)), (boxes[i], (0, 170, 255))]:
                if valid_box(b, image.shape[:2]):
                    x, y, w, h = np.rint(b).astype(int)
                    cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            name = f'frame_{i:06d}.jpg'
            cv2.imwrite(str(args.output_dir/name), image, [cv2.IMWRITE_JPEG_QUALITY, 88])
            caption = f"{ref['time']:.1f} s | IoU {scored['iou']:.3f} | frame credit {100*scored['credit']:.1f}%"
            figures.append(f'<figure><figcaption>{html.escape(caption)}</figcaption><img src="{name}" loading="lazy"></figure>')
        page = ('<!doctype html><meta charset="utf-8"><title>HW09 public results</title>'
                '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:0 15px}'
                'figure{margin:24px 0}img{width:100%}</style><h1>HW09 public results</h1>'
                f"<p>Score: {summary['points']:.4f} / 2.5. Cyan: reference. Orange: prediction.</p>"
                '<p>IoU 0.20 earns full frame credit; 90% average frame credit earns full clip credit.</p>'
                + ''.join(figures))
        (args.output_dir/'review.html').write_text(page, encoding='utf-8')
        print('Overlays:', args.output_dir/'review.html')
    (args.output_dir/'public_results.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
