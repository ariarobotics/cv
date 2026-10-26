"""Run unlimited local checks and save overlays on the three public images."""
import argparse
import importlib.util
from pathlib import Path
import json
import cv2
import numpy as np
from hwb01_support import ContractError, check_mask, check_circle
from hwb01_support import localization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module", type=Path, default=Path(__file__).resolve().parent / 'hwb01.py')
    parser.add_argument("--output-dir", "--output", dest="output", type=Path, default=Path("results"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("student", args.module.resolve())
    student = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(student)
    refs = json.loads((root / "public_references.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    total = 0.0
    for name, ref in refs.items():
        image = cv2.imread(str(root / name))
        before = image.copy()
        try:
            mask = student.red_mask(image)
            check_mask(mask, image.shape[:2])
            if not np.array_equal(image, before):
                raise ContractError("red_mask modified its input.")
            circle = student.detect_sign(image)
            check_circle(circle, image.shape[:2])
            if not np.array_equal(image, before):
                raise ContractError("detect_sign modified its input.")
            result = localization(circle, ref)
            accuracy = (result["center_credit"] + result["radius_credit"])/2
            total += 0.3 * accuracy / 3
            print(f"{name}: center {100*result['center_credit']:.0f}%, "
                  f"radius {100*result['radius_credit']:.0f}%")
            if circle is not None:
                x, y, r = np.rint(circle).astype(int)
                cv2.circle(image, (x, y), r, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.drawMarker(image, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
            cv2.imwrite(str(args.output / f"{Path(name).stem}_overlay.jpg"), image)
            cv2.imwrite(str(args.output / f"{Path(name).stem}_mask.png"), mask)
        except Exception as error:
            print(f"{name}: {type(error).__name__}: {error}")
    print(f"Public localization score: {total:.4f} / 0.3000")
    print("Local checks do not replace the required-method checks or code review.")


if __name__ == "__main__":
    main()
