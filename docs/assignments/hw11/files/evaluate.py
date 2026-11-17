"""PROVIDED local checks for HW11. Leave this file unchanged.

    python evaluate.py                    # public tests on the courtyard pair
    python evaluate.py --photos photos    # run your code on your own photos

Each run saves reconstruction.jpg, sparse.ply, and dense.ply in results/ (or the
--output-dir folder). They are for checking your work; do not submit them.
"""
import argparse
from pathlib import Path

from hw11_support import (FUNCTIONS, Observer, load_module, load_pair, load_test_inputs, run_pipeline,
                          run_tests, save_results, unpack)

HOME = Path(__file__).resolve().parent


def make_executor(module, observer):
    """Run one of the student's functions on copies of the inputs and record the OpenCV calls."""
    def execute(stage, arguments):
        observer.reset()
        try:
            function = getattr(module, FUNCTIONS[stage][0])
            outputs = unpack(function(*(a.copy() for a in arguments)), stage)
            return outputs, dict(observer.events), ""
        except Exception as error:
            return None, dict(observer.events), f"{type(error).__name__}: {error}"
    return execute


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--photos", type=Path, help="folder with image1, image2, and calibration.json")
    parser.add_argument("--module", type=Path, default=HOME / "hw11.py")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    folder = args.photos or HOME / "public"
    output = args.output_dir or Path("results_own" if args.photos else "results")
    try:
        pair = load_pair(folder)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    if args.photos:
        print(f"Photos used at {pair[0].shape[1]} x {pair[0].shape[0]} pixels.")

    observer = Observer().install()          # before importing hw11.py
    try:
        module = load_module(args.module)
        execute = make_executor(module, observer)
        if not args.photos:
            tests = run_tests(execute, pair, load_test_inputs(folder))
            for test in tests:
                print(f"{'PASS' if test['status'] == 'passed' else 'FAIL'}  {test['name']}: {test['output']}")
            print(f"Public score: {sum(t['score'] for t in tests):.1f} / 2.0")
        (sparse, rectified, dense), message = run_pipeline(execute, pair)
    finally:
        observer.restore()
    print("Complete pipeline ran." if not message else f"Pipeline stopped at {message}")
    save_results(output, pair[0], sparse, rectified, dense)
    print(f"Saved reconstruction.jpg and the point clouds in {output}/.")


if __name__ == "__main__":
    main()
