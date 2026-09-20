"""
Single entry point that runs the whole pipeline end-to-end, in order:

  1. unfolded_optics_deblur.py  - architecture shape/gradient-flow self-test
  2. demo_real_image.py         - single real-photo blur/fit sanity demo
  3. train_on_dataset.py        - multi-image training with held-out validation

Run everything (from PyCharm: right-click this file -> Run, or from a terminal):

    python run_all.py

Run a subset with --only (comma-separated: selftest, demo, train), e.g. to skip the
~4-5 minute training stage while iterating on the earlier stages:

    python run_all.py --only selftest,demo

All artifacts (checkpoint + figures) land in outputs/, same as running each script
individually.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

STAGES = {
    "selftest": "unfolded_optics_deblur.py",
    "demo": "demo_real_image.py",
    "train": "train_on_dataset.py",
}


def run_stage(root: Path, name: str, script: str) -> None:
    print("\n" + "=" * 88)
    print(f"[{name}] python {script}")
    print("=" * 88, flush=True)
    t0 = time.time()
    result = subprocess.run([sys.executable, str(root / script)], cwd=root)
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[{name}] FAILED (exit code {result.returncode}) after {elapsed:.1f}s")
        sys.exit(result.returncode)
    print(f"\n[{name}] done in {elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the deblurring pipeline stages in order.")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help=f"Comma-separated subset of stages to run, from {list(STAGES)}. Default: all, in order.",
    )
    args = parser.parse_args()

    selected = list(STAGES) if args.only is None else [s.strip() for s in args.only.split(",")]
    unknown = [s for s in selected if s not in STAGES]
    if unknown:
        parser.error(f"Unknown stage(s): {unknown}. Valid stages: {list(STAGES)}")

    root = Path(__file__).resolve().parent
    (root / "outputs").mkdir(exist_ok=True)

    t_start = time.time()
    for name in selected:
        run_stage(root, name, STAGES[name])

    print("\n" + "=" * 88)
    print(f"All {len(selected)} stage(s) completed successfully in {time.time()-t_start:.1f}s.")
    print(f"Outputs written to: {root / 'outputs'}")
    print("=" * 88)


if __name__ == "__main__":
    main()
