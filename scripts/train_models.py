"""
Reruns the whole pipeline from scratch by calling each existing script
in order, exactly the same commands you'd run by hand, just in one go
and stopping immediately if any step fails instead of plowing ahead on
bad data.

The slow synopsis fetch (over an hour against the live API) is skipped
by default since it rarely needs to change once it's been run once, use
--refetch-synopses to force it.

USAGE
    python scripts/train_models.py
    python scripts/train_models.py --refetch-synopses
    python scripts/train_models.py --skip-download
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run_step(description: str, script: str = None, module: str = None,
             extra_args: list | None = None):
    print(f"\n=== {description} ===")
    if module:
        cmd = [PYTHON, "-m", module] + (extra_args or [])
    else:
        cmd = [PYTHON, str(ROOT / "scripts" / script)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"\nStopped: {module or script} exited with code {result.returncode}.")
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                         help="skip re-downloading the raw Kaggle dataset")
    parser.add_argument("--refetch-synopses", action="store_true",
                         help="force refetching synopses from Jikan (slow, over an hour)")
    args = parser.parse_args()

    if not args.skip_download:
        run_step("Downloading raw dataset", "download_dataset.py")

    run_step("Cleaning and splitting data", "prepare_data.py")

    synopses_path = ROOT / "data" / "processed" / "anime_synopses.parquet"
    if args.refetch_synopses or not synopses_path.exists():
        run_step("Fetching synopses from Jikan (this is the slow one)", "fetch_synopses.py")
    else:
        print("\n=== Synopses already cached, skipping fetch ===")

    run_step("Training the CF model", module="src.anime_recommender.models.train_cf")
    run_step("Exporting lightweight model factors", "export_factors.py")
    run_step("Building the content model", "build_content_model.py")
    run_step("Evaluating CF and content models", "evaluate_models.py")
    run_step("Comparing hybrid weight combinations", "tune_hybrid_weights.py")
    run_step("Building the final comparison", "final_comparison.py")

    print("\nDone. Full pipeline rebuilt from scratch.")


if __name__ == "__main__":
    main()
