"""Resume final feedback and product publication from a completed joint solve."""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.Solver.JointChannel.resume import resume_joint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--setting-dir', default='../setting/Calibrate')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    setting_dir = Path(args.setting_dir).resolve()
    os.chdir(Path(__file__).resolve().parent)
    resume_joint(run_root, setting_dir, check_only=args.check_only)


if __name__ == '__main__':
    main()
