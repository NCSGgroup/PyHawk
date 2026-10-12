"""Separate entry point; never invokes or rewrites CalibrateOrbit.run()."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Opt-in shared-local joint estimation and physical feedback')
    sub = parser.add_subparsers(dest='command', required=True)
    solve = sub.add_parser('solve', help='jointly solve existing orbit/KBR design matrices')
    solve.add_argument('--settings', default=str(ROOT / 'setting' / 'Calibrate'))
    solve.add_argument('--path-base', default=str(ROOT / 'demo'), help='base for legacy relative JSON paths')
    solve.add_argument('--arcs', nargs='+', type=int, required=True, help='explicit accepted arc IDs')
    solve.add_argument('--output', required=True, help='new directory; existing directories are rejected')
    solve.add_argument('--no-empirical', action='store_true', help='do not estimate KBR observation nuisance')
    solve.add_argument('--covariance', action='store_true', help='use existing covariance files, not scalar weights')
    solve.add_argument('--feedback-dir', help='use a completed feedback run as the new linearization point')
    solve.add_argument('--rcond', type=float, default=1e-12)
    solve.add_argument('--allow-rank-deficient', action='store_true', help='diagnostic global pseudoinverse, not a regularized field')
    feedback = sub.add_parser('feedback', help='apply global AND local corrections and reintegrate')
    feedback.add_argument('--solution', required=True)
    feedback.add_argument('--output', required=True)
    feedback.add_argument('--damping', type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.command == 'solve':
        from src.Solver.JointChannel.channel import Settings, run_joint
        dm_root = base_state = None
        if args.feedback_dir:
            previous = Path(args.feedback_dir).resolve()
            if not (previous / 'complete.json').is_file():
                parser.error('feedback directory is incomplete')
            with open(previous / 'complete.json', encoding='utf-8') as f:
                info = json.load(f)
            if sorted(args.arcs) != sorted(info['arcs']):
                parser.error('feedback continuation must use the same arcs')
            args.settings = str(previous / 'settings')
            dm_root, base_state = previous / 'DesignMatrix', previous / 'state.h5'
        settings = Settings(args.settings, args.path_base)
        # Resolve the covariance mode from the same user configuration as the legacy entry.
        control = Path(args.settings) / 'CalibrateOrbitConfig.json'
        if control.exists():
            with open(control, encoding='utf-8-sig') as f:
                mode = json.load(f).get('GLSIterationConfig', {}).get('OrbitCovarianceMode')
            if mode:
                settings.raw['SolverConfig']['OrbitCovarianceMode'] = mode
        result = run_joint(settings, args.arcs, args.output, not args.no_empirical,
                           args.covariance, dm_root, base_state, args.rcond, args.allow_rank_deficient)
    else:
        from src.Solver.JointChannel.feedback import run_feedback
        result = run_feedback(args.solution, args.output, args.damping)
    print('Completed:', result)


if __name__ == '__main__':
    main()
