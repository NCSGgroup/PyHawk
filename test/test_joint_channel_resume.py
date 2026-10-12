"""Checkpoint resume must reuse solved increments and skip the old solve."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
from numpy.testing import assert_allclose

from src.Solver.JointChannel import resume


class ResumeTests(unittest.TestCase):
    def test_resume_uses_completed_second_gls_without_resolving(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'run'
            config = Path(temporary) / 'settings'
            config.mkdir()
            configuration = {
                'JointEstimationConfig': {
                    'Enabled': True, 'FeedbackEnabled': True,
                    'MaxOuterIterations': 2, 'Damping': 1.0},
                'ParallelControlConfig': {
                    'GetNEQProcess': 4, 'GetGravityDesignMatProcess': 5},
                'StepControlConfig': {'isGLS': True},
                'GLSIterationConfig': {
                    'MaxIteration': 1, 'OrbitCovarianceMode': 'EstimatedFull'}}
            (config / 'CalibrateOrbitConfig.json').write_text(json.dumps(configuration))
            first = root / 'outer_00_gls_00'
            second = root / 'outer_01_gls_00'
            state = root / 'state_01'
            for path in (root / 'outer_00_ols', first, state,
                         root / 'outer_01_ols', second):
                path.mkdir(parents=True)
                (path / 'complete.json').write_text('{}')
            first_delta = np.array([1., 2.])
            second_delta = np.array([.5, -.25])
            for directory, delta, manifest in (
                    (first, first_delta, {'arcs': [0, 1]}),
                    (second, second_delta, {
                        'arcs': [0, 1], 'base_state': str(state / 'state.h5'),
                        'dm_root': str(state / 'DesignMatrix')})):
                (directory / 'manifest.json').write_text(json.dumps(manifest))
                with h5py.File(directory / 'global.h5', 'w') as h:
                    h['delta'] = delta
            with h5py.File(state / 'state.h5', 'w') as h:
                h.attrs['source_solution'] = str(first)
                h.attrs['damping'] = 1.
                h['gravity_delta'] = first_delta
            (state / 'DesignMatrix').mkdir()
            report = Path(temporary) / 'reports'
            report.mkdir()
            for name in ('arcft', 'arc'):
                folder = report / name
                folder.mkdir()
                (folder / '2000-01-01_2000-01-31.txt').write_text('arcs')
            raw = {
                'SolverConfig': {'OrbitCovarianceMode': 'EstimatedFull'},
                'GravityDesignParameterConfig': {'AccelerometerConfig': {'isScale': False}},
                'InterfaceConfig': {'PathOfFilesConfig': {
                    'report_arcft': str(report / 'arcft'),
                    'report_arc': str(report / 'arc')}}}
            def settings(*args):
                return SimpleNamespace(raw=copy.deepcopy(raw),
                                       span=['2000-01-01', '2000-01-31'])
            def feedback(solution, output, damping, workers):
                output.mkdir()
                return output
            with patch.object(resume, 'Settings', side_effect=settings),                  patch.object(resume, 'run_feedback', side_effect=feedback) as run_feedback,                  patch.object(resume, 'publish_products', return_value={'SH': 'field.gfc'}) as publish,                  patch.object(resume.os, 'chdir'):
                resume.resume_joint(root, config)
            self.assertEqual(run_feedback.call_args.kwargs['workers'], 5)
            self.assertEqual(publish.call_args.kwargs['workers'], 4)
            assert_allclose(publish.call_args.args[2], first_delta + second_delta)
            assert_allclose(publish.call_args.args[3], first_delta)
            completed = json.loads((root / 'complete.json').read_text())
            self.assertEqual(completed['arcs'], 2)
            self.assertEqual(completed['products']['SH'], 'field.gfc')


if __name__ == '__main__':
    unittest.main()
