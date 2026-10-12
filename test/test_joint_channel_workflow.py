"""Dispatch tests keep the expensive physical solver independently testable."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
from numpy.testing import assert_allclose
from src.Solver.JointChannel import workflow as w

class WorkflowTests(unittest.TestCase):
    def test_default_off_and_invalid_feedback(self):
        self.assertIsNone(w.run_calibrate_joint(None,None,[],{},False,None))
        with self.assertRaises(ValueError):
            w.joint_options({'MonthlyScaleEnabled':True})
        with self.assertRaises(ValueError):
            w.joint_options({'FeedbackEnabled':True})
        with self.assertRaises(ValueError):
            w.joint_options({'Enabled':True,'MaxOuterIterations':0})

    def test_joint_workers_inherit_parallel_control(self):
        parallel = {'GetNEQProcess': 4, 'GetGravityDesignMatProcess': 5}
        options = w.joint_options({'Enabled': True}, parallel)
        self.assertEqual((options['ArcProcesses'], options['FeedbackProcesses']), (4, 5))
        override = w.joint_options({'Enabled': True, 'ArcProcesses': 2}, parallel)
        self.assertEqual((override['ArcProcesses'], override['FeedbackProcesses']), (2, 5))

    def test_itsg_mode_rejects_gnv_before_solving(self):
        settings = SimpleNamespace(raw={'SolverConfig': {}, 'NEQConfig': {
            'OrbitAConfig': {'IsRequired': True, 'UseKinematicOrbit': False},
            'OrbitBConfig': {'IsRequired': True, 'UseKinematicOrbit': True}}})
        gls = SimpleNamespace(OrbitCovarianceMode='ITSGEpoch')
        with patch.object(w, 'Settings', return_value=settings), patch.object(w, 'run_joint') as solve:
            with self.assertRaisesRegex(ValueError, 'EstimatedFull'):
                w.run_calibrate_joint(Mock(), 'settings', [0], {'Enabled': True}, True, gls)
            solve.assert_not_called()

    def test_kin_epoch_seed_only_first_outer_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=SimpleNamespace(raw={'SolverConfig':{'InitialOrbitWeighting':'KINRelative','CovariancePoolingMode':'WithinArc'},
                'NEQConfig':{'PathOfFilesConfig':{'NormalEqTemp':tmp}},
                'GravityDesignParameterConfig':{'AccelerometerConfig':{'isScale':False}}},tag='month')
            gls=SimpleNamespace(OrbitCovarianceMode='EstimatedFull',MaxIteration=1,MinIteration=1,
                RandomSeed=1,AbsoluteTolerance=0.,RelativeTolerance=0.)
            with patch.object(w,'Settings',return_value=settings), \
                 patch.object(w,'run_joint',side_effect=lambda *a,**k:a[2]) as joint, \
                 patch.object(w,'load_result',return_value=({},np.array([1.,2.]))), \
                 patch.object(w,'run_feedback',side_effect=lambda *a,**k:a[1]), \
                 patch.object(w,'export_residuals'),patch.object(w,'publish_products',return_value={}), \
                 patch('src.Solver.JointChannel.arc_weights.estimate_monthly_noise',return_value={'converged':False}):
                w.run_calibrate_joint(Mock(),tmp,[0],dict(Enabled=True,FeedbackEnabled=True,MaxOuterIterations=2),True,gls)
            self.assertEqual([call.args[4] for call in joint.call_args_list],[False,True,True,True])
            self.assertTrue(str(joint.call_args_list[2].args[2]).endswith('outer_01_gls_seed'))

    def test_initial_wls_basis_survives_outer_iterations(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=SimpleNamespace(raw={'SolverConfig':{'InitialOrbitWeighting':'KINRelative','CovariancePoolingMode':'WithinArc','GLSResidualSpace':'InitialWLS','OrbitKinFactor':100.,'InitialOrbitMeanPrecision':2.,'InitialWeightingAudit':{}},
                'NEQConfig':{'PathOfFilesConfig':{'NormalEqTemp':tmp}},
                'GravityDesignParameterConfig':{'AccelerometerConfig':{'isScale':False}}},tag='month')
            gls=SimpleNamespace(OrbitCovarianceMode='EstimatedFull',MaxIteration=1,MinIteration=1,
                RandomSeed=1,AbsoluteTolerance=0.,RelativeTolerance=0.)
            with patch.object(w,'Settings',return_value=settings), \
                 patch.object(w,'run_joint',side_effect=lambda *a,**k:a[2]) as joint, \
                 patch.object(w,'load_result',return_value=({},np.array([1.,2.]))), \
                 patch.object(w,'run_feedback',side_effect=lambda *a,**k:a[1]), \
                 patch.object(w,'export_residuals'),patch.object(w,'publish_products',return_value={}), \
                 patch('src.Solver.JointChannel.arc_weights.estimate_monthly_noise',return_value={'converged':False}):
                w.run_calibrate_joint(Mock(),tmp,[0],dict(Enabled=True,FeedbackEnabled=True,MaxOuterIterations=2),True,gls)
            self.assertEqual([call.args[4] for call in joint.call_args_list],[False,True,True,True])
            self.assertTrue(str(joint.call_args_list[2].args[2]).endswith('outer_01_gls_seed'))

    def test_gls_and_cumulative_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = SimpleNamespace(raw={'SolverConfig':{},'NEQConfig':{'PathOfFilesConfig':{'NormalEqTemp':tmp}},
                'GravityDesignParameterConfig':{'AccelerometerConfig':{'isScale':False}}},tag='month')
            owner = Mock()
            gls = SimpleNamespace(OrbitCovarianceMode='FixedDiagonal',MaxIteration=1,MinIteration=1,
                RandomSeed=1,OrbitFixedDiagonal=.004,AbsoluteTolerance=0.,RelativeTolerance=0.)
            def solve(*args,**kwargs):
                return args[2]
            def feedback(*args,**kwargs):
                return args[1]
            with patch.object(w,'Settings',return_value=settings), patch.object(w,'run_joint',side_effect=solve) as joint, \
                 patch.object(w,'load_result',return_value=({},np.array([2.,4.]))), \
                 patch.object(w,'run_feedback',side_effect=feedback) as feed, \
                 patch.object(w,'export_residuals'), patch.object(w,'publish_products',return_value={}) as publish:
                w.run_calibrate_joint(owner,tmp,[0],dict(Enabled=True,FeedbackEnabled=True,MaxOuterIterations=2,Damping=.5,
                    ArcProcesses=4,FeedbackProcesses=5),True,gls)
                self.assertEqual(joint.call_count,4)
                self.assertEqual(owner.GLS.call_count,2)
                self.assertEqual(feed.call_count,2)
                self.assertTrue(all(call.kwargs['workers']==4 for call in joint.call_args_list))
                self.assertTrue(all(call.kwargs['workers']==5 for call in feed.call_args_list))
                assert_allclose(owner.cs,[2.,4.])
                assert_allclose(publish.call_args[0][3],[1.,2.])
                self.assertEqual(joint.call_args_list[2][0][6],feed.call_args_list[0][0][1]/'state.h5')

    def test_gls_only_exports_each_solution_without_feedback(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings=SimpleNamespace(raw={'SolverConfig':{'CovariancePoolingMode':'WithinArc'},
                'NEQConfig':{'PathOfFilesConfig':{'NormalEqTemp':tmp}},
                'GravityDesignParameterConfig':{'AccelerometerConfig':{'isScale':False}}},tag='month')
            gls=SimpleNamespace(OrbitCovarianceMode='EstimatedFull',MaxIteration=2,MinIteration=2,
                RandomSeed=1,AbsoluteTolerance=0.,RelativeTolerance=0.)
            with patch.object(w,'Settings',return_value=settings), \
                 patch.object(w,'run_joint',side_effect=lambda *a,**k:a[2]) as joint, \
                 patch.object(w,'load_result',return_value=({},np.array([1.,2.]))), \
                 patch.object(w,'run_feedback') as feedback, \
                 patch.object(w,'export_residuals'),patch.object(w,'publish_products',return_value={}), \
                 patch('src.Solver.JointChannel.arc_weights.estimate_monthly_noise',return_value={'converged':False}), \
                 patch('src.Solver.JointChannel.arc_weights.collect_noise_groups',return_value={}), \
                 patch('src.Solver.JointChannel.iteration_products.save_input_psds') as psd, \
                 patch('src.Solver.JointChannel.iteration_products.export_iteration_gfc') as gfc:
                w.run_calibrate_joint(Mock(),tmp,[0],dict(Enabled=True,FeedbackEnabled=False,SaveIterationProducts=True),True,gls)
                self.assertEqual(joint.call_count,3)
                self.assertEqual(gfc.call_count,3)
                feedback.assert_not_called()
                self.assertEqual(psd.call_args.kwargs['purpose'],'final_postfit')

    def test_one_feedback_then_final_solve_uses_updated_state_and_products(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            settings=SimpleNamespace(raw={'SolverConfig':{'InitialOrbitWeighting':'KINRelative','CovariancePoolingMode':'WithinArc'},
                'NEQConfig':{'PathOfFilesConfig':{'NormalEqTemp':tmp}},
                'GravityDesignParameterConfig':{'AccelerometerConfig':{'isScale':False}}},tag='month')
            gls=SimpleNamespace(OrbitCovarianceMode='EstimatedFull',MaxIteration=1,MinIteration=1,
                RandomSeed=1,AbsoluteTolerance=0.,RelativeTolerance=0.)
            owner=Mock()
            deltas=[({},np.array(v)) for v in ([9.,9.],[2.,4.],[.8,1.],[.5,1.])]
            with patch.object(w,'Settings',return_value=settings), \
                 patch.object(w,'run_joint',side_effect=lambda *a,**k:a[2]) as joint, \
                 patch.object(w,'load_result',side_effect=deltas), \
                 patch.object(w,'run_feedback',side_effect=lambda *a,**k:a[1]) as feed, \
                 patch.object(w,'export_residuals'),patch.object(w,'publish_products',return_value={}) as publish, \
                 patch('src.Solver.JointChannel.arc_weights.estimate_monthly_noise',return_value={'converged':False}):
                w.run_calibrate_joint(owner,tmp,[0],dict(Enabled=True,FeedbackEnabled=True,
                    FeedbackIterations=1,MaxOuterIterations=1,Damping=.5,
                    ArcProcesses=4,FeedbackProcesses=5),True,gls)
            self.assertEqual(feed.call_count,1)
            self.assertEqual(joint.call_count,4)
            self.assertEqual([c.args[4] for c in joint.call_args_list],[False,True,True,True])
            state=feed.call_args.args[1]
            for call in joint.call_args_list[2:]:
                self.assertEqual(call.args[5],state/'DesignMatrix')
                self.assertEqual(call.args[6],state/'state.h5')
            self.assertTrue(all(c.kwargs['workers']==4 for c in joint.call_args_list))
            self.assertEqual(feed.call_args.kwargs['workers'],5)
            self.assertEqual(publish.call_args.args[1],joint.call_args_list[-1].args[2])
            self.assertIsNone(publish.call_args.args[4])
            assert_allclose(owner.cs,[1.5,3.])
            assert_allclose(publish.call_args.args[3],[1.,2.])
            complete=json.loads(next(Path(tmp).glob('JointChannel/month/*/complete.json')).read_text())
            self.assertEqual(complete['feedback_integrations'],1)
            self.assertTrue(complete['final_solve_after_feedback'])
            self.assertIsNotNone(complete['iterations'][0]['feedback'])
            self.assertIsNone(complete['iterations'][1]['feedback'])

    def test_explicit_feedback_count_validation(self):
        for value in (-1,True,1.5):
            with self.assertRaises(ValueError):
                w.joint_options(dict(Enabled=True,FeedbackEnabled=True,FeedbackIterations=value))
        with self.assertRaises(ValueError):
            w.joint_options(dict(Enabled=True,FeedbackEnabled=False,FeedbackIterations=1))

if __name__=='__main__':
    unittest.main()
