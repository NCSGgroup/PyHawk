"""Run: python -m unittest discover -s test -p test_joint_channel.py -v"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from numpy.testing import assert_allclose
from src.Solver.JointChannel.core import ObservationBlock, reduce_arc, solve_global, postfit
from src.Solver.JointChannel.channel import Settings, read_arc, run_joint, epoch_indices, read_qc_basis, range_rate
from src.Solver.JointChannel.feedback import OffsetGravityModel, update_physical_state, write_design_matrices

ROOT = Path(__file__).resolve().parents[1]


class JointAlgebraTests(unittest.TestCase):
    def test_shared_solution_matches_full_weighted_system(self):
        rng = np.random.default_rng(36)
        ng, nl = 4, 7
        blocks = []
        for i, rows in enumerate((26, 21, 31)):
            a, b = rng.normal(size=(rows, ng)), rng.normal(size=(rows, nl))
            # Disjoint POD support, shared A/B parameters in KBR, KBR-only nuisance.
            if i == 0:
                b[:, 3:] = 0
            elif i == 1:
                b[:, :3] = 0
                b[:, 6] = 0
            d = rng.normal(size=(rows, rows))
            c = d @ d.T + np.eye(rows)
            blocks.append(ObservationBlock(str(i), a, b, rng.normal(size=rows), c, i+1))
        white = [b.whiten() for b in blocks]
        design = np.vstack([np.column_stack((a, b)) for a, b, y in white])
        y = np.concatenate([v[2] for v in white])
        direct = np.linalg.lstsq(design, y, rcond=None)[0]
        reduced = reduce_arc(blocks)
        g, _ = solve_global(reduced.normal, reduced.rhs)
        local = reduced.recover(g)
        assert_allclose(np.r_[g, local], direct, atol=1e-12)
        assert_allclose(design.T @ (y - design @ np.r_[g, local]), 0, atol=1e-11)

    def test_independent_elimination_is_different(self):
        a = [np.array([[1.], [2.], [3.]]), np.array([[2.], [1.], [4.]])]
        b = [np.ones((3, 1)), np.array([[1.], [2.], [1.]])]
        ys = [np.array([4.1, 5, 6.2]), np.array([5., 7.1, 7.2])]
        blocks = [ObservationBlock(str(i), a[i], b[i], ys[i]) for i in range(2)]
        joint = reduce_arc(blocks)
        separate = [reduce_arc([item]) for item in blocks]
        gj = solve_global(joint.normal, joint.rhs)[0]
        gs = solve_global(sum(x.normal for x in separate), sum(x.rhs for x in separate))[0]
        self.assertGreater(abs(gj[0]-gs[0]), 1e-4)

    def test_rank_and_covariance_rejected(self):
        a = np.arange(10.)[:, None]
        with self.assertRaisesRegex(ValueError, 'rank deficient'):
            reduce_arc([ObservationBlock('bad', a, np.ones((10, 2)), a[:, 0])])
        with self.assertRaises(ValueError):
            ObservationBlock('bad', a, a, a[:, 0], -1).whiten()
        with self.assertRaisesRegex(ValueError, 'global rank deficient'):
            solve_global(np.ones((2, 2)), np.ones(2))

    def test_monthly_accumulation_matches_all_arc_unknowns(self):
        rng = np.random.default_rng(12)
        global_dm = rng.normal(size=(80, 3))
        local_dm = np.zeros((80, 4))
        local_dm[:40, :2] = rng.normal(size=(40, 2))
        local_dm[40:, 2:] = rng.normal(size=(40, 2))
        y = rng.normal(size=80)
        reduced = [reduce_arc([ObservationBlock(str(i), global_dm[40*i:40*(i+1)],
                    local_dm[40*i:40*(i+1), 2*i:2*(i+1)], y[40*i:40*(i+1)])]) for i in range(2)]
        g, _ = solve_global(sum(x.normal for x in reduced), sum(x.rhs for x in reduced))
        direct = np.linalg.lstsq(np.column_stack((global_dm, local_dm)), y, rcond=None)[0]
        assert_allclose(np.r_[g, reduced[0].recover(g), reduced[1].recover(g)], direct, atol=1e-12)

    def test_parameter_scaling(self):
        rng = np.random.default_rng(7)
        a = rng.normal(size=(40, 2)) * [1e10, 1e-3]
        b = rng.normal(size=(40, 2)) * [1e-6, 1e6]
        truth_g, truth_l = np.array([2e-10, 3e3]), np.array([5e5, 6e-6])
        reduced = reduce_arc([ObservationBlock('test', a, b, a @ truth_g+b @ truth_l)])
        g, _ = solve_global(reduced.normal, reduced.rhs)
        assert_allclose(g, truth_g, rtol=1e-10)
        assert_allclose(reduced.recover(g), truth_l, rtol=1e-10)

    def test_fractional_epochs_not_truncated(self):
        self.assertEqual(epoch_indices([10.1, 10.9], [10.9])[0], 1)
        with self.assertRaises(ValueError):
            epoch_indices([10.1], [10.9])


class StateTests(unittest.TestCase):
    def test_local_initial_acc_and_cumulative_gravity(self):
        initial = np.arange(12.).reshape(2, 6)
        acc = np.array([1., 2.])
        delta = np.arange(18.) * 0.01  # last two are observational nuisance
        result, calibrated, g = update_physical_state(initial, acc, acc, delta,
                                                      np.array([0.2, 0.3]), np.ones(2), damping=0.5)
        assert_allclose(result[0], initial[0] + delta[:6]*0.5)
        assert_allclose(result[1], initial[1] + delta[8:14]*0.5)
        assert_allclose(calibrated[1], acc + delta[14:16]*0.5)
        assert_allclose(g, [1.1, 1.15])
        assert_allclose(initial, np.arange(12.).reshape(2, 6))

    def test_monthly_scale_goes_to_correct_satellite(self):
        acc = np.r_[np.ones(9), 0., 0.]
        dg = np.arange(20.) * 0.01
        _, calibrated, g = update_physical_state(np.zeros((2, 6)), acc, acc,
                                                 np.zeros(16), dg, np.zeros(2), True)
        assert_allclose(calibrated[0][:9], 1 + dg[:9])
        assert_allclose(calibrated[1][:9], 1 + dg[9:18])
        assert_allclose(g, dg[18:])

    def test_reference_offset_does_not_mutate_or_double_add(self):
        class Reference:
            def __init__(self):
                self.c, self.s = np.arange(6.), np.zeros(6)
            def getCS(self, degree):
                return self.c, self.s
        original = Reference()
        updated = OffsetGravityModel(original, np.ones(6)*0.1, np.zeros(6))
        assert_allclose(updated.getCS(2)[0], original.c + 0.1)
        assert_allclose(updated.getCS(2)[0], original.c + 0.1)
        assert_allclose(original.c, np.arange(6.))


class LegacyAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.settings = Settings(ROOT/'setting/Calibrate', ROOT/'demo')
        s = self.settings.raw
        s['InterfaceConfig']['date_span'] = ['2000-01-01', '2000-01-01']
        s['InterfaceConfig']['PathOfFilesConfig']['temp_raw_data'] = str(self.root/'Raw')
        s['SolverConfig']['DesignMatrixTemp'] = str(self.root/'DM')
        s['SolverConfig']['OrbitKinFactor'] = 20.
        s['GravityDesignParameterConfig']['StokesCoefficientsConfig'].update(Parameter_Number=5, MinDegree=2, MaxDegree=2)
        s['GravityDesignParameterConfig']['AccelerometerConfig']['isScale'] = False
        s['AdjustOrbitConfig']['PathOfFilesConfig']['RangeRateTemp'] = str(self.root/'QC')
        # Synthetic noise tests must not inherit production outlier thresholds.
        s['AdjustOrbitConfig']['RangeRateConfig']['PhysicalOCOutlierEnabled'] = False
        for sat in ('A', 'B'):
            s['NEQConfig']['Orbit%sConfig'%sat].update(IsRequired=True, UseKinematicOrbit=True, FixKinematicOrbit=False)
        for directory in (self.root/'Raw', self.root/'DM'/self.settings.tag, self.root/'QC'/self.settings.tag):
            directory.mkdir(parents=True)
        rng = np.random.default_rng(44)
        t = np.arange(30.) * 60
        r = np.tile([7e6, 0., 0., 7e6, 2e5, 0.], (30, 1))
        v = rng.normal(size=(30, 6))
        lr, gr = rng.normal(size=(30, 6, 8)), rng.normal(size=(30, 6, 5))
        kb, ka, e = rng.normal(size=(30, 16)), rng.normal(size=(30, 5)), np.ones((30, 1))
        self.g, self.local = rng.normal(size=5)*0.001, rng.normal(size=17)*0.001
        self.t, self.r, self.v = t, r, v
        with h5py.File(self.root/'DM'/self.settings.tag/'Orbit_0_AB.hdf5', 'w') as h:
            for key, val in dict(t=t, r=r, v=v, local_r=lr, global_r=gr).items():
                h[key] = val
        with h5py.File(self.root/'DM'/self.settings.tag/'RangeRate_0.hdf5', 'w') as h:
            h['t'], h['local'], h['global'] = t, kb, ka
        with h5py.File(self.root/'QC'/self.settings.tag/'RangeRate_0.hdf5', 'w') as h:
            h['empirical_prefit_t'] = t
            h['design_matrix'] = e
            h['arc_qc_epoch_keep'] = np.ones(t.size, bool)
            h.attrs['onecpr_role'] = 'outlier_detection_only'
        with h5py.File(self.root/'Raw'/(self.settings.tag+'.hdf5'), 'w') as h:
            for i, sat in enumerate(('A', 'B')):
                y = (gr[:, i*3:i*3+3].reshape(-1, 5) @ self.g + lr[:, i*3:i*3+3].reshape(-1, 8) @ self.local[i*8:(i+1)*8]).reshape(-1, 3)
                h['KinematicOrbit_%s_arc0'%sat] = np.column_stack((t, r[:, i*3:i*3+3]+y))
            y = ka @ self.g + kb @ self.local[:16] + e @ self.local[16:]
            h['KBR_A_arc0'] = np.column_stack((t, np.zeros(t.size), range_rate(r,v)+y))

    def tearDown(self):
        self.temp.cleanup()

    def test_physical_qc_keeps_raw_oc_and_replaces_cached_mask(self):
        self.settings.raw['AdjustOrbitConfig']['RangeRateConfig'].update(
            PhysicalOCOutlierEnabled=True, OutlierRMS=1., OutlierTimes=6,
            OutlierObs=6, EpochOutlierEnabled=False, Iterations=5)
        rawfile = self.root/'Raw'/(self.settings.tag+'.hdf5')
        signal = .01 + np.random.default_rng(19).normal(0, 1e-5, self.t.size)
        signal[15] += .1
        with h5py.File(rawfile, 'a') as h:
            h['KBR_A_arc0'][:, 2] = range_rate(self.r, self.v) + signal
        with h5py.File(self.root/'QC'/self.settings.tag/'RangeRate_0.hdf5', 'a') as h:
            h['arc_qc_epoch_keep'][3] = False
        blocks, layout = read_arc(self.settings, 0)
        block = blocks[-1]
        self.assertIn(self.t[3], block.time)
        self.assertNotIn(self.t[15], block.time)
        idx = epoch_indices(self.t, block.time)
        assert_allclose(block.residual, signal[idx], atol=1e-14)
        self.assertGreater(np.mean(block.residual), .009)
        self.assertFalse(layout['physical_oc_qc']['empirical_subtracted'])
        again, _ = read_arc(self.settings, 0)
        assert_allclose(again[-1].time, block.time)

    def test_parallel_gls_vce_statistics_and_covariance_snapshot(self):
        import shutil
        from types import SimpleNamespace
        from src.Solver.JointChannel.arc_weights import estimate_monthly_noise,read_statistics,update_variances
        from src.Solver.CovarianceModel import save_stationary_covariance,range_rate_covariance_filename
        solver=self.settings.raw['SolverConfig']
        solver.update(PostOrbit=str(self.root/'PostOrbit'),PostKBRR=str(self.root/'PostKBRR'),
            ArcWeightingConfig={'Mode':'VCE','RedundancyProbes':128},OrbitCovarianceMode='FixedDiagonal',
            CovarianceEpochMapping='PhysicalTime')
        for prefix,suffix in (('Orbit_','_AB.hdf5'),('RangeRate_','.hdf5')):
            shutil.copyfile(self.root/'DM'/self.settings.tag/(prefix+'0'+suffix),
                            self.root/'DM'/self.settings.tag/(prefix+'1'+suffix))
        shutil.copyfile(self.root/'QC'/self.settings.tag/'RangeRate_0.hdf5',self.root/'QC'/self.settings.tag/'RangeRate_1.hdf5')
        rng=np.random.default_rng(21)
        with h5py.File(self.root/'Raw'/(self.settings.tag+'.hdf5'),'a') as h:
            for name in ('KinematicOrbit_A','KinematicOrbit_B','KBR_A'):
                h.copy(name+'_arc0',name+'_arc1')
                for arc in (0,1):
                    key=name+'_arc%d'%arc
                    data=h[key][()]
                    if name=='KBR_A':data[:,2]+=rng.normal(0,.001*(arc+1),len(data))
                    else:data[:,1:4]+=rng.normal(0,.005*(arc+1),(len(data),3))
                    h[key][...]=data
        ols=run_joint(self.settings,[0,1],self.root/'ols_vce',workers=2)
        gls=SimpleNamespace(OrbitCovarianceMode='FixedDiagonal',OrbitFixedDiagonal=.004)
        estimate_monthly_noise(self.settings,ols,gls,workers=2)
        solved=run_joint(self.settings,[0,1],self.root/'gls_vce',covariance=True,workers=2)
        stats=read_statistics(solved)
        config,diagnostics=update_variances(solver,stats)
        self.assertTrue(all(value['redundancy']>0 for group in stats.values() for value in group.values()))
        json.dumps(diagnostics)
        frozen=Settings(solved/'settings',solved)
        before,_=read_arc(frozen,0,covariance=True)
        save_stationary_covariance(range_rate_covariance_filename(solver['PostKBRR'],self.settings.span),diagonal_value=999.)
        after,_=read_arc(frozen,0,covariance=True)
        assert_allclose(before[-1].covariance,after[-1].covariance,atol=0,rtol=0)
        with h5py.File(solved/'global.h5') as h:self.assertEqual(h['redundancy_probes'].shape,(5,128))
        from src.Solver.JointChannel.workflow import run_calibrate_joint
        from unittest.mock import patch,Mock
        solver.update(PostParameter=str(self.root/'PostParameter'))
        self.settings.raw['NEQConfig']['PathOfFilesConfig']['NormalEqTemp']=str(self.root/'NEQ')
        config_dir=self.root/'workflow_settings';self.settings.save(config_dir)
        gls=SimpleNamespace(OrbitCovarianceMode='FixedDiagonal',OrbitFixedDiagonal=.004,
            MaxIteration=2,MinIteration=2,RandomSeed=1,AbsoluteTolerance=0.,RelativeTolerance=0.)
        with patch('src.Solver.JointChannel.workflow.publish_products',return_value={}) as publish:
            run_calibrate_joint(Mock(),config_dir,[0,1],dict(Enabled=True,ArcProcesses=2,FeedbackProcesses=2),True,gls)
        runs=list((self.root/'NEQ'/'JointChannel'/self.settings.tag).iterdir())
        record=json.loads((runs[0]/'complete.json').read_text())
        self.assertEqual(record['iterations'][0]['gls_iterations'],2)
        weights=json.loads((runs[0]/'arc_weights_00_01.json').read_text())
        self.assertTrue(weights['records'])
        self.assertTrue((runs[0]/'outer_00_gls_01'/'covariance'/'RangeRate.h5').is_file())


    def test_itsg_epoch_arc_vce_uses_standardized_temporal_model(self):
        from types import SimpleNamespace
        from src.Solver.JointChannel.arc_weights import estimate_monthly_noise,read_statistics,_noise_arc_job,precision_factor
        from src.Solver.OrbitObservations import epoch_covariance
        solver=self.settings.raw['SolverConfig']
        solver.update(PostOrbit=str(self.root/'PostOrbit'),PostKBRR=str(self.root/'PostKBRR'),
            ArcWeightingConfig={'Mode':'VCE','RedundancyProbes':128,'PriorPrecision':{'Orbit_A':.5}},
            OrbitCovarianceMode='ITSGEpoch',CovarianceEpochMapping='PhysicalTime')
        rng=np.random.default_rng(111)
        components=np.tile([1e-4,2e-4,3e-4,1e-5,2e-5,1e-5],(len(self.t),1))
        with h5py.File(self.root/'Raw'/(self.settings.tag+'.hdf5'),'a') as h:
            for sat in ('A','B'):
                key='KinematicOrbit_%s_arc0'%sat;obs=h[key][()];del h[key]
                obs[:,1:4]+=rng.normal(0,.01,obs[:,1:4].shape)
                h[key]=np.column_stack((obs,components))
            data=h['KBR_A_arc0'][()];data[:,2]+=rng.normal(0,.001,len(data));h['KBR_A_arc0'][...]=data
        ols=run_joint(self.settings,[0],self.root/'itsg_ols',workers=1)
        data=_noise_arc_job(0,ols,solver)
        with h5py.File(ols/'arc_0.h5') as h:
            e=h['Orbit_A/linear_postfit'][()].reshape(-1,3)
        expected=np.linalg.solve(np.linalg.cholesky(epoch_covariance(components)),e[:,:,None])[:,:,0]*np.sqrt(precision_factor(solver,0,'Orbit_A'))
        assert_allclose(data['Orbit_A'][1],expected)
        estimate_monthly_noise(self.settings,ols,SimpleNamespace(OrbitCovarianceMode='ITSGEpoch'),workers=1)
        result=run_joint(self.settings,[0],self.root/'itsg_gls',covariance=True,workers=1)
        stats=read_statistics(result)['0']
        self.assertTrue(all(st['redundancy']>0 for st in stats.values()))
        self.assertEqual(stats['Orbit_A']['precision_weight'],.5)

    def test_kin_relative_initial_weights_and_residual_only_gls(self):
        from src.Solver.JointChannel.initial_weights import prepare_initial_weights
        from src.Solver.CovarianceModel import save_stationary_covariance,orbit_covariance_filename,range_rate_covariance_filename
        solver=self.settings.raw['SolverConfig']
        solver.update(InitialOrbitWeighting='KINRelative',OrbitCovarianceMode='EstimatedFull',
            PostOrbit=str(self.root/'PostOrbit'),PostKBRR=str(self.root/'PostKBRR'),
            CovarianceEpochMapping='PhysicalTime',ArcWeightingConfig={'Mode':'Off'})
        path=self.root/'Raw'/(self.settings.tag+'.hdf5')
        with h5py.File(path,'a') as h:
            for i,sat in enumerate(('A','B')):
                key='KinematicOrbit_%s_arc0'%sat;obs=h[key][()];del h[key]
                components=np.tile(np.array([1.,2.,3.,.1,.2,.3])*1e-4*(i+1),(len(obs),1))
                components*=np.linspace(1.,4.,len(obs))[:,None]
                obs[:,1:4]+=np.random.default_rng(i+123).normal(0,.01,obs[:,1:4].shape)
                h[key]=np.column_stack((obs,components))
            data=h['KBR_A_arc0'][()];data[:,2]+=np.random.default_rng(12).normal(0,.001,len(data));h['KBR_A_arc0'][...]=data
        result=run_joint(self.settings,[0],self.root/'initial_wls',workers=2)
        frozen=Settings(result/'settings',result)
        blocks,_=read_arc(frozen,0)
        inverses=[np.linalg.inv(b.covariance) for b in blocks[:2]]
        average=sum(np.trace(v,axis1=1,axis2=2).sum() for v in inverses)/(2*len(self.t)*3)
        assert_allclose(average,1.,rtol=1e-12)
        self.assertEqual(blocks[-1].weight,solver['OrbitKinFactor'])
        white=[b.whiten() for b in blocks]
        matrix=np.column_stack((np.concatenate([w[0] for w in white]),np.concatenate([w[1] for w in white])))
        direct=np.linalg.lstsq(matrix,np.concatenate([w[2] for w in white]),rcond=None)[0]
        with h5py.File(result/'global.h5') as h:assert_allclose(h['delta'][()],direct[:5],atol=2e-10)
        # A common rescaling of all formal KIN variances cannot change relative F.
        with h5py.File(path,'a') as h:
            for sat in ('A','B'):
                key='KinematicOrbit_%s_arc0'%sat;data=h[key][()];data[:,4:10]*=49.;h[key][...]=data
        prepare_initial_weights(self.settings,[0],None,2)
        changed,_=read_arc(self.settings,0)
        for before,after in zip(blocks,changed):
            for x,y in zip(before.whiten(),after.whiten()):assert_allclose(x,y,rtol=1e-12,atol=1e-12)
        # Subsequent GLS uses raw-unit residual covariance, neither epoch S nor F.
        for sat in ('A','B'):
            save_stationary_covariance(orbit_covariance_filename(solver['PostOrbit'],self.settings.span,sat),diagonal_value=.003)
        save_stationary_covariance(range_rate_covariance_filename(solver['PostKBRR'],self.settings.span),diagonal_value=.002)
        solver['OrbitKinFactor']=1e25
        with h5py.File(path,'a') as h:
            for sat in ('A','B'):
                key='KinematicOrbit_%s_arc0'%sat;data=h[key][()];data[:,4:10]=np.nan;h[key][...]=data
        gls,_=read_arc(self.settings,0,covariance=True)
        for block in gls[:2]:assert_allclose(block.covariance,.003*np.eye(len(block.residual)))
        self.assertEqual(gls[-1].weight,1.)
        assert_allclose(gls[-1].covariance,.002*np.eye(len(self.t)))
        from types import SimpleNamespace
        from src.Solver.JointChannel.arc_weights import estimate_monthly_noise,read_statistics,update_variances,_noise_arc_job
        solver['ArcWeightingConfig']={'Mode':'VCE','RedundancyProbes':128}
        raw_noise=_noise_arc_job(0,result,solver)
        with h5py.File(result/'arc_0.h5') as h:
            assert_allclose(raw_noise['Orbit_A'][1],h['Orbit_A/linear_postfit'][()].reshape(-1,3))
        estimate_monthly_noise(self.settings,result,SimpleNamespace(OrbitCovarianceMode='EstimatedFull'),workers=2)
        solved=run_joint(self.settings,[0],self.root/'residual_gls',covariance=True,workers=2)
        stats=read_statistics(solved)
        self.assertTrue(all(st['redundancy']>0 for st in stats['0'].values()))
        weights,diagnostic=update_variances(solver,stats)
        self.assertTrue(diagnostic['records'])

    def test_cached_pipeline_and_common_backsubstitution(self):
        inputs = {p: p.read_bytes() for p in self.root.rglob('*.hdf5')}
        out = run_joint(self.settings, [0], self.root/'result')
        with h5py.File(out/'global.h5', 'r') as h:
            assert_allclose(h['delta'][()], self.g, atol=2e-10)
        with h5py.File(out/'arc_0.h5', 'r') as h:
            assert_allclose(h['local_delta'][()], self.local, atol=2e-10)
            for block in ('Orbit_A', 'Orbit_B', 'RangeRate'):
                assert_allclose(h[block+'/linear_postfit'][()], 0, atol=1e-9)
        for p, data in inputs.items():
            self.assertEqual(p.read_bytes(), data)
        with self.assertRaises(FileExistsError):
            run_joint(self.settings, [0], out)

    def test_two_arc_parallel_matches_serial(self):
        import shutil
        tag = self.settings.tag
        for folder, source, target in (
                (self.root/'DM'/tag, 'Orbit_0_AB.hdf5', 'Orbit_1_AB.hdf5'),
                (self.root/'DM'/tag, 'RangeRate_0.hdf5', 'RangeRate_1.hdf5'),
                (self.root/'QC'/tag, 'RangeRate_0.hdf5', 'RangeRate_1.hdf5')):
            shutil.copyfile(folder/source, folder/target)
        with h5py.File(self.root/'Raw'/(tag+'.hdf5'), 'a') as h:
            for name in ('KinematicOrbit_A', 'KinematicOrbit_B', 'KBR_A'):
                h.copy(name+'_arc0', name+'_arc1')
        serial = run_joint(self.settings, [0,1], self.root/'serial', workers=1)
        parallel = run_joint(self.settings, [0,1], self.root/'parallel', workers=2)
        with h5py.File(serial/'global.h5') as left, h5py.File(parallel/'global.h5') as right:
            for name in ('normal','rhs','delta'):
                assert_allclose(left[name][()],right[name][()],rtol=1e-12,atol=1e-12)
        for arc in (0,1):
            with h5py.File(serial/('arc_%d.h5'%arc)) as left, h5py.File(parallel/('arc_%d.h5'%arc)) as right:
                assert_allclose(left['local_delta'][()],right['local_delta'][()],atol=1e-10)
                for name in ('Orbit_A','Orbit_B','RangeRate'):
                    assert_allclose(left[name+'/linear_postfit'][()],right[name+'/linear_postfit'][()],atol=1e-10)

    def test_qc_role_does_not_disable_empirical_columns(self):
        path = self.root/'QC'/self.settings.tag/'RangeRate_0.hdf5'
        t, e, columns = read_qc_basis(path, True)
        self.assertEqual(e.shape, (30, 1))
        with h5py.File(path, 'a') as h:
            keep = h['arc_qc_epoch_keep'][()]
            keep[3] = False
            h['arc_qc_epoch_keep'][...] = keep
        blocks, layout = read_arc(self.settings, 0)
        self.assertEqual(blocks[-1].residual.size, 29)
        self.assertEqual(layout['empirical_count'], 1)

    def test_new_prefit_comes_from_updated_orbit(self):
        before, _ = read_arc(self.settings, 0)
        path = self.root/'DM'/self.settings.tag/'Orbit_0_AB.hdf5'
        with h5py.File(path, 'a') as h:
            v = h['v'][()]
            v[:, 1] += 0.001
            h['v'][...] = v
        after, _ = read_arc(self.settings, 0)
        assert_allclose(after[-1].residual-before[-1].residual, 0.001, atol=1e-14)


if __name__ == '__main__':
    unittest.main()
