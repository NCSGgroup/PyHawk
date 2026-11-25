"""KBRR epochs/arcs may return after the physical linearization changes."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import h5py
import numpy as np
from src.Preference.Pre_AdjustOrbit import AdjustOrbitConfig
from src.Solver.AdjustSST import RangeRate
from src.Solver.CalibrateOrbit import CalibrateOrbit
from src.Solver.JointChannel.feedback_parallel import _feedback_arc_job


class ReselectionTests(unittest.TestCase):
    def config(self):
        c = AdjustOrbitConfig().RangeRate()
        c.ParameterFitting = 'biasC0C1'
        c.Iterations = 3
        c.OutlierTimes = 8
        c.OutlierRMS = 1e-4
        c.OutlierObs = None
        c.EpochOutlierEnabled = False
        c.OneCPROutlierOnly = False
        return c

    def write(self, path, oc, config):
        t = np.arange(len(oc), dtype=float) * 5 + .25
        r = np.tile([7e6, 0, 0, 7e6, 1e5, 0], (len(t), 1))
        v = np.zeros_like(r)
        raw = np.column_stack((t, np.zeros(len(t)), oc))
        with patch.object(RangeRate, 'mean_motion_of_the_mid_point', return_value=np.ones(len(t))*.001):
            RangeRate.from_integrated_orbit(0, ['2010-01-01','2010-01-31'], config,
                                            raw, t, r, v, path)
        return t

    def test_epoch_returns_from_original_observation_after_new_orbit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            c = self.config()
            t = np.arange(600.) * 5 + .25
            raw_y = np.random.default_rng(1).normal(0, 1e-7, len(t))
            r = np.tile([7e6, 0, 0, 7e6, 1e5, 0], (len(t),1))
            old_v = np.zeros_like(r)
            old_v[250,1] = .001  # old computed range-rate is wrong
            raw = np.column_stack((t, np.zeros(len(t)), raw_y))
            with patch.object(RangeRate, 'mean_motion_of_the_mid_point', return_value=np.ones(len(t))*.001):
                for name,v in [('first',old_v),('second',np.zeros_like(r))]:
                    RangeRate.from_integrated_orbit(0, ['2010-01-01','2010-01-31'],c,
                                                    raw,t,r,v,root/(name+'.h5'))
            with h5py.File(root/'first.h5','r') as h:
                self.assertNotIn(t[250],h['post_t'][()])
                self.assertEqual(len(h['pre_t']),len(t))
            with h5py.File(root/'second.h5','r') as h:
                self.assertIn(t[250],h['post_t'][()])
                np.testing.assert_allclose(h['pre_residual'][()],raw_y)
                self.assertEqual(len(h['post_t']),len(t))

    def test_arc_returns_when_all_candidates_are_screened_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = AdjustOrbitConfig()
            c = self.config()
            c.Iterations = 0  # isolate whole-arc selection
            cfg.RangeRateConfig.update(c.__dict__)
            detector = CalibrateOrbit()
            detector._CalibrateOrbit__adjust_config = cfg
            detector._CalibrateOrbit__date_span = ['2010-01-01','2010-01-31']
            candidates = list(range(20))
            for stage in ('first','second'):
                folder = root/stage
                for arc in candidates:
                    y = np.random.default_rng(arc).normal(0,1e-7,600)
                    if stage == 'first' and arc == 9:
                        y *= 1000
                    self.write(folder/('RangeRate_%d.hdf5'%arc),y,c)
                accepted,rejected = detector._CalibrateOrbit__screenKBRRArcs(candidates,res_dir=folder)
                if stage == 'first':
                    self.assertIn(9,rejected)
                else:
                    self.assertIn(9,accepted)
            with h5py.File(root/'first/RangeRate_9.hdf5','r') as h:
                self.assertFalse(h.attrs['arc_qc_accepted'])

    def test_parallel_refresh_uses_full_raw_epochs_and_new_orbit(self):
        from src.Solver.JointChannel.channel import Settings
        from src.Solver.JointChannel.refresh_qc import refresh_kbrr_qc
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            repo=Path(__file__).resolve().parents[1]
            settings=Settings(repo/'setting/Calibrate',repo/'demo')
            settings.raw['AdjustOrbitConfig']['RangeRateConfig'].update(self.config().__dict__)
            settings.raw['InterfaceConfig']['PathOfFilesConfig']['temp_raw_data']=str(root/'Raw')
            settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['RangeRateTemp']=str(root/'old')
            (root/'Raw').mkdir(); dm=root/'DM'/settings.tag;dm.mkdir(parents=True)
            t=np.arange(600.)*5+400000000.25
            r=np.tile([7e6,0,0,7e6,1e5,0],(len(t),1))
            v=np.tile([0.,7500.,0.,0.,7500.,0.],(len(t),1))
            with h5py.File(root/'Raw'/(settings.tag+'.hdf5'),'w') as raw:
                for arc in (0,1):
                    y=np.random.default_rng(arc+11).normal(0,1e-7,len(t));y[250]+=.001
                    raw['KBR_A_arc%d'%arc]=np.column_stack((t,np.zeros(len(t)),y))
                    with h5py.File(dm/('Orbit_%d_AB.hdf5'%arc),'w') as h:
                        h['t']=t;h['r']=r;h['v']=v
            first=refresh_kbrr_qc(settings,[0,1],[0],root/'DM',root/'qc0',2)
            self.assertEqual(first,[0,1])
            with h5py.File(root/'qc0'/settings.tag/'RangeRate_0.hdf5','r') as h:
                self.assertNotIn(t[250],h['post_t'][()])
            for arc in (0,1):
                with h5py.File(dm/('Orbit_%d_AB.hdf5'%arc),'r+') as h:
                    updated=v.copy();updated[250,1]-=.001;h['v'][...]=updated
            refresh_kbrr_qc(settings,[0,1],first,root/'DM',root/'qc1',2)
            with h5py.File(root/'qc1'/settings.tag/'RangeRate_0.hdf5','r') as h:
                self.assertEqual(len(h['pre_t']),len(t))
                self.assertIn(t[250],h['post_t'][()])
                self.assertFalse(h.attrs['previous_selection_applied'])
            with h5py.File(root/'qc0'/settings.tag/'RangeRate_0.hdf5','r') as h:
                self.assertNotIn(t[250],h['post_t'][()])
            self.assertTrue(settings.raw['SolverConfig']['KBRRQCRebuilt'])

    def test_rejected_arc_integrates_shared_update_without_invented_local_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'solve';source.mkdir()
            base=root/'base.h5'
            acc=np.r_[np.ones(9),.01,.02]
            initial=np.arange(12.).reshape(2,6)
            with h5py.File(base,'w') as h:
                h['gravity_delta']=np.array([.1,.2])
                h['monthly_scale']=np.ones((2,9))
                h['arc_9/initial']=initial
                h['arc_9/acc_A']=acc;h['arc_9/acc_B']=acc
            delta=np.arange(20.)*.001
            with h5py.File(source/'global.h5','w') as h:h['delta']=delta
            (source/'manifest.json').write_text(json.dumps(dict(arcs=[0],base_state=str(base))))
            settings=SimpleNamespace(raw={'GravityDesignParameterConfig':{'AccelerometerConfig':{'isScale':True}}},obj=lambda name:None)
            with patch('src.Solver.JointChannel.feedback_parallel.Settings',return_value=settings), \
                 patch('src.Interface.ArcOperation.ArcData'), \
                 patch('src.Solver.JointChannel.feedback_parallel.integrate_state',return_value=([],[],[])) as integrate:
                arc,data,_,diagnostic=_feedback_arc_job(9,source,root/'out',.5)
            np.testing.assert_array_equal(data['initial'],initial)
            np.testing.assert_allclose(data['acc_A'][:9],acc[:9]+.5*delta[:9])
            np.testing.assert_array_equal(data['acc_A'][9:],acc[9:])
            np.testing.assert_allclose(integrate.call_args[0][4],[.109,.2095])
            self.assertEqual(diagnostic['local_update'],'zero')


if __name__ == '__main__': unittest.main()

