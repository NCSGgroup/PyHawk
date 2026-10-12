import unittest
import numpy as np
from src.Solver.JointChannel.physical_qc import screen_physical_oc

class PhysicalQCTests(unittest.TestCase):
    def test_onecpr_signal_preserved_and_spike_removed(self):
        t = np.arange(1000.)
        e = np.column_stack([np.ones(t.size), t/t.max(), np.sin(t/80), np.cos(t/80)])
        y = e @ np.array([.02,.01,.03,-.01]) + np.random.default_rng(7).normal(0,1e-7,t.size)
        y[400] += .001
        before = y.copy()
        keep = screen_physical_oc(t,y,e,dict(OutlierTimes=6,OutlierRMS=6e-5,OutlierObs=6))
        self.assertFalse(keep[400])
        self.assertGreater(keep.sum(),990)
        np.testing.assert_array_equal(y,before)

    def test_insufficient_support_fails(self):
        with self.assertRaises(ValueError):
            screen_physical_oc(np.arange(2.),np.ones(2),np.eye(2),{})
