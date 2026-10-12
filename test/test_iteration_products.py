import tempfile,unittest
from pathlib import Path
import numpy as np
from src.Solver.JointChannel.iteration_products import pooled_periodogram,save_input_psds

class TestIterationPSD(unittest.TestCase):
    def test_preserves_dc_and_energy(self):
        t=np.arange(16.)*5
        f,p,m=pooled_periodogram([(t,np.full(16,3.))])
        self.assertAlmostEqual(p.sum()*(f[1]-f[0]),9.)
        self.assertAlmostEqual(p[0,0],720.)
        self.assertFalse(m['demean'])
    def test_independent_arcs_and_gaps(self):
        t=np.array([0.,5.,15.,20.]);x=np.array([2.,-1.,3.,4.])
        f,p,m=pooled_periodogram([(t,x),(t+100,-x)])
        self.assertAlmostEqual(p.sum()*(f[1]-f[0]),np.mean(x*x))
        self.assertEqual(m['samples'],8)
    def test_output(self):
        with tempfile.TemporaryDirectory() as d:
            root=save_input_psds({'RangeRate':[(np.arange(32.)*5,np.sin(np.arange(32.)))]},d,{})
            for name in ('RangeRate_PSD.npz','RangeRate_PSD.png','RangeRate_PSD.pdf','metadata.json'):
                self.assertGreater((root/name).stat().st_size,0)
if __name__=='__main__':unittest.main()
