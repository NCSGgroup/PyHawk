import unittest
import numpy as np
from scipy.fft import dct
from src.Solver.JointChannel.arc_weights import pooled_autocovariance

class ZeroFrequencyTest(unittest.TestCase):
    def test_only_dc_changes_and_default_is_legacy(self):
        arcs=[(np.arange(32.),2.+np.sin(np.arange(32.)))]
        old,pairs=pooled_autocovariance(arcs,1.,8)
        keep,_=pooled_autocovariance(arcs,1.,8,"Keep")
        np.testing.assert_array_equal(old,keep)
        new,newpairs=pooled_autocovariance(arcs,1.,8,"CopyFirstNonzero")
        np.testing.assert_array_equal(pairs,newpairs)
        a,b=dct(old,type=1),dct(new,type=1)
        np.testing.assert_allclose(b[0],b[1],rtol=1e-12)
        # Existing diagonal jitter is retained and changes by O(1e-10).
        np.testing.assert_allclose(b[1:],a[1:],rtol=1e-8,atol=1e-8)
        lag=np.abs(np.arange(9)[:,None]-np.arange(9)[None,:])
        np.linalg.cholesky(new[lag])
        self.assertGreater(abs(a[0]-b[0]),1.)
    def test_invalid_policy_fails(self):
        with self.assertRaisesRegex(ValueError,"CovarianceZeroFrequencyPolicy"):
            pooled_autocovariance([(np.arange(4.),np.ones(4))],1.,2,"typo")
if __name__=="__main__":unittest.main()
