import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import block_diag
from src.Solver.JointChannel.initial_weights import basis_residual, physical_covariance, wls_basis

class WLSBasisTests(unittest.TestCase):
    def test_two_stage_normal_equations_equal_physical_pullback(self):
        rng=np.random.default_rng(42)
        components=np.array([[4.,9.,16.,1.,.3,.5],[9.,4.,6.,.2,.1,.4]])
        solver=dict(InitialOrbitWeighting='KINRelative',GLSResidualSpace='InitialWLS',InitialOrbitMeanPrecision=.4,OrbitKinFactor=100.)
        for name in ('Orbit_A','RangeRate'):
            n=6 if name=='Orbit_A' else 2
            z=rng.normal(size=(n,n)); t=z@z.T+np.eye(n)
            x=rng.normal(size=(n,4))
            w0=np.column_stack([basis_residual(np.eye(n)[:,i],name,solver,components).reshape(-1) for i in range(n)])
            c=physical_covariance(t,name,solver,components)
            assert_allclose(c,c.T,rtol=0,atol=0)
            assert_allclose(x.T@np.linalg.solve(c,x),(w0@x).T@np.linalg.solve(t,w0@x),rtol=1e-12,atol=1e-12)
            # Includes the RHS in the final column, not only the design columns.
            assert_allclose(w0@c@w0.T,t,rtol=1e-12,atol=1e-12)
    def test_old_mode_and_invalid_mode(self):
        self.assertFalse(wls_basis({}))
        with self.assertRaises(ValueError): wls_basis({'GLSResidualSpace':'InitialWLS'})

if __name__=='__main__': unittest.main()
