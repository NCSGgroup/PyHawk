import unittest
from types import SimpleNamespace
import numpy as np
from numpy.testing import assert_allclose
from src.Solver.JointChannel.monthly_scale import bootstrap_parameters, validate_shared_scale
from src.Solver.JointChannel.feedback import update_physical_state
from src.Solver.JointChannel.core import ObservationBlock, reduce_arc, solve_global
from src.Auxilary.StandardNEQ import StandardNEQ

class MonthlyScaleTests(unittest.TestCase):
    def test_fixed_and_selected_scale_bootstrap(self):
        scale=np.arange(9.)
        assert_allclose(bootstrap_parameters([1.,2.],0,11,scale),np.r_[scale,1.,2.])
        assert_allclose(bootstrap_parameters([99.,88.,1.,2.],2,11,scale),np.r_[scale,1.,2.])
        with self.assertRaises(ValueError):
            bootstrap_parameters([1.],0,11,scale)

    def test_absolute_shared_update(self):
        common=np.tile(np.eye(3).reshape(9),(2,1))
        acc=[np.r_[row,[1.,2.]] for row in common]
        delta=np.arange(20.)*.001
        for j in (0,1):
            initial,new,gravity=update_physical_state(np.ones((2,6))*j,*acc,np.ones(16),delta,np.zeros(2),True,.5)
            validate_shared_scale(new,common+.5*delta[:18].reshape(2,9))
            assert_allclose(gravity,.5*delta[18:])
        with self.assertRaises(ValueError):
            validate_shared_scale([acc[0]+.1,acc[1]],common)

    def test_two_arc_joint_solution_and_scale_schur(self):
        rng=np.random.default_rng(42)
        gs=[];ls=[];ys=[];reductions=[]
        for arc in range(2):
            g=rng.normal(size=(80,23)); l=rng.normal(size=(80,4)); y=rng.normal(size=80)
            gs.append(g);ls.append(l);ys.append(y)
            reductions.append(reduce_arc([ObservationBlock('all',g,l,y)]))
        design=np.block([[gs[0],ls[0],np.zeros((80,4))],[gs[1],np.zeros((80,4)),ls[1]]])
        expected=np.linalg.lstsq(design,np.r_[ys[0],ys[1]],rcond=None)[0][:23]
        n=sum(r.normal for r in reductions); rhs=sum(r.rhs for r in reductions)
        assert_allclose(np.linalg.solve(n,rhs),expected,atol=1e-12)
        writer=StandardNEQ(); writer._StandardNEQ__scaleParameterCount=18
        reduced,rr=writer._StandardNEQ__keep_gravity_only(n,rhs)
        assert_allclose(np.linalg.solve(reduced,rr),expected[18:],atol=1e-12)
        assert_allclose(np.linalg.inv(reduced),np.linalg.inv(n)[18:,18:],atol=1e-12)

if __name__=='__main__': unittest.main()
