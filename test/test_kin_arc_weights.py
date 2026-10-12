import json
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import cholesky,solve_triangular
from src.Solver.OrbitObservations import align_positions,fit_position_increment,epoch_covariance,orbit_payload
from src.Preference.EnumType import Payload
from src.Solver.JointChannel.core import ObservationBlock,reduce_arc
from src.Solver.JointChannel.arc_weights import precision_factor,block_statistics,update_variances,pooled_autocovariance,covariance_probes

class KinWeightsTests(unittest.TestCase):
    def test_kin_frame_conversion_preserves_covariance_and_fractional_epochs(self):
        from src.Interface.KinematicOrbit import KinematicOrbit, _kinematic_itrs_to_gcrs
        from types import SimpleNamespace
        r=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        class TestFrame:
            def setTime(self,t):
                self.last=t
                return self
            getRotationMatrix=(r,np.zeros((3,3)))
        frame=TestFrame()
        rows=np.array([[2.25,1.,2.,3.,1.,2.,3.,.1,.2,.3,99.]])
        result=_kinematic_itrs_to_gcrs(rows,frame)
        assert_allclose(result[0,1:4],[2.,-1.,3.])
        assert_allclose(result[0,4:10],[2.,1.,3.,-.1,.3,-.2])
        self.assertEqual(frame.last,2.25);self.assertEqual(result[0,10],99.)
        l1b=SimpleNamespace(is_GNV_GCRS=True)
        kin=KinematicOrbit(l1b);kin._KiOr_A=rows;kin._KiOr_B=np.empty((0,11));kin.isGCRS=False
        kin.GNVandKiOrbitItrs2Gcrs(frame)
        assert_allclose(l1b.KiOrb_A,result);self.assertTrue(kin.isGCRS)
        kin.GNVandKiOrbitItrs2Gcrs(frame)
        assert_allclose(l1b.KiOrb_A,result)

    def test_sparse_positions_and_covariance_columns_are_not_velocities(self):
        t=420724800.+np.arange(20)*5
        rng=np.random.default_rng(7);dm=rng.normal(size=(20,3,4));delta=np.array([.2,-.1,.05,.3])
        xyz=np.zeros((20,3,5));xyz[:,:,1:]=dm
        idx=np.array([0,2,3,7,8,11,18]);obs=np.column_stack((t[idx],dm[idx]@delta,
            np.tile([1.,2.,3.,.1,.2,.1],(len(idx),1))))
        assert_allclose(fit_position_increment(obs,t,xyz,True),delta,atol=1e-12)
        rows,matched=align_positions(obs,t)
        assert_allclose(matched,idx)
        self.assertEqual(orbit_payload({'ObservationSource':'KIN'}),Payload.KinematicOrbit)
        with self.assertRaises(ValueError):fit_position_increment(obs[:,:4],t,xyz,True)

    def test_alignment_rejects_duplicate_and_missing_epochs_without_interpolation(self):
        obs=np.array([[0.,1.,2.,3.],[2.5,1.,2.,3.],[5.,1.,2.,3.],[10.,np.nan,2.,3.]])
        rows,idx=align_positions(obs,[0.,5.,10.]);assert_allclose(rows,[0,2]);assert_allclose(idx,[0,1])
        with self.assertRaises(ValueError):align_positions(np.tile(obs[0],(2,1)),[0.,5.])
        with self.assertRaises(ValueError):epoch_covariance([[1.,1.,1.,2.,0.,0.]])
        phase=np.column_stack((np.arange(5.,185.,10.),np.ones((18,3))))
        rows,_=align_positions(phase,np.arange(0.,185.,5.),60.,'FirstEpoch')
        assert_allclose(phase[rows,0],[5.,65.,125.])

    def test_precision_is_prior_divided_by_variance_not_sigma(self):
        solver={'ArcWeightingConfig':{'Mode':'Fixed','PriorPrecision':{'RangeRate':{'3':.5}},
                  'VarianceFactors':{'RangeRate':{'3':4.}},'GroupVarianceFactors':{'RangeRate':2.}}}
        self.assertEqual(precision_factor(solver,3,'RangeRate'),.0625)
        b=ObservationBlock('r',np.ones((2,1)),np.empty((2,0)),np.ones(2),2.,.0625)
        assert_allclose(b.whiten()[0],np.ones((2,1))*np.sqrt(.0625/2))

    def test_redundancy_includes_joint_local_and_global_leverage(self):
        rng=np.random.default_rng(8);blocks=[]
        for name,n in [('Orbit_A',14),('Orbit_B',17),('RangeRate',21)]:
            blocks.append(ObservationBlock(name,rng.normal(size=(n,3)),rng.normal(size=(n,4)),rng.normal(size=n),np.arange(n)+1.))
        red=reduce_arc(blocks);delta=np.linalg.solve(red.normal,red.rhs);local=red.recover(delta)
        # Deterministic probes: PP^T=N^-1, so trace is exact for this test.
        probes=solve_triangular(cholesky(red.normal,lower=True).T,np.eye(3),lower=False)
        stats=block_statistics(blocks,red.local_global,delta,local,probes)
        white=[b.whiten() for b in blocks];design=np.column_stack((np.concatenate([w[0] for w in white]),np.concatenate([w[1] for w in white])))
        hat=design@np.linalg.inv(design.T@design)@design.T;start=0
        for block in blocks:
            n=len(block.residual);assert_allclose(stats[block.name]['redundancy'],n-np.trace(hat[start:start+n,start:start+n]),atol=1e-12);start+=n
        assert_allclose(sum(s['redundancy'] for s in stats.values()),52-7,atol=1e-12)
        self.assertTrue(np.isfinite(covariance_probes(red.normal,64,1)).all())

    def test_variance_update_downweights_noisy_arc_and_keeps_prior(self):
        solver={'OrbitCovarianceMode':'FixedDiagonal','ArcWeightingConfig':{'Mode':'VCE','UpdateDamping':1.,'MaxAdjustment':100.,'PriorPrecision':{'RangeRate':{'2':.5}}}}
        statistics={str(i):{'RangeRate':{'weighted_rss':10*q,'redundancy':10.},'Orbit_A':{'weighted_rss':40.,'redundancy':10.}} for i,q in enumerate([1.,1.,9.])}
        c,d=update_variances(solver,statistics)
        self.assertEqual(c['VarianceFactors']['RangeRate']['2'],9.)
        self.assertEqual(c['GroupVarianceFactors']['Orbit_A'],4.)
        self.assertEqual(c['PriorPrecision']['RangeRate']['2'],.5)
        json.dumps(d);self.assertFalse(d['converged'])

    def test_covariance_retains_constant_residual_mean(self):
        cov,pairs=pooled_autocovariance([(np.arange(8.),np.full(8,2.))],1.,3)
        assert_allclose(pairs,[8,7,6,5])
        # Constant residual power survives lag taper and the 1e-4 PSD floor.
        assert_allclose(cov,[4.,3.,2.,1.],rtol=0,atol=4e-4)

    def test_covariance_pairs_never_cross_arcs_or_fill_gaps(self):
        arcs=[(np.array([0.,1.,3.]),np.array([1.,-1.,2.])),(np.array([4.,5.,7.]),np.array([2.,-2.,3.]))]
        cov,pairs=pooled_autocovariance(arcs,1.,4)
        assert_allclose(pairs,[6,2,2,2,0])
        # Verify the actual supported-lag matrix, with zero outside support.
        lag=np.abs(np.arange(5)[:,None]-np.arange(5)[None,:]);np.linalg.cholesky(cov[lag])

if __name__=='__main__':unittest.main()
