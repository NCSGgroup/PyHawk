import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import h5py
import numpy as np
from numpy.testing import assert_allclose
from src.Solver.JointChannel.parallel import map_arcs
from src.Solver.JointChannel.feedback import write_design_matrices
from src.Solver.JointChannel.bootstrap_cache import input_fingerprints, validate_bootstrap_settings
from src.Solver.Parameterization import Parameterization
from src.Preference.Pre_Parameterization import ParameterConfig
from src.Auxilary.SSTpartial import SSTpartial


def scheduling_job(arc, directory):
    flag=Path(directory)/'third-started'
    if arc==0:
        end=time.monotonic()+8
        while not flag.exists() and time.monotonic()<end:
            time.sleep(.02)
        if not flag.exists():
            raise RuntimeError('head-of-line blocked idle worker')
    if arc==2:
        flag.write_text('started')
    return arc


def square(arc):
    return arc*arc


class BootstrapOptimizationTests(unittest.TestCase):
    def test_refill_does_not_wait_for_first_arc(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(sorted(map_arcs(scheduling_job,[0,1,2],(d,),2,ordered=False)),[0,1,2])

    def test_ordered_compatibility_and_single_worker(self):
        self.assertEqual(list(map_arcs(square,[3,1,2],workers=2)),[9,1,4])
        self.assertEqual(list(map_arcs(square,[3,1,2],workers=1,ordered=False)),[9,1,4])

    def test_block_projection_matches_legacy_full_matrix(self):
        for scale in (False,True):
            with self.subTest(scale=scale),tempfile.TemporaryDirectory() as d:
                p=ParameterConfig()
                p.TransitionMatrixConfig.update(isRequired=True,Parameter_Number=6,AllisGlobal=False)
                p.AccelerometerConfig.update(isRequired=True,isScale=scale,AllisGlobal=False)
                p.StokesCoefficientsConfig.update(isRequired=True,Parameter_Number=5,AllisGlobal=True)
                settings=SimpleNamespace(tag='span',obj=lambda name:p,
                    raw={'GravityDesignParameterConfig':{'AccelerometerConfig':{'isScale':scale}}})
                n,nodes=47,3
                rng=np.random.default_rng(11)
                rr=rng.normal(size=(n,6,1+6+nodes+5+(9 if scale else 0)))
                vv=rng.normal(size=rr.shape)
                t=np.arange(n)*5.
                Path(d,'span').mkdir()
                result=write_design_matrices(settings,2,Path(d),nodes,state=(t,rr,vv),block_size=7)
                pp=Parameterization().configure(p,nodes)
                r,lr,gr=pp.split_global_local(rr,True);v,lv,gv=pp.split_global_local(vv,True)
                pra,pva,prb,pvb=SSTpartial(r[:,:3],v[:,:3],r[:,3:],v[:,3:]).getPartial_RangeRate()
                def proj(pr,pv,dr,dv):
                    return np.einsum('ij,ijk->ik',pr,dr)+np.einsum('ij,ijk->ik',pv,dv)
                ba,bb=proj(pra,pva,lr[:,:3],lv[:,:3]),proj(prb,pvb,lr[:,3:],lv[:,3:])
                ga,gb=proj(pra,pva,gr[:,:3],gv[:,:3]),proj(prb,pvb,gr[:,3:],gv[:,3:])
                expected=np.column_stack((ga[:,:9],gb[:,:9],ga[:,9:]+gb[:,9:])) if scale else ga+gb
                with h5py.File(Path(d,'span/Orbit_2_AB.hdf5'),'r') as h:
                    for key,a in dict(r=r,v=v,local_r=lr,global_r=gr,local_v=lv,global_v=gv).items():
                        np.testing.assert_array_equal(h[key][()],a)
                with h5py.File(Path(d,'span/RangeRate_2.hdf5'),'r') as h:
                    assert_allclose(h['global'][()],expected,rtol=1e-14,atol=1e-14)
                    assert_allclose(h['local'][()],np.column_stack((ba,bb)),rtol=1e-14,atol=1e-14)
                np.testing.assert_array_equal(result[1],r)
                # Also exercise bounded on-disk recovery instead of in-memory input.
                with h5py.File(Path(d,'span/3_AB.hdf5'),'w') as h:
                    for key,a in dict(t=t,r=rr,v=vv).items():h[key]=a
                write_design_matrices(settings,3,Path(d),nodes,block_size=13)
                with h5py.File(Path(d,'span/RangeRate_3.hdf5'),'r') as h:
                    assert_allclose(h['global'][()],expected,rtol=1e-14,atol=1e-14)

    def test_checkpoint_rejects_changed_input(self):
        import json
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'span').mkdir()
            (root/'span.hdf5').write_bytes(b'raw')
            fit=root/'span/0_AB.hdf5';fit.write_bytes(b'fit')
            settings=SimpleNamespace(tag='span',raw={
                'InterfaceConfig':{'PathOfFilesConfig':{'temp_raw_data':d}},
                'AdjustOrbitConfig':{'PathOfFilesConfig':{'OrbitAdjustResTemp':d}}})
            (root/'manifest.json').write_text(json.dumps(dict(arcs=[0],inputs=input_fingerprints(settings,[0]))))
            with patch('src.Solver.JointChannel.bootstrap_cache.Settings',return_value=settings):
                validate_bootstrap_settings(settings,root)
                fit.write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError,'inputs changed'):
                    validate_bootstrap_settings(settings,root)

if __name__=='__main__':unittest.main()