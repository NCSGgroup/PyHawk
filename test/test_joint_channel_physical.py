"""Physical smoke test using repository ODE, gravity DLL, ACC and EOP data.
Skips only if the bundled physical runtime/data are unavailable.
"""
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from numpy.testing import assert_allclose
from src.Solver.JointChannel.channel import Settings, range_rate, write_json, read_arc
from src.Solver.JointChannel.feedback import integrate_state, run_feedback

ROOT = Path(__file__).resolve().parents[1]
AVAILABLE = all((ROOT/p).exists() for p in ('lib/GravHar.dll', 'data/ephemerides/de421.bsp', 'data/eop/eopc04_14_IAU2000.62-now'))


@unittest.skipUnless(AVAILABLE, 'physical runtime/EOP/ephemeris data unavailable')
class PhysicalFeedbackTests(unittest.TestCase):
    def test_actual_acc_initial_gravity_integration_and_next_iteration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            settings = Settings(ROOT/'setting/Calibrate', ROOT/'demo')
            s = settings.raw
            s['InterfaceConfig']['date_span'] = ['2013-05-01', '2013-05-31']
            s['InterfaceConfig']['PathOfFilesConfig']['temp_raw_data'] = str(root/'Raw')
            for name in ('GravityDesignParameterConfig', 'OrbitFitParameterConfig'):
                s[name]['AccelerometerConfig'].update(isScale=False, isRequired=True, AllisGlobal=False)
                s[name]['TransitionMatrixConfig'].update(isRequired=True, Parameter_Number=6, AllisGlobal=False)
            s['GravityDesignParameterConfig']['StokesCoefficientsConfig'].update(Parameter_Number=5, MinDegree=2, MaxDegree=2)
            for name in ('AccelerometerConfig_A','AccelerometerConfig_B','DesignAccelerometerConfig_A','DesignAccelerometerConfig_B'):
                s[name]['cali']['Scale'] = [False]*9
                s[name]['cali']['Bias_constant'] = [True]*3
                s[name]['cali']['Bias_trend'] = [True]*3
                s[name]['cali']['Bias_quadratic'] = [False]*3
            force = s['ForceModelConfig']
            force['include_ForceType'] = dict(NonConserveForce=True, GravHar=True, Relativity=False, PlanetaryPerturbation=False)
            force['include_GravHarType'] = {key:key=='ReferenceModel' for key in force['include_GravHarType']}
            force['RefGravModelConfig'].update(StaticModel='Gif48', Nmax=2, kind=0, TimeVariableDate=None)
            gravity_file = root/'central.gfc'
            gravity_file.write_text('max_degree 2\nearth_gravity_constant 398600441500000\nradius 6378136.3\nend_of_head\n' +
                                   '\n'.join('gfc %d %d %s 0 0 0'%(n,m,1 if n==0 else 0) for n in range(3) for m in range(n+1)))
            force['PathOfFilesConfig']['Gif48'] = str(gravity_file)
            s['AdjustOrbitConfig']['PathOfFilesConfig']['OrbitAdjustResTemp'] = str(root/'Initial')
            s['AdjustOrbitConfig']['PathOfFilesConfig']['RangeRateTemp'] = str(root/'QC')
            for key in ('PostOrbit','PostKBRR','PostParameter','CovarianceMatrix'):
                s['SolverConfig'][key] = str(root/key)
            for key in ('NormalEqTemp','ResultCS','StandardNEQTemp'):
                s['NEQConfig']['PathOfFilesConfig'][key] = str(root/key)
            s['AdjustOrbitConfig']['PathOfFilesConfig']['StandardKBRRTemp'] = str(root/'KBRR1C')
            for sat in ('A','B'):
                s['AccelerometerConfig_'+sat]['StandardACCTemp'] = str(root/'ACC1C')
            for folder in (root/'Raw',root/'Initial'/settings.tag,root/'QC'/settings.tag):
                folder.mkdir(parents=True)
            time = 420724800. + np.arange(121)*5
            omega = np.sqrt(398600441500000. / (7e6)**3)
            trajectories = []
            for phase in (0., 0.03):
                angle = (time-time[4])*omega+phase
                pos = np.column_stack((7e6*np.cos(angle),7e6*np.sin(angle),np.zeros(time.size)))
                vel = np.column_stack((-7e6*omega*np.sin(angle),7e6*omega*np.cos(angle),np.zeros(time.size)))
                trajectories.append(np.column_stack((time,pos,vel)))
            initial = np.vstack([x[4,1:7] for x in trajectories])
            acc = [np.zeros(6),np.zeros(6)]
            with h5py.File(root/'Initial'/settings.tag/'0_AB.hdf5','w') as h:
                h['TransitionMatrix'],h['Accelerometer'] = initial,np.array(acc)
            raw_path = root/'Raw'/(settings.tag+'.hdf5')
            with h5py.File(raw_path,'w') as h:
                for i,sat in enumerate(('A','B')):
                    h['GNV_%s_arc0'%sat] = trajectories[i]
                    h['ACC_%s_arc0'%sat] = np.column_stack((time,np.tile([1e-8,2e-8,-1e-8],(time.size,1))))
                    h['SCA_%s_arc0'%sat] = np.column_stack((time,np.ones(time.size),np.zeros((time.size,3))))
            baseline = root/'baseline'
            t,r,v = integrate_state(settings,0,initial,acc,np.zeros(5),baseline)
            self.assertEqual(t.size,121)
            with h5py.File(raw_path,'a') as h:
                for i,sat in enumerate(('A','B')):
                    h['KinematicOrbit_%s_arc0'%sat] = np.column_stack((t,r[:,i*3:i*3+3]))
                h['KBR_A_arc0'] = np.column_stack((t,np.zeros(t.size),range_rate(r,v)))
            with h5py.File(root/'QC'/settings.tag/'RangeRate_0.hdf5','w') as h:
                h['empirical_prefit_t'] = t
                h['design_matrix'] = np.empty((t.size,0))
            s['SolverConfig']['DesignMatrixTemp'] = str(baseline/'DesignMatrix')
            force['PathOfFilesConfig']['temp_non_conservative_data'] = str(baseline/'NonConservativeForce')
            for sat in ('A','B'):
                s['NEQConfig']['Orbit%sConfig'%sat].update(IsRequired=True,UseKinematicOrbit=True,FixKinematicOrbit=False)
            blocks,layout = read_arc(settings,0,False)
            nl = layout['local_per_satellite']
            self.assertEqual(nl,12)
            # Check each kind of physical correction independently against actual integration.
            for name in ('initial','acc','gravity'):
                ii,aa,gg = initial.copy(),[x.copy() for x in acc],np.zeros(5)
                dl,dg = np.zeros(2*nl),np.zeros(5)
                if name=='initial':
                    ii[0,0]+=0.001
                    dl[0]=0.001
                elif name=='acc':
                    aa[0][0]+=1e-9
                    dl[6]=1e-9
                else:
                    gg[0]=1e-10
                    dg[0]=1e-10
                _,r2,v2 = integrate_state(settings,0,ii,aa,gg,root/name)
                observed = range_rate(r2,v2)-range_rate(r,v)
                predicted = blocks[-1].global_dm@dg+blocks[-1].local_dm@dl
                self.assertGreater(np.max(np.abs(r2-r)),1e-7)
                self.assertGreater(np.max(np.abs(observed)),1e-10)
                assert_allclose(observed,predicted,atol=3e-9,rtol=0.01)
            # Exercise the complete feedback command and its state continuation.
            solution = root/'solution'
            solution.mkdir()
            settings.save(solution/'settings')
            dg,dl = np.zeros(5),np.zeros(2*nl)
            dg[0],dl[0],dl[6] = 1e-10,0.001,1e-9
            write_json(solution/'manifest.json',dict(arcs=[0],empirical=False,covariance=False,base_state=None,dm_root=str(baseline/'DesignMatrix')))
            write_json(solution/'complete.json',dict(status='complete'))
            with h5py.File(solution/'global.h5','w') as h:
                h['delta']=dg
                h['normal']=np.eye(5)
                h['rhs']=dg
            with h5py.File(solution/'arc_0.h5','w') as h:
                h.attrs['layout']=json.dumps(layout)
                h.attrs['local_rank']=2*nl
                for block in blocks[:-1]:
                    h[block.name+'/time']=block.time
                    h[block.name+'/prefit']=block.residual
                    h[block.name+'/linear_postfit']=block.residual-block.global_dm@dg-block.local_dm@dl
                h['local_delta']=dl
                h['RangeRate/time']=t
                h['RangeRate/prefit']=np.zeros(t.size)
                h['RangeRate/linear_postfit']=-(blocks[-1].global_dm@dg+blocks[-1].local_dm@dl)
            updated=run_feedback(solution,root/'updated')
            updated_parallel=run_feedback(solution,root/'updated_parallel',workers=2)
            with h5py.File(updated/'state.h5','r') as serial, h5py.File(updated_parallel/'state.h5','r') as parallel:
                for name in ('gravity_delta','arc_0/initial','arc_0/acc_A','arc_0/acc_B',
                             'arc_0/kbr_physical_prefit','arc_0/kbr_true_postfit',
                             'arc_0/Orbit_A/physical_prefit','arc_0/Orbit_B/physical_prefit'):
                    assert_allclose(serial[name][()],parallel[name][()],rtol=0,atol=1e-9)
            for name in ('Orbit_0_AB.hdf5','RangeRate_0.hdf5'):
                self.assertTrue((updated_parallel/'DesignMatrix'/settings.tag/name).is_file())
            import shutil
            twin = root/'solution_two_arcs'
            shutil.copytree(solution,twin)
            shutil.copyfile(twin/'arc_0.h5',twin/'arc_1.h5')
            twin_manifest=json.loads((twin/'manifest.json').read_text())
            twin_manifest['arcs']=[0,1]
            (twin/'manifest.json').write_text(json.dumps(twin_manifest))
            shutil.copyfile(root/'Initial'/settings.tag/'0_AB.hdf5',root/'Initial'/settings.tag/'1_AB.hdf5')
            shutil.copyfile(root/'QC'/settings.tag/'RangeRate_0.hdf5',root/'QC'/settings.tag/'RangeRate_1.hdf5')
            for prefix in ('Orbit_','RangeRate_'):
                suffix='_AB.hdf5' if prefix=='Orbit_' else '.hdf5'
                shutil.copyfile(baseline/'DesignMatrix'/settings.tag/(prefix+'0'+suffix),
                                baseline/'DesignMatrix'/settings.tag/(prefix+'1'+suffix))
            shutil.copyfile(baseline/'DesignMatrix'/settings.tag/'0_AB.hdf5',
                            baseline/'DesignMatrix'/settings.tag/'1_AB.hdf5')
            for sat in ('A','B'):
                shutil.copyfile(baseline/'NonConservativeForce'/(settings.tag+'_'+sat+'_0.hdf5'),
                                baseline/'NonConservativeForce'/(settings.tag+'_'+sat+'_1.hdf5'))
            with h5py.File(raw_path,'a') as h:
                for name in list(h.keys()):
                    if name.endswith('_arc0'):
                        h.copy(name,name[:-1]+'1')
            twins=run_feedback(twin,root/'updated_two_arcs',workers=2)
            with h5py.File(twins/'state.h5') as h:
                assert_allclose(h['arc_0/kbr_true_postfit'][()],h['arc_1/kbr_true_postfit'][()],atol=1e-9)
            for arc in (0,1):
                self.assertTrue((twins/'DesignMatrix'/settings.tag/('Orbit_%d_AB.hdf5'%arc)).is_file())
            with h5py.File(updated/'state.h5','r') as h:
                self.assertAlmostEqual(h['arc_0/acc_A'][0],1e-9)
                assert_allclose(h['gravity_delta'][()],dg)
                self.assertGreater(np.max(np.abs(h['arc_0/kbr_physical_prefit'][()])),1e-10)
                assert_allclose(h['arc_0/kbr_true_postfit'][()],h['arc_0/kbr_linear_prediction'][()],atol=4e-9)
            from src.Solver.JointChannel.workflow import publish_products
            products = publish_products(settings,solution,dg,np.zeros(5),updated,solution/'settings')
            from src.Solver.JointChannel.workflow import formal_covariance, export_residuals, export_acc
            cov_serial, stats_serial = formal_covariance(settings,solution)
            cov_parallel, stats_parallel = formal_covariance(settings,solution,workers=2)
            assert_allclose(cov_serial,cov_parallel,rtol=1e-12,atol=1e-12)
            self.assertEqual(stats_serial,stats_parallel)
            parallel_kbrr=export_residuals(settings,solution,dg,updated,l1c=True,workers=2)
            parallel_acc=export_acc(settings,solution,updated,workers=2)
            self.assertEqual(len(parallel_kbrr),len(products['KBRR1C']))
            self.assertEqual(len(parallel_acc),len(products['ACC1C']))
            self.assertTrue(Path(products['SH']).is_file())
            self.assertTrue(Path(products['NEQ1C']).is_file())
            self.assertTrue(products['ACC1C'])
            self.assertTrue(products['KBRR1C'])
            for product in products['ACC1C']+products['KBRR1C']:
                self.assertTrue(Path(product).is_file())
                self.assertTrue(str(Path(product)).startswith(str(root)))
            # A second feedback uses state.h5, not the original OrbitAdjustment values.
            manifest=dict(arcs=[0],empirical=False,base_state=str(updated/'state.h5'),dm_root=str(updated/'DesignMatrix'))
            (solution/'manifest.json').write_text(json.dumps(manifest))
            second=run_feedback(solution,root/'updated_again')
            with h5py.File(second/'state.h5','r') as h:
                self.assertAlmostEqual(h['arc_0/acc_A'][0],2e-9)
                assert_allclose(h['gravity_delta'][()],2*dg)
                self.assertAlmostEqual(h['arc_0/initial'][0,0],initial[0,0]+0.002,places=7)

            # Switch from saved fixed-scale fits to a common monthly absolute state.
            from src.Solver.JointChannel.monthly_scale import configure_monthly_scale, bootstrap_monthly_scale
            shutil.copyfile(root/'Initial'/settings.tag/'0_AB.hdf5',root/'Initial'/settings.tag/'1_AB.hdf5')
            configure_monthly_scale(settings)
            monthly = bootstrap_monthly_scale(settings,[0,1],root/'monthly',2)
            # A complete cache must skip propagation; an interrupted arc must
            # be recomputed without touching the valid peer or old checkpoint.
            reused = bootstrap_monthly_scale(settings,[0,1],root/'monthly_reused',2,cache=monthly)
            for arc in (0,1):
                checkpoint=json.loads((reused/'checkpoints'/('arc_%d.json'%arc)).read_text())
                self.assertTrue(checkpoint['reused'])
            (reused/'DesignMatrix'/settings.tag/'RangeRate_1.hdf5').unlink()
            recovered = bootstrap_monthly_scale(settings,[0,1],root/'monthly_recovered',2,cache=reused)
            self.assertTrue(json.loads((recovered/'checkpoints/arc_0.json').read_text())['reused'])
            self.assertFalse(json.loads((recovered/'checkpoints/arc_1.json').read_text())['reused'])
            with h5py.File(monthly/'state.h5') as h:
                common = h['monthly_scale'][()]
                ma = [h['arc_0/acc_'+sat][()] for sat in ('A','B')]
                for arc in (0,1):
                    for i,sat in enumerate(('A','B')):
                        assert_allclose(h['arc_%d/acc_%s'%(arc,sat)][:9],common[i],rtol=0,atol=0)
            mb,ml = read_arc(settings,0,False,False,monthly/'DesignMatrix')
            self.assertTrue(ml['scale_global'])
            self.assertEqual(mb[-1].global_dm.shape[1],23)
            with h5py.File(monthly/'DesignMatrix'/settings.tag/'Orbit_0_AB.hdf5') as h:
                mr,mv = h['r'][()],h['v'][()]
            from src.Interface.Accelerometer import AccCaliPar_V3, Accelerometer_V3
            from src.Interface.ArcOperation import ArcData
            rd = ArcData(settings.obj('InterfaceConfig'))
            p = s['GravityDesignParameterConfig']['AccelerometerConfig']
            ap = AccCaliPar_V3(rd,'A',0,p['X_AdjustLength'],p['Y_AdjustLength'],p['Z_AdjustLength'],
                              settings.obj('DesignAccelerometerConfig_A'))
            nominal = Accelerometer_V3(ap).updatePar(ma[0]).get_acceleration()
            jac = Accelerometer_V3(ap).updatePar(ma[0]).get_du2p()
            for column in range(9):
                x = ma[0].copy(); x[column] += .01
                physical = Accelerometer_V3(ap).updatePar(x).get_acceleration()
                assert_allclose((physical-nominal)/.01,jac[:,:,column],atol=1e-20,rtol=1e-10)
            perturbed = [x.copy() for x in ma]
            perturbed[0][0] += .01
            _,pr,pv = integrate_state(settings,0,initial,perturbed,np.zeros(5),root/'monthly_fd')
            dg = np.zeros(23); dg[0] = .01
            assert_allclose(range_rate(pr,pv)-range_rate(mr,mv),mb[-1].global_dm@dg,atol=3e-9,rtol=.02)

            # Monthly feedback must preserve the same absolute scale in both arcs.
            stage = root/'monthly_solution'; stage.mkdir()
            settings.save(stage/'settings')
            write_json(stage/'manifest.json',dict(arcs=[0,1],empirical=False,covariance=False,
                       base_state=str(monthly/'state.h5'),dm_root=str(monthly/'DesignMatrix')))
            write_json(stage/'complete.json',dict(status='complete'))
            with h5py.File(stage/'global.h5','w') as h:
                h['delta']=dg; h['normal']=np.eye(23); h['rhs']=dg
            for arc in (0,1):
                blocks,layout=read_arc(settings,arc,False,False,monthly/'DesignMatrix')
                with h5py.File(stage/('arc_%d.h5'%arc),'w') as h:
                    h.attrs['layout']=json.dumps(layout)
                    h['local_delta']=np.zeros(2*layout['local_per_satellite'])
                    for block in blocks:
                        h[block.name+'/time']=block.time
                        h[block.name+'/prefit']=block.residual
                        h[block.name+'/linear_postfit']=block.residual-block.global_dm@dg
            final=run_feedback(stage,root/'monthly_feedback',workers=2)
            with h5py.File(final/'state.h5') as h:
                assert_allclose(h['monthly_scale'][()],common+dg[:18].reshape(2,9),atol=1e-14)
                for arc in (0,1):
                    for i,sat in enumerate(('A','B')):
                        assert_allclose(h['arc_%d/acc_%s'%(arc,sat)][:9],h['monthly_scale'][i],atol=1e-14)
                    assert_allclose(h['arc_%d/kbr_true_postfit'%arc][()],
                                    h['arc_%d/kbr_linear_prediction'%arc][()],atol=3e-9)
            monthly_acc=export_acc(settings,stage,workers=2)
            self.assertTrue(monthly_acc)

            # Fit sparse, independent KIN positions (and epoch covariance),
            # while GNV supplies only the propagation seed and time grid.
            from src.Solver.AdjustOrbit import AdjustOrbit
            from src.SecDerivative.Instance.DualOrbitAndVar import Orbit_var_2nd_diff
            s['AdjustOrbitConfig']['OrbitConfig'].update(ObservationSource='KIN',UseEpochCovariance=True,SamplingSeconds=30.,SamplingOrigin='FirstEpoch')
            s['AdjustOrbitConfig']['PathOfFilesConfig']['StateVectorDataTemp']=str(root/'KinFitState')
            with h5py.File(raw_path,'a') as h:
                for i,sat in enumerate(('A','B')):
                    old=h['KinematicOrbit_%s_arc0'%sat][()]
                    keep=np.arange(i+1,len(old),2+i)
                    kin=np.column_stack((old[keep],np.tile([1e-4,2e-4,3e-4,0.,0.,0.],(len(keep),1))))
                    del h['KinematicOrbit_%s_arc0'%sat]
                    h['KinematicOrbit_%s_arc0'%sat]=kin
            fit=AdjustOrbit([settings.obj('AccelerometerConfig_'+sat) for sat in ('A','B')],settings.span,{'A':True,'B':True})
            fit.configure(SerDer=Orbit_var_2nd_diff,arcNo=0,AdjustOrbitConfig=settings.obj('AdjustOrbitConfig'),
                ODEConfig=settings.obj('ODEConfig'),FMConfig=settings.obj('ForceModelConfig'),
                ParameterConfig=settings.obj('OrbitFitParameterConfig'),InterfaceConfig=settings.obj('InterfaceConfig'),
                FrameConfig=settings.obj('FrameConfig')).calibrate(iteration=1)
            with h5py.File(root/'Initial'/settings.tag/'0_AB.hdf5') as h:
                self.assertEqual(h.attrs['orbit_observation_source'],'KinematicOrbit')
                self.assertEqual(h['Accelerometer'].shape,(2,6))
            self.assertEqual(fit._fit_observations['A'].shape[1],10)
            self.assertNotEqual(len(fit._fit_observations['A']),len(fit._fit_observations['B']))


if __name__=='__main__':
    unittest.main()
