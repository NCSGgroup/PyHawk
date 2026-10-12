import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.Solver.JointChannel.channel import Settings,read_arc
from src.Solver.JointChannel.workflow import run_calibrate_joint,joint_options
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--setting-dir',required=True);p.add_argument('--bootstrap',required=True);p.add_argument('--check-only',action='store_true');a=p.parse_args()
    config=Path(a.setting_dir).resolve();boot=Path(a.bootstrap).resolve();os.chdir(Path(__file__).resolve().parent)
    cal=json.loads((config/'CalibrateOrbitConfig.json').read_text());settings=Settings(config,Path.cwd());checkpoint=json.loads((boot/'complete.json').read_text());arcs=checkpoint['arcs']
    assert checkpoint['scale_scope']==settings.span
    assert settings.raw['AdjustOrbitConfig']['OrbitConfig']['ObservationSource']=='GNV'
    assert settings.raw['SolverConfig']['InitialOrbitWeighting'] in ('Unit','KINRelative')
    assert settings.raw['SolverConfig']['GLSResidualSpace']=='Physical'
    from src.Interface.ArcOperation import ArcData
    from src.Preference.EnumType import Payload
    reader=ArcData(settings.obj('InterfaceConfig'))
    for arc in arcs:
        for sat in 'AB':
            payload=Payload.KinematicOrbit if settings.raw['NEQConfig']['Orbit'+sat+'Config']['UseKinematicOrbit'] else Payload.GNV
            assert len(reader.getData(arc,payload,sat))>0
        assert (boot/'DesignMatrix'/settings.tag/('Orbit_%d_AB.hdf5'%arc)).is_file()
    print('Validated configured observations and existing dynamics:',len(arcs),'arcs',flush=True)
    if not a.check_only:
        paths=settings.raw['InterfaceConfig']['PathOfFilesConfig']
        run_calibrate_joint(SimpleNamespace(),str(config),arcs,joint_options(cal['JointEstimationConfig'],cal['ParallelControlConfig']),True,SimpleNamespace(**cal['GLSIterationConfig']),str(Path(paths['report_arcft'])/(settings.tag+'.txt')),str(Path(paths['report_arc'])/(settings.tag+'.txt')),resume_bootstrap=boot)
