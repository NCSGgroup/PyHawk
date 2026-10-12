import argparse,json,os,sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.Solver.JointChannel.channel import Settings
from src.Solver.JointChannel.workflow import run_calibrate_joint,joint_options,load_result
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',required=True)
    parser.add_argument('--calibration',required=True)
    parser.add_argument('--setting-dir', help='Override only GLSResidualSpace in checkpoint settings')
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    seed=Path(args.seed).resolve();cal=json.loads(Path(args.calibration).read_text())
    os.chdir(Path(__file__).resolve().parent)
    manifest,_=load_result(seed)
    assert (seed/'complete.json').is_file() and not manifest['covariance']
    settings=Settings(seed/'settings',seed)
    setting_dir=seed/'settings'
    if args.setting_dir:
        import copy
        setting_dir=Path(args.setting_dir).resolve()
        selected=Settings(setting_dir,Path.cwd())
        expected=copy.deepcopy(settings.raw)
        expected['SolverConfig']['GLSResidualSpace']=selected.raw['SolverConfig'].get('GLSResidualSpace','Physical')
        before=expected['SolverConfig'].get('ArcWeightingConfig',{})
        after=selected.raw['SolverConfig'].get('ArcWeightingConfig',{})
        if before.get('Mode') != after.get('Mode'):
            if after.get('Mode') != 'Off' or any(before.get(k,{}) for k in ('PriorPrecision','VarianceFactors','GroupVarianceFactors')):
                raise ValueError('cannot reuse weighted seed when disabling arc weights')
            before['Mode']='Off'
        if expected != selected.raw:
            raise ValueError('only GLSResidualSpace or disabling neutral arc weights may differ from initial checkpoint')
        settings=selected
    assert settings.raw['SolverConfig'].get('GLSResidualSpace','Physical') in ('Physical','InitialWLS')
    for arc in manifest['arcs']:
        import h5py
        with h5py.File(seed/('arc_%d.h5'%arc),'r') as h:
            for name in ('Orbit_A','Orbit_B','RangeRate'):
                assert name+'/linear_postfit' in h
    options=joint_options(cal['JointEstimationConfig'],cal['ParallelControlConfig'])
    print('Validated WLS checkpoint:',seed,'arcs:',len(manifest['arcs']),'workers:',options['ArcProcesses'],flush=True)
    if not args.check_only:
        paths=settings.raw['InterfaceConfig']['PathOfFilesConfig']
        run_calibrate_joint(SimpleNamespace(),str(setting_dir),manifest['arcs'],options,
            cal['StepControlConfig']['isGLS'],SimpleNamespace(**cal['GLSIterationConfig']),
            str(Path(paths['report_arcft'])/(settings.tag+'.txt')),
            str(Path(paths['report_arc'])/(settings.tag+'.txt')),resume_seed=seed)
