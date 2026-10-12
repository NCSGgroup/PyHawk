"""Resume an incomplete monthly-scale bootstrap, then run the saved joint solve."""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.Solver.JointChannel.channel import Settings
from src.Solver.JointChannel.monthly_scale import bootstrap_monthly_scale
from src.Solver.JointChannel.bootstrap_cache import validate_bootstrap_settings, input_fingerprints
from src.Solver.JointChannel.workflow import run_calibrate_joint, joint_options


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-bootstrap')
    parser.add_argument('--fresh-bootstrap',action='store_true',help='Rebuild only bootstrap if the old checkpoint was removed')
    parser.add_argument('--output-bootstrap',required=True)
    parser.add_argument('--setting-dir',required=True)
    parser.add_argument('--arcs-json',required=True)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    os.chdir(Path(__file__).resolve().parent)
    if bool(args.source_bootstrap)==bool(args.fresh_bootstrap):
        parser.error('select exactly one of --source-bootstrap or --fresh-bootstrap')
    source=Path(args.source_bootstrap).resolve() if args.source_bootstrap else None
    output,config=map(lambda p:Path(p).resolve(),(args.output_bootstrap,args.setting_dir))
    if output.exists():
        raise FileExistsError('Use a fresh output directory; pass previous output as source to resume it')
    settings=Settings(config,Path.cwd())
    if source is not None:
        validate_bootstrap_settings(settings,source)
    arcs=json.loads(Path(args.arcs_json).read_text())
    if not arcs or len(set(arcs))!=len(arcs) or any(type(a)!=int or a<0 for a in arcs):
        raise ValueError('invalid explicit arc list')
    manifest=source/'manifest.json' if source is not None else None
    if manifest is not None and manifest.exists() and json.loads(manifest.read_text())['arcs']!=arcs:
        raise ValueError('arc list differs from checkpoint')
    cal=json.loads((config/'CalibrateOrbitConfig.json').read_text())
    options=joint_options(cal['JointEstimationConfig'],cal['ParallelControlConfig'])
    if not options['Enabled'] or not options['MonthlyScaleEnabled']:
        raise ValueError('requires joint monthly-scale estimation')
    before=input_fingerprints(settings,arcs)
    # Do not wait until a costly integration to discover missing observations/QC.
    from src.Interface.ArcOperation import ArcData
    from src.Preference.EnumType import Payload
    import h5py
    reader=ArcData(settings.obj('InterfaceConfig'))
    for arc in arcs:
        for sat in 'AB':
            payload=Payload.KinematicOrbit if settings.raw['NEQConfig']['Orbit'+sat+'Config']['UseKinematicOrbit'] else Payload.GNV
            if len(reader.getData(arc,payload,sat))==0:
                raise ValueError('missing orbit observations arc %d %s'%(arc,sat))
        qc=Path(settings.raw['AdjustOrbitConfig']['PathOfFilesConfig']['RangeRateTemp'])/settings.tag/('RangeRate_%d.hdf5'%arc)
        with h5py.File(qc,'r') as h:
            for key in ('empirical_prefit_t','design_matrix'):
                if key not in h:
                    raise ValueError('missing KBRR QC basis %s arc %d'%(key,arc))
    print('Preflight passed: %d arcs, %d dynamics workers, %d NEQ workers'%
          (len(arcs),options['FeedbackProcesses'],options['ArcProcesses']),flush=True)
    if args.check_only:
        return
    boot=bootstrap_monthly_scale(settings,arcs,output,options['FeedbackProcesses'],cache=source)
    if input_fingerprints(settings,arcs)!=before:
        raise ValueError('inputs changed; refusing joint solve')
    paths=settings.raw['InterfaceConfig']['PathOfFilesConfig']
    run_calibrate_joint(SimpleNamespace(),str(config),arcs,options,
                        cal['StepControlConfig']['isGLS'],SimpleNamespace(**cal['GLSIterationConfig']),
                        str(Path(paths['report_arcft'])/(settings.tag+'.txt')),
                        str(Path(paths['report_arc'])/(settings.tag+'.txt')),resume_bootstrap=boot)


if __name__=='__main__':
    main()