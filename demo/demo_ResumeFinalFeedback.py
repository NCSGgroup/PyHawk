import argparse,json,sys,os
from pathlib import Path
import h5py,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.Solver.JointChannel.channel import Settings,write_json
from src.Solver.JointChannel.workflow import load_result,publish_products,joint_options
from src.Solver.JointChannel.feedback_parallel import run_feedback_parallel
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--configuration',required=True);a=p.parse_args()
    run=Path(a.run).resolve();config=Path(a.configuration).resolve();os.chdir(Path(__file__).resolve().parent)
    if (run/'complete.json').exists():raise ValueError('already completed')
    source=run/'outer_01_gls_04';m,delta=load_result(source)
    assert (source/'complete.json').is_file()
    settings=Settings(source/'settings',source);cal=json.loads((config/'CalibrateOrbitConfig.json').read_text());opt=joint_options(cal['JointEstimationConfig'],cal['ParallelControlConfig'])
    baseline=np.asarray([settings.raw['DesignAccelerometerConfig_'+sat]['cali_value']['Scale'] for sat in 'AB'])
    with h5py.File(m['base_state'],'r') as h:
        base=np.r_[(h['monthly_scale'][()]-baseline).reshape(-1),h['gravity_delta'][()]]
    cumulative=base+opt['Damping']*delta
    out=run/'state_02_verified_resume'
    feedback=run_feedback_parallel(source,out,opt['Damping'],opt['FeedbackProcesses'],cache=run/'state_02')
    paths=settings.raw['InterfaceConfig']['PathOfFilesConfig']
    products=publish_products(settings,source,cumulative,base,feedback,str(config),str(Path(paths['report_arcft'])/(settings.tag+'.txt')),str(Path(paths['report_arc'])/(settings.tag+'.txt')),workers=opt['ArcProcesses'])
    scale=run/'monthly_scale.json';write_json(scale,dict(scope=settings.span,baseline=baseline.tolist(),increment=cumulative[:18].reshape(2,9).tolist(),absolute=(baseline+cumulative[:18].reshape(2,9)).tolist()));products['MonthlyScale']=str(scale)
    write_json(run/'complete.json',dict(products=products,resumed_from=str(source),feedback=str(feedback),note='Reused completed GLS and valid feedback arcs; no GLS rerun'))
    print('Completed feedback resume and products:',products,flush=True)
