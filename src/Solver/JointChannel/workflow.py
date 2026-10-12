"""CalibrateOrbit dispatch and canonical products for the opt-in joint channel."""
import json
from pathlib import Path
from datetime import datetime
import uuid
import h5py
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .channel import Settings, read_arc, run_joint, write_json, epoch_indices
from .feedback import run_feedback
from .core import ObservationBlock

DEFAULTS = dict(Enabled=False, FeedbackEnabled=False, MaxOuterIterations=2,
                FeedbackIterations=None,
                EstimateEmpirical=True, InitialUseCovariance=False, Damping=1.0,
                Rcond=1e-12, MonthlyScaleEnabled=False, SaveIterationProducts=False)


def joint_options(config, parallel=None):
    options = dict(DEFAULTS)
    parallel = parallel or {}
    options['ArcProcesses'] = parallel.get('GetNEQProcess', 1)
    options['FeedbackProcesses'] = parallel.get('GetGravityDesignMatProcess', 1)
    options.update(config or {})
    for key in ('Enabled','FeedbackEnabled','EstimateEmpirical','InitialUseCovariance','MonthlyScaleEnabled','SaveIterationProducts'):
        if not isinstance(options[key], bool):
            raise ValueError(key+' must be a JSON boolean')
    if options['MonthlyScaleEnabled'] and not options['Enabled']:
        raise ValueError('MonthlyScaleEnabled requires JointEstimationConfig.Enabled=true')
    if options['FeedbackEnabled'] and not options['Enabled']:
        raise ValueError('FeedbackEnabled requires JointEstimationConfig.Enabled=true')
    feedback_iterations = options['FeedbackIterations']
    if feedback_iterations is not None:
        if isinstance(feedback_iterations, bool) or not isinstance(feedback_iterations, int) or feedback_iterations < 0:
            raise ValueError('FeedbackIterations must be a nonnegative integer or null')
        if feedback_iterations > 0 and not options['FeedbackEnabled']:
            raise ValueError('FeedbackIterations > 0 requires FeedbackEnabled=true')
    if options['Enabled']:
        for key in ('ArcProcesses', 'FeedbackProcesses'):
            value = options[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(key+' must be a positive integer')
        count = options['MaxOuterIterations']
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError('MaxOuterIterations must be a positive integer')
        if not 0 < options['Damping'] <= 1 or not 0 < options['Rcond'] < 1:
            raise ValueError('invalid joint damping or rank threshold')
    return options


def load_result(solution):
    with open(Path(solution)/'manifest.json', encoding='utf-8') as f:
        manifest = json.load(f)
    with h5py.File(Path(solution)/'global.h5', 'r') as h:
        delta = h['delta'][()]
    return manifest, delta


def save_h5(path, values, attributes=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as h:
        for name, value in values.items():
            h.create_dataset(name, data=value)
        for name, value in (attributes or {}).items():
            h.attrs[name] = value


def export_residuals(settings, solution, cumulative, feedback=None, l1c=False, workers=1):
    from .products_parallel import export_residuals_parallel
    return export_residuals_parallel(settings, solution, cumulative, feedback, l1c, int(workers))


def formal_covariance(settings, solution, workers=1):
    """Absolute GLS covariance, or weighted OLS variance with all local DOFs."""
    manifest, delta = load_result(solution)
    with h5py.File(Path(solution)/'global.h5', 'r') as h:
        normal = h['normal'][()]
    scale = np.sqrt(np.diag(normal))
    scaled = normal / scale[:,None] / scale[None,:]
    covariance = cho_solve(cho_factor(scaled, lower=True), np.eye(scale.size))
    covariance /= scale[:,None]
    covariance /= scale[None,:]
    rss, count, locals_count = 0., 0, 0
    from .products_parallel import rss_parallel
    for arc_rss, arc_count, local_rank in rss_parallel(solution, manifest['arcs'], int(workers)):
        rss += arc_rss
        count += arc_count
        locals_count += local_rank
    dof = count - locals_count - delta.size
    if dof <= 0:
        raise ValueError('nonpositive joint residual degrees of freedom')
    factor = 1.0 if manifest['covariance'] else rss/dof
    return covariance*factor, dict(weighted_rss=rss, residual_dof=dof, variance_factor=factor)


def export_acc(settings, solution, feedback=None, workers=1):
    from .products_parallel import export_acc_parallel
    return export_acc_parallel(settings, solution, feedback, int(workers))


def publish_products(settings, solution, cumulative, base_global, feedback, setting_directory,
                     arc_filter_path=None, arc_definition_path=None, workers=1):
    from src.ForceModel.RefGravModel import RefGravModel
    from src.Auxilary.SortCS import SortCS
    from src.Auxilary.StandardSH import StandardSH
    from src.Auxilary.StandardNEQ import StandardNEQ
    from src.Auxilary.StandardConfig import StandardConfig
    neq = settings.raw['NEQConfig']['PathOfFilesConfig']
    solver = settings.raw['SolverConfig']
    sh = settings.raw['GravityDesignParameterConfig']['StokesCoefficientsConfig']
    skip = 18 if settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']['isScale'] else 0
    force = settings.obj('ForceModelConfig')
    force.resolve_time_variable_date(settings.span)
    reference = RefGravModel().configure(force).getStaticModel()
    covariance, statistics = formal_covariance(settings,solution,workers=workers)
    save_h5(Path(solver['CovarianceMatrix'])/('-'.join(settings.span)+'.hdf5'),dict(cov=covariance),statistics)
    with h5py.File(Path(solution)/'global.h5','r') as h:
        normal,rhs = h['normal'][()],h['rhs'][()]
    # Change the increment origin back to the original reference for legacy NEQ consumers.
    rhs = rhs + normal @ base_global
    save_h5(Path(neq['NormalEqTemp'])/settings.tag/('-'.join(settings.span)+'-NormalEquation.hdf5'),
            dict(NMatrix=normal,LMatrix=rhs[:,None]),dict(joint_channel=True,parameter_number=cumulative.size,
                 linearization='last outer iterate; RHS shifted to original reference'))
    snx = StandardNEQ().configure(neq.get('StandardNEQTemp','../temp/StandardNEQ'),settings.span,
        sh['MaxDegree'],sh['MinDegree'],sh['SortMethod'],skip,apriori_model=reference).save(normal,rhs)
    sort = SortCS(sh['MaxDegree'],sh['MinDegree'],sh['SortMethod'])
    c,s = sort.invert(cumulative[skip:])
    sc,ss = sort.invert(np.sqrt(np.maximum(np.diag(covariance)[skip:],0)))
    rc,rs = reference.getCS(sh['MaxDegree'])
    product = StandardSH().configure(neq['ResultCS'],settings.span,sh['MaxDegree'],c+rc,s+rs,sc,ss,
        mission=settings.raw['InterfaceConfig']['mission'],reference_model=reference,
        unused_days=settings.raw['InterfaceConfig'].get('date_delete',[])).unfiltered()
    archive = StandardConfig(setting_directory)
    archived_settings = archive.save(product)
    if skip or settings.raw['SolverConfig'].get('ArcWeightingConfig',{}).get('Mode','Off')!='Off':
        # Archive the effective design layout, alongside original fitting masks.
        for name in ('GravityDesignParameterConfig','DesignAccelerometerConfig_A','DesignAccelerometerConfig_B','SolverConfig'):
            StandardConfig._atomic_copy(Path(solution)/'settings'/(name+'.json'),
                                        archived_settings/(name+'.json'))
    archive.archive_quality_control(product,arc_filter_path,arc_definition_path)
    kbrr = export_residuals(settings,solution,cumulative,feedback,l1c=True,workers=workers)
    acc = export_acc(settings,solution,feedback,workers=workers)
    return dict(SH=str(product),NEQ1C=str(snx),KBRR1C=[str(p) for p in kbrr],ACC1C=acc,statistics=statistics)


def run_calibrate_joint(owner, setting_directory, arcs, options, is_gls, gls,
                        arc_filter_path=None, arc_definition_path=None, resume_seed=None, resume_bootstrap=None):
    options = joint_options(options)
    if not options['Enabled']:
        return None
    settings = Settings(setting_directory,Path.cwd())
    from .monthly_scale import configure_monthly_scale, bootstrap_monthly_scale
    monthly_scale = options['MonthlyScaleEnabled'] or settings.raw.get('GravityDesignParameterConfig', {}).get('AccelerometerConfig', {}).get('isScale', False)
    if monthly_scale:
        configure_monthly_scale(settings)
    settings.raw['SolverConfig'].pop('CovarianceFiles',None)
    settings.raw['SolverConfig']['OrbitCovarianceMode'] = gls.OrbitCovarianceMode
    from .arc_weights import weight_options,update_variances,read_statistics,estimate_monthly_noise
    settings.raw['SolverConfig']['SaveIterationProducts']=options['SaveIterationProducts']
    arc_weight_mode=weight_options(settings.raw['SolverConfig'])['Mode']
    from .initial_weights import initial_mode
    kin_seed = initial_mode(settings.raw['SolverConfig']) == 'KINRelative'
    from .initial_weights import wls_basis
    wls_basis(settings.raw['SolverConfig'])
    if kin_seed:
        if options['InitialUseCovariance']:
            raise ValueError('KINRelative initial solve must not reuse a previous GLS covariance')
        if not is_gls or gls.OrbitCovarianceMode != 'EstimatedFull':
            raise ValueError('KINRelative seed requires subsequent EstimatedFull GLS')
        if settings.raw['SolverConfig'].get('CovariancePoolingMode') != 'WithinArc':
            raise ValueError('KINRelative seed requires WithinArc residual covariance pooling')
    if settings.raw['SolverConfig'].get('CovariancePoolingMode','Legacy') not in ('Legacy','WithinArc'):
        raise ValueError('CovariancePoolingMode must be Legacy or WithinArc')
    if arc_weight_mode!='Off' or settings.raw['SolverConfig'].get('CovariancePoolingMode')=='WithinArc':
        settings.raw['SolverConfig']['CovarianceEpochMapping']='PhysicalTime'
    if arc_weight_mode=='VCE' and (not is_gls or gls.MaxIteration<2):
        raise ValueError('arc VCE requires GLS and at least two GLS iterations')
    if (is_gls or options['InitialUseCovariance']) and gls.OrbitCovarianceMode == 'ITSGEpoch':
        for sat in ('A', 'B'):
            orbit = settings.raw['NEQConfig']['Orbit'+sat+'Config']
            if orbit['IsRequired'] and not orbit['UseKinematicOrbit']:
                raise ValueError(
                    'ITSGEpoch requires kinematic XYZ covariance for satellite '+sat+
                    '; NEQConfig.Orbit'+sat+'Config.UseKinematicOrbit is false. '
                    'Set GLSIterationConfig.OrbitCovarianceMode to EstimatedFull '
                    'for GNV orbit observations, or supply kinematic XYZ covariance.'
                )
    neq = settings.raw['NEQConfig']['PathOfFilesConfig']
    runroot = Path(neq['NormalEqTemp'])/'JointChannel'/settings.tag/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    runroot.mkdir(parents=True)
    feedback, dm_root, base_state, cumulative = None,None,None,None
    if resume_seed is not None:
        resume_seed = Path(resume_seed).resolve()
        manifest, _ = load_result(resume_seed)
        if manifest['covariance'] or manifest['arcs'] != list(arcs):
            raise ValueError('resume requires initial WLS checkpoint with identical arcs')
        if not (resume_seed/'complete.json').is_file():
            raise ValueError('initial WLS checkpoint is incomplete')
        for arc in arcs:
            with h5py.File(resume_seed/('arc_%d.h5'%arc),'r') as h:
                for name in ('Orbit_A','Orbit_B','RangeRate'):
                    if name in h and 'linear_postfit' not in h[name]:
                        raise ValueError('missing checkpoint postfit residuals')
        dm_root, base_state = manifest['dm_root'], manifest['base_state']
        if not Path(dm_root).is_dir() or (base_state and not Path(base_state).is_file()):
            raise ValueError('missing checkpoint dynamical state')
        write_json(runroot/'resume.json',dict(source=str(resume_seed),stage='initial WLS complete'))
        print('Resume completed initial WLS: '+str(resume_seed),flush=True)
    elif resume_bootstrap is not None:
        bootstrap = Path(resume_bootstrap).resolve()
        checkpoint = json.loads((bootstrap/'complete.json').read_text())
        if checkpoint['arcs'] != list(arcs) or checkpoint['scale_scope'] != list(settings.span):
            raise ValueError('bootstrap arcs/date span do not match')
        dm_root, base_state = bootstrap/'DesignMatrix', bootstrap/'state.h5'
        if not Path(base_state).is_file():
            raise ValueError('bootstrap state missing')
        for arc in arcs:
            if not (dm_root/settings.tag/('Orbit_%d_AB.hdf5'%arc)).is_file():
                raise ValueError('bootstrap design matrix missing')
        write_json(runroot/'resume.json',dict(source=str(bootstrap),stage='bootstrap complete'))
        print('Reuse completed dynamics bootstrap: '+str(bootstrap),flush=True)
    elif monthly_scale:
        bootstrap = bootstrap_monthly_scale(settings, arcs, runroot/'state_00', options['FeedbackProcesses'])
        dm_root, base_state = bootstrap/'DesignMatrix', bootstrap/'state.h5'
    if options['FeedbackIterations'] is None:
        # Preserve legacy behavior for existing settings: each outer pass feeds back.
        outer_count = options['MaxOuterIterations'] if options['FeedbackEnabled'] else 1
    else:
        # An explicit feedback count adds one final solve after the last integration.
        outer_count = options['FeedbackIterations'] + 1
    records = []
    for outer in range(outer_count):
        # Keep the previous state as the linearization, not as final products.
        # A feedback product belongs only to the solve that generated it.
        feedback = None
        print('Joint outer solve %d/%d' % (outer + 1, outer_count), flush=True)
        # Use KIN-based weights only once. Later dynamical linearizations start
        # with the residual covariance and VCE factors from the previous outer loop.
        seed_covariance = options['InitialUseCovariance'] or (kin_seed and outer > 0)
        seed_name = ('outer_%02d_gls_seed' if seed_covariance and kin_seed else 'outer_%02d_ols') % outer
        if outer == 0 and resume_seed is not None:
            solution = resume_seed
        else:
            solution = run_joint(settings,arcs,runroot/seed_name,
                options['EstimateEmpirical'],seed_covariance,dm_root,base_state,rcond=options['Rcond'],
                workers=options['ArcProcesses'])
        # Freeze the initial WLS normalization for all subsequent GLS/outer stages.
        if outer == 0 and wls_basis(settings.raw['SolverConfig']):
            seed_solver = Settings(Path(solution)/'settings', Path(solution)).raw['SolverConfig']
            for key in ('InitialOrbitMeanPrecision', 'InitialWeightingAudit'):
                settings.raw['SolverConfig'][key] = seed_solver[key]
        _,delta = load_result(solution)
        if cumulative is None:
            cumulative = np.zeros_like(delta)
        base_global = cumulative.copy()
        export_residuals(settings,solution,base_global+delta,workers=options['ArcProcesses'])
        if options["SaveIterationProducts"]:
            from .iteration_products import export_iteration_gfc
            export_iteration_gfc(settings,solution,base_global+delta,options["ArcProcesses"])
        previous = None
        gls_converged=None
        gls_iterations=0
        if is_gls:
            gls_converged=False
            for iteration in range(max(1,int(gls.MaxIteration))):
                print('Estimate covariance from current postfit residuals: '+str(solution), flush=True)
                gls_iterations=iteration+1
                owner._gls_rng = np.random.default_rng(int(gls.RandomSeed)+iteration)
                weight_converged=True
                weight_diagnostic=None
                if arc_weight_mode=='VCE':
                    with h5py.File(Path(solution)/'global.h5','r') as prior:
                        have_statistics='redundancy_probes' in prior
                    if have_statistics:
                        new_weights,weight_diagnostic=update_variances(settings.raw['SolverConfig'],read_statistics(solution))
                        settings.raw['SolverConfig']['ArcWeightingConfig']=new_weights
                        weight_converged=weight_diagnostic['converged']
                    else:weight_converged=False
                if arc_weight_mode!='Off' or settings.raw['SolverConfig'].get('CovariancePoolingMode','Legacy')=='WithinArc':
                    noise=estimate_monthly_noise(settings,solution,gls,options['ArcProcesses'])
                    weight_converged=weight_converged and noise['converged']
                    write_json(runroot/('noise_%02d_%02d.json'%(outer,iteration)),noise)
                    if weight_diagnostic is not None:
                        write_json(runroot/('arc_weights_%02d_%02d.json'%(outer,iteration)),weight_diagnostic)
                elif gls.OrbitCovarianceMode == 'FixedDiagonal':
                    owner.GLS(orbit_fixed_diagonal=float(gls.OrbitFixedDiagonal))
                else:
                    owner.GLS_full()
                write_json(runroot/('covariance_%02d_%02d.json'%(outer,iteration)),
                    dict(residual_source=str(solution), mode=gls.OrbitCovarianceMode,
                         orbit_fixed_diagonal=float(gls.OrbitFixedDiagonal) if gls.OrbitCovarianceMode == 'FixedDiagonal' else None))
                print('Covariance ready; rebuilding GLS NEQ from all arcs', flush=True)
                solution = run_joint(settings,arcs,runroot/('outer_%02d_gls_%02d'%(outer,iteration)),
                    options['EstimateEmpirical'],True,dm_root,base_state,rcond=options['Rcond'],
                    workers=options['ArcProcesses'])
                _,delta = load_result(solution)
                export_residuals(settings,solution,base_global+delta,workers=options['ArcProcesses'])
                if options["SaveIterationProducts"]:
                    export_iteration_gfc(settings,solution,base_global+delta,options["ArcProcesses"])
                if previous is not None and iteration+1 >= int(gls.MinIteration):
                    change = np.linalg.norm(delta-previous)
                    size = max(np.linalg.norm(delta),np.linalg.norm(previous),np.finfo(float).eps)
                    if weight_converged and (change <= gls.AbsoluteTolerance or change/size <= gls.RelativeTolerance):
                        gls_converged=True
                        break
                previous = delta.copy()
        if options['SaveIterationProducts']:
            from .arc_weights import collect_noise_groups
            from .iteration_products import save_input_psds
            groups=collect_noise_groups(solution,settings.raw['SolverConfig'],options['ArcProcesses'])
            save_input_psds(groups,solution,settings.raw['SolverConfig'],purpose='final_postfit')
        do_feedback = options['FeedbackEnabled'] and (
            options['FeedbackIterations'] is None or outer < options['FeedbackIterations'])
        if do_feedback:
            if options['FeedbackIterations'] is not None:
                print('Dynamics feedback integration %d/%d' %
                      (outer + 1, options['FeedbackIterations']), flush=True)
            feedback = run_feedback(solution,runroot/('state_%02d'%(outer+1)),options['Damping'],
                                    workers=options['FeedbackProcesses'])
            dm_root,base_state = feedback/'DesignMatrix',feedback/'state.h5'
            cumulative = base_global+options['Damping']*delta
        else:
            cumulative = base_global+delta
        records.append(dict(iteration=outer+1,gls_iterations=gls_iterations,gls_converged=gls_converged,solution=str(solution),feedback=str(feedback) if feedback else None,
                            gravity_step_norm=float(np.linalg.norm(delta[18:] if settings.raw['GravityDesignParameterConfig']['AccelerometerConfig']['isScale'] else delta))))
    owner.cs = cumulative.copy()
    products = publish_products(settings,solution,cumulative,base_global,feedback,setting_directory,
                                arc_filter_path,arc_definition_path,workers=options['ArcProcesses'])
    if monthly_scale:
        baseline = np.asarray([settings.raw['DesignAccelerometerConfig_'+sat]['cali_value']['Scale']
                               for sat in ('A','B')],float)
        scale_path = runroot/'monthly_scale.json'
        write_json(scale_path,dict(scope=list(settings.span),order='A row-major9, B row-major9',
            baseline=baseline.tolist(),increment=cumulative[:18].reshape(2,9).tolist(),
            absolute=(baseline+cumulative[:18].reshape(2,9)).tolist()))
        products['MonthlyScale'] = str(scale_path)
    write_json(runroot/'complete.json',dict(products=products,iterations=records,
        feedback_integrations=sum(record['feedback'] is not None for record in records),
        final_solution=str(solution), final_solve_after_feedback=bool(
            options['FeedbackIterations'] is not None and options['FeedbackIterations'] > 0)))
    print('Joint estimation products:',products)
    return products
