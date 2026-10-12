# PyHawk

**A Python framework for gravity-field recovery from low–low satellite-to-satellite tracking (ll-SST).**

PyHawk processes GRACE and GRACE-FO observations to estimate time-variable gravity fields and generate along-orbit products. It combines configurable force models, numerical orbit and variational integration, accelerometer calibration, and arc-parallel estimation.

<p align="center"><img src="module.png" width="450" alt="PyHawk processing modules"></p>

## Processing workflow

```text
L1B + kinematic orbits + background models
  → preprocessing and arc selection
  → orbit fitting and KBRR calibration
  → orbit/variational integration and design matrices
  → weighted gravity-field estimation and GLS
  → optional dynamical feedback and updated joint solution
  → spherical harmonics and standard along-orbit products
```

The optional joint channel estimates gravity coefficients, shared monthly accelerometer scale parameters, arc-local initial states/ACC parameters, and KBRR empirical terms in one system. Observation blocks share the same physical parameters before local elimination and back-substitution. GLS updates the stochastic model; dynamical feedback updates physical parameters and reintegrates the orbit and variational equations. These are separate operations.

## Installation

Clone the `master` branch, which contains the current solver:

```bash
git clone --branch master https://github.com/NCSGgroup/PyHawk.git
cd PyHawk
conda create -n pyhawk python=3.8.10
conda activate pyhawk
python -m pip install -r requirments.txt
```

`requirments.txt` is the existing filename. Python 3.8 is the legacy runtime used for the current processing workflow; the dependency list is unpinned, so retain an environment snapshot for reproducible runs. Some optional modules require additional packages, such as `netCDF4` for GOT5.6.

Native libraries are loaded from [`lib/`](lib): `sofa.dll` / `GravHar.dll` on Windows, or `sofa.so` / `GravHar.so` on Linux. They must match the host architecture and runtime. The current loaders explicitly support Windows and Linux; macOS requires adapting the library loading/build setup.

## Data preparation

Observation and model datasets must be obtained separately. Configure their locations before running:

| Input | Typical location under the project | Notes |
|---|---|---|
| GRACE L1B | `data/GRACE_RL02/YYYY/YYYY-MM`, `data/GRACE_RL03/grace_1B_YYYY-MM_03` | RL03 KBR/SCA and RL02 ACC/GNV for the mixed-release reader |
| GRACE-FO L1B | `data/GRACE_FO_RL04` | Use the directory hierarchy and acceleration product expected by the selected reader |
| Kinematic orbit | `data/KinematicOrbit/YYYY/YYYY-MM` | Include epoch covariances when covariance weighting is enabled |
| AOD | `data/AOD/RL07` | Cover the requested interval and interpolation boundaries |
| Background models | Paths in `ForceModelConfig.json` and `FrameConfig.json` | EOP, ephemerides, reference gravity, tides and auxiliary coefficients |

Check both satellites, valid dates, units and reference frames. Missing data must be resolved or explicitly represented in `date_delete`; changing a path alone does not convert a data format.

## Configuration and execution

Start from [`setting/Calibrate/`](setting/Calibrate), copy it to a separate directory for each experiment, and review:

| Configuration file | Controls |
|---|---|
| `InterfaceConfig.json` | Mission, `date_span`, `date_delete`, sampling, arcs and observation paths |
| `CalibrateOrbitConfig.json` | Processing stages, process counts, GLS, joint estimation, feedback and LGD |
| `AdjustOrbitConfig.json` | Orbit observation source, fitting and KBRR quality control |
| `AccelerometerConfig_A/B.json` | ACC calibration parameters and initial values |
| `DesignAccelerometerConfig_A/B.json`, `DesignParameterConfig.json` | Parameter layout for design integration; keep consistent with the chosen estimation scheme |
| `ForceModelConfig.json`, `FrameConfig.json`, `ODEConfig.json` | Force models, frame/time conversion and integration |
| `NEQConfig.json`, `SolverConfig.json` | Observations used in NEQ, weighting, covariance and solver outputs |

Run from the project root after preparing the data and settings:

```bash
python demo/demo_CalibrateOrbit.py --setting-dir setting/MyExperiment
```

The entry point changes the working directory to `demo/`: JSON paths such as `../data` and `../temp` are relative to **`demo/`, not the setting directory**. Keep the default output roots unless a validated workflow requires otherwise.

The supplied `Calibrate` settings are a processing example, **not a fresh-run preset**: some stages are disabled and require existing intermediate files. For a new run, enable the necessary `StepControlConfig` stages. Reuse a disabled stage only when its inputs, configuration and cached results match. Monthly-scale bootstrap generates its own design matrices; avoid an unnecessary preliminary design-integration pass for that workflow.

### Estimation options

- Orbit fitting and gravity estimation can use different orbit observations: `OrbitConfig.ObservationSource` selects the fitting source; `NEQConfig` selects the NEQ observations.
- In the joint channel, `InitialOrbitWeighting="KINRelative"` uses normalized KIN epoch precision for the initial WLS, with `OrbitKinFactor` setting relative KBRR precision. KBRR remains in **m/s**.
- `GLSResidualSpace="Physical"` with `OrbitCovarianceMode="EstimatedFull"` estimates subsequent covariance from physical post-fit residuals. Arc weighting is configured separately through `ArcWeightingConfig`; it is not enabled by selecting GLS alone.
- Set `JointEstimationConfig.Enabled=true` to use the joint channel. Monthly scale estimation also requires `MonthlyScaleEnabled=true` and compatible ACC parameter layouts.
- To apply one feedback integration **and solve again using the updated dynamics**, set `FeedbackEnabled=true` and `FeedbackIterations=1`. `GLSIterationConfig.MaxIteration` independently limits GLS iterations.
- KBRR calibration rejection, epoch/arc screening and physical O–C screening have separate switches. Disabling them does not remove basic validity checks or orbit quality control. `PhysicalOCOutlierEnabled=true` requires positive `RangeRateConfig.Iterations`.

These options describe supported modes, not a universal parameter recommendation. The full configuration and available defaults are defined in [`src/Preference/`](src/Preference) and [`src/Solver/JointChannel/`](src/Solver/JointChannel).

## Parallel runs and restart

`ParallelControlConfig` sets worker counts for arc processing. Match them to allocated CPUs and memory. For multi-process jobs, set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` to `1` to avoid nested BLAS oversubscription. Under Slurm, run the same Python command inside a site-specific batch script. Interleaved worker progress bars are not an overall monthly completion percentage.

Joint runs store manifests, settings and stage completion records under `temp/NEQ/JointChannel/`. Restart entry points are in [`demo/`](demo), including `resume_JointChannel.py` and `demo_ResumeBootstrap.py`; inspect `--help` and select the entry point matching the completed stage. A checkpoint is reusable only with compatible configuration and valid files. Do not modify shared code during active runs or delete active run directories.

## Outputs

Default paths below are relative to the project root. Products are generated according to the enabled stages:

| Path | Product |
|---|---|
| `result/SH/` | Spherical-harmonic gravity coefficients (`.gfc`) |
| `temp/StandardKBRR/` | Standard KBRR products |
| `temp/StandardLGD/` | Line-of-sight gravity difference products |
| `temp/StandardNEQ/` | Standard normal-equation products |
| `temp/StandardACC/` | Accelerometer calibration products |
| `report/`, `log/` | Preprocessing, arc selection and runtime diagnostics |
| Other `temp/` subdirectories | RawData, fitted states, design matrices and solver intermediates |

Archive the configuration and completion records with the products. Before cleaning intermediate files, verify the final products are readable, finite and cover the intended dates; a Slurm exit status alone is insufficient. **Do not delete the whole `temp/` directory: it also contains standard products.**

## Code and validation

[`src/`](src) contains the implementation, [`demo/`](demo) the entry points, [`benchmark/`](benchmark) force-model benchmarks, and [`test/`](test) regression tests. A focused joint-solver check is:

```bash
python -m unittest discover -s test -p test_joint_channel.py -v
```

The small synthetic tests validate solver behavior; they do not replace full-month scientific validation with mission data.

## Contact

Yi Wu — wu_yi@hust.edu.cn · Fan Yang — fany@plan.aau.dk · ShuHao Liu — liushuhao@hust.edu.cn

Huazhong University of Science and Technology; [Geodesy Group, Aalborg University](https://aaugeodesy.com/).
