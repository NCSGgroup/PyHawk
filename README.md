# PyHawk: An efficient gravity recovery solver for low-low satellite-to-satellite tracking gravity missions

# 1. Description
The low-low satellite-to-satellite tracking (ll-sst) gravity missions,such as the Gravity Recovery and Climate Experiment (GRACE) and its Follow-On (GRACE-FO), provide an important space-based Essential Climate Variable (ECV) over the last two decades. The measurement systems of these ll-sst missions are highly complex, thus, a data processing chain is required to exploit the potential of their high-precision measurements, which challenges both the general and expert users. In this study, we present an open-source, user-friendly, cross-platform and integrated toolbox "PyHawk", which is the first Python-based software in relevant field, to address the complete data processing chain of ll-sst missions including GRACE, GRACE-FO and likely the future gravity missions. This toolbox provides non-expert users an easy access to the payload data pre-processing, background force modeling, orbit integration, ranging calibration, as well as the ability for temporal gravity field recovery using ll-sst measurements.
In addition, a series of high-standard benchmark tests have been provided
to evaluate PyHawk, confirming its performance to be comparable with
those are being used for providing the official Level-2 time-variable
gravity field solutions of GRACE and GRACE-FO. Researchers working with
the low-Earth-orbit space geodetic techniques, GNSS based orbit
determination, and gravity field modeling can benefit from this toolbox.

<div align=center>
    <img src="module.png" width="450" >
</div>

# 2. Contact

Yi Wu (wu_yi@hust.edu.cn) , Fan Yang (fany@plan.aau.dk) , ShuHao Liu (liushuhao@hust.edu.cn)

Huazhong University of Science and Technology
Geodesy Group (see https://aaugeodesy.com/), Department of Sustainability and Planning, Aalborg University, Aalborg 9000, Denmark

This work is supported by the Huazhong University of Science and Technology. Additional supports come from and Aalborg University.


# 3. Features
- It adopts advanced data exchange mechanisms, enabling users to interact with the software through configuration files, which eliminates the need to modify source code and reduces the risk or complexity associated with code changes.
- The software has been optimized to achieve computational efficiency comparable to that of Fortran and C++, languages commonly used in existing GRACE (-FO) toolboxes.
- PyHawk is compatible with multiple platforms and allows for easy and fast installation on systems such as Windows, Linux or clusters.
- PyHawk has a modular structure, where code is highly decoupled internally for an easy use, comprehension and extension.

# 4. Installation

The project can be firstly downloaded from GitHub (\url{https://github.com/NCSGgroup/PyHawk.git}). Then, a Python environment must be created, where two options are offered: manual installation and automatic installation. The latter option is generally quicker and thus is recommended if they have already installed Conda.

For a manual installation:
1. First, make sure you have at least Python version 3.8 installed on your system: 
    `python --version`
2. Then, make sure to set up pip, the Python package installer, navigate to the main directory (where PyHawk is located) and install the tool using the following command.
    `python -m pip install pip`
3. Now, you can install the remaining extension libraries that the project depends on in sequence:
    - `pip install numpy`
    - `pip install tqdm`
    - `pip install scipy`
    - `pip install h5py`
    - `pip install numba`
    - `pip install Quaternion`
    - `pip install jplephem`
    - `pip install matplotlib`

For quicker setup, assuming that the Conda (\url{https://www.anaconda.com/download}) is already configured, one can follow

1. `conda create -n py-hawk python=3.8.10`
2. `source activate py-hawk` 
3. `pip install -r requirments.txt` 

**Troubleshooting:** This toolbox has a dependence on two dynamic libraries as addressed in previous section. For this, we have provided the pre-compiled dynamic libraries (created individually for Linux and Windows) to simplify the installation. However, these pre-built libraries may occasionally fail because of incompatibility between the libraries and running platform. If this is the case, we suggest you to use the provided C (C++) source code together with the compiling script (PyHawk/lib) to automatically compile a library compatible with your own platform. This can be easily implemented if one runs PyHawk at Linux platform, however, for Windows, you may need an extra software (Visual C++) to generate the library. Please also feel free to reach out to us about this issue. 

# 5. Sample data
The operation of the PyHawk software requires essential data such as GRACE (-FO) Level-1b data, de-aliasing product, and Earth Orientation Parameter (EOP), as listed below. For these, we provide sample data for running the demos provided in next section, which mainly includes (but not limited to):
- **GRACE-(FO) Level-1b data:** A few months data for demo. Suggested default path: /PyHawk/data.
- **De-aliasing product:** A few months data. Suggested default path: /PyHawk/data/AOD/RL06.
- **EOP:** Suggested default path: /PyHawk/data/eop.
- **Load Love Number:** Suggested default path: /PyHawk/data/LoveNumber.
- **Static Gravity:** Suggested default path: /PyHawk/data/StaticGravityField.
- **Ephemerids:** Suggested default path: /PyHawk/data/Ephemerides.
- **Auxiliary:** Suggested default path: /PyHawk/data/Auxiliary.
- **Benchmark:** Suggested default path: /PyHawk/data/Benchmark.

Be aware that these sample data are of large size, so that we put them at an open data repository (\url{https://doi.org/10.5281/zenodo.14196917}) for use when running the demos.

# 6. Quick Start
Here we present three demo (demo1, demo2, demo3), which are available under /PyHawk/demo, to showcase the usage of PyHawk. For the sake of generality, we design three specific demos to cover the major interests of the users: (1) demo-1 for orbit determination (2) demo-2 for background force modeling and (3) for the gravity field modeling. Be aware that, since PyHawk is a comprehensive toolbox for various research objectives, it is unlikely to be exhaustive to list all operations here. Nevertheless, one can see the detailed description at the comments of each demo in its scripts.

To be able to run the demo, please be aware that sample data as addressed in previous section are required. And for a quick start, we suggest to put all these sample data at default path: /PyHawk/data/. Nevertheless, as an advanced user, one can place data at any desired place as long as the setting files are well configured. Here is a brief tutorial of three demos.

## 6.1 Demo-1
This demo is the basis of orbit determination. Be aware that the desired platform requires at least 4 CPU cores for parallel computation, with an estimated completion time of 7 minutes.

2. Then, navigate to the demo file path and execute the command: `python demo_1.py`

## 6.2 Demo-2
This demo is the basis of orbit determination. Be aware that the desired platform requires at least 4 CPU cores for parallel computation, with an estimated completion time of 7 minutes.

2. TThen, navigate to the demo file path and execute the command: `python demo_2.py`

## 6.3 Demo-3
This demo is the basis of orbit determination. Be aware that the desired platform requires at least 4 CPU cores for parallel computation, with an estimated completion time of 7 minutes.

2. Then, navigate to the demo file path and execute the command: `python demo_3.py`

# 7. License

Please feel free to contact us for possible more examples.