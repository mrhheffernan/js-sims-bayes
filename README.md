# Bayesian parameter estimation for relativistic heavy-ion collisions

Workflow for a full Bayesian analysis in relativistic heavy-ion collisions: soft sector based on Jetscape
- The physical model simulation is based on [JETSCAPE framework](https://arxiv.org/abs/1903.07706). Regarding the `Installation`, `configuration`, running `JETSCAPE` and `analyze`, please check the official repository.
- This repository is devoted to **Bayesian analysis** of soft observables in heavy-ion collisions, where most of the statistical analysis work is inherited from [hic-param-est](http://qcd.phy.duke.edu/hic-param-est/). 
- There are four main stages to finish the analysis.
    - [1. **design points** preparation](#1-design-points-preparation)
    - [2. **model calculation** at each design point](#2-model-calculation-at-each-design-point)
    - [3. raw observables calculation, preparing the **training data**](#3-training-data-preparation)
    - [4. **calibration** to experimental measurements](#4-calibration-to-experimental-measurements)


## dependencies
preprequisites:
-  R: necessary to use R-lhc package to generate design points
- python3: `numpy, scipy, matplotlib, hdf5, pandas, seaborn, scikit-learn` ==> [`conda`](https://docs.conda.io/en/latest/miniconda.html) virtual environment recommended
- [`emcee`](https://emcee.readthedocs.io/en/stable/): the python ensemble sampling toolkit for affine-invariant MCMC  

## 1. design points preparation
related files: 
- *src/design.py*: the source code to generate the parameter points
- *src/design_write_module_inputs.py*: where the default values of parameters are assigned
- *src/configurations.py*: where some commonly changed configuration are configured

```
# to generate design points using Latin-hypercube method 
# and stored in param_points folder (X_train)
python src/design.py  param_points
```

commonly modified parameters:
- *src/configurations.py*
    - `n_design_pts_main`, `n_design_pts_validation`: where number of design points are assigned
    - `systems`: where the collision system is assigned

- *src/design.py*
    - one can choose different parameters that you would like to vary and set the range for the parameters
    - `seed`: one can fix/change the seed if you want to generate same/different design points


## 2. model calculation at each design point
> all the calculation is done through [stampede2](https://portal.tacc.utexas.edu/user-guides/stampede2)

- to be finished


## 3. training data preparation
related files:
- *src/calculations_average_obs.py*
- *src/configurations.py*: where the directory of result files, the filename of output files, the kinetic cut for observables are set

```
# to process the event-by-event information for each design points 
# and generate training data (Y_train)
python src/calculations_average_obs.py 
```
commonly modified/assigned arguments:
- *src/configurations.py*
    - `number_of_event_per_design`: number of minimum bias events for a single design points
    - `f_events_main`: the folder where the results of ebe information are stored 
    - `f_obs_main`: the filename of Y_train


## 4. calibration to experimental measurements


