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
- `src/design.py`: the source code to generate the parameter points
- `src/design_write_module_inputs.py`: where the default values of parameters are assigned
- `src/configurations.py`: where some commonly changed configuration are configured

```
# to generate design points using Latin-hypercube method
# and stored in param_points folder
python src/design.py  param_points
```


## 2. model calculation at each design point
> all the calculation is done through [stampede2](https://portal.tacc.utexas.edu/user-guides/stampede2)

- to be finished


## 3. training data preparation

## 4. calibration to experimental measurements

