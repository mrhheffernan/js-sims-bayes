## Configurations

First see the parameters located in `src/configurations.py`. These set the number of design points, principal components, etc...

## Generating Design

 To generate the design points: `./src/design.py design_pts`

This will create a directory called `design_pts`. 



To make a plot checking the design prior for all observables:

```./src/bayes_plot.py plots/obs_prior.png```

To generate a plot of the parameter values of all design points:

``` ./src/bayes_plot.py plots/param_prior.png```

## Running Events

Now copy `design_pts` to `sims_scripts/input_config/design_pts` on stampede2. 

Then look the script `submit_launcher_design_new_norm`, and change the parameters controlling number of design points and number of events per design point appropriately.

Submit the job : `sbatch submit_launcher_design_new_norm`

## Event Averaging

Once events have finished, cp the results `<job_ID>` folder with name given by job ID to the `model_calculations` directory (on stampede2 or locally)

Use `sh prepare_stampede2.sh` to load modules necessary if performing event averaging on stampede2.

The scripts `cat_events_for_each_design_pt.sh` and `average_events_for_each_design_pt.sh` can be used to generate the `obs.dat` file for each design point. Then be careful to use brace expansion to cat these files together so that their indexing is preserved: `cat <job_ID>/{0..9}/obs.dat >> obs.dat`

## Building Emulator

`obs.dat` generated above should be copied to `model_calculations/obs.dat`



Then, to build the emulator: 

```./src/emulator.py --retrain --npc 10 --nrestarts 4```

This will build the emulator and store it as a dill file.



## Validating Emulator

To validate the emulator using a validation data set:

```./src/emulator_load_and_validate.py```



## MCMC

To run the MCMC for parameter estimation:

```./src/bayes_mcmc.py 2000 --nwalkers 100 --nburnsteps 500```



To plot the estimation of parameters:

```./src/bayes_plot.py plots/diag_posterior.png```

To plot the entire posterior of all parameters:

```./src/bayes_plot.py plots/posterior.png```

To plot the posterior of viscosities:

```./src/bayes_plot.py plots/viscous_posterior.png```

To plot the emulator prediction using best fit parameters against data (or pseudodata):

```./src/bayes_plot.py plots/obs_validation```

