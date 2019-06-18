## Usage

All relevant source/scripts are located in the directories `src`, `model_calculations` or `expt`

## Configurations

First see the parameters located in `src/configurations.py`. These set the number of design points, principal components, etc...

## Generating Design

 To generate the design points: `python3 src/design.py design_pts`

This will create a directory called `design_pts`. 

## Running Events

Now copy `design_pts` to `sims_scripts/input_config/design_pts` on stampede2. 

Then look the script `submit_launcher_design_new_norm`, and change the parameters controlling number of design points and number of events per design point appropriately.

Submit the job : `sbatch submit_launcher_design_new_norm`

## Event Averaging

Once events have finished, cp the results `<job_ID>` folder with name given by job ID to the `model_calculations` directory (on stampede2 or locally)

Use `sh prepare_stampede2.sh` to load modules necessary if performing event averaging on stampede2.

The scripts `cat_events_for_each_design_pt.sh` and `average_events_for_each_design_pt.sh` can be used to generate the `obs.dat` file for each design point. Then be careful to use brace expansion to cat these files together so that their indexing is preserved: `cat <job_ID>/{0..9}/obs.dat >> obs.dat`

## Building/Validating Emulator

`obs.dat` generated above should be copied to `model_calculations/obs.dat`

Then, to build the emulator use `python3 src/emulator.py`

This will also dill the emulator and save it in `emulators` directory.

To load the emulator from file and make some plots, use `python3 src/emulator_load_and_validate.py`

