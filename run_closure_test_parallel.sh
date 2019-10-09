#!/usr/bin/bash

cd closure_parallel

for i in {0..99}; do
  {
  mkdir $i
  cd $i
  pwd

  cp -r ../../src .
  cp -r ../../HIC_experimental_data .
  cp -r ../../design_pts .
  cp -r ../../model_calculations .
  cp -r ../../emulator .

  mkdir mcmc
  mkdir closure_plots
  mkdir plots
  mkdir closure_truth_dob

  #change the validation point
  sed -i .bak "s/validation_pt=.*/validation_pt=$i/g" src/configurations.py
  #perform MCMC
  ./src/bayes_mcmc.py 2000 --nwalkers 50 --nburnsteps 1000

  #generate plots
  ./src/bayes_plot.py plots/diag_posterior.png

  #generate file of truth and credibility
  ./src/emulator_load_and_validate.py
  cd ..
  } &
done

#wait til all processes have finished
wait
echo "Finished closure test! Goodbye"
