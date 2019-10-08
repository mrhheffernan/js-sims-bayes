#!/usr/bin/bash

for i in {0..99}; do
  {
  mkdir closure_parallel/$i
  cd closure_parallel/$i

  cp -r ../../src .
  cp -r ../../HIC_experimental_data .
  cp -r ../../closure_truth_dob .
  cp -r ../../design_pts .
  cp -r ../../model_calculations .
  cp -r ../../emulator .
  cp -r ../../mcmc .

  sed -i .bak "s/validation_pt=.*/validation_pt=$i/g" src/configurations.py

  #remove existing mcmc chains
  rm mcmc/chain.hdf

  #perform MCMC
  ./src/bayes_mcmc.py 4000 --nwalkers 100 --nburnsteps 1000

  #generate plots
  #./src/bayes_plot.py plots/diag_posterior.png
  #cp plots/diag_posterior.png closure_plots/diag_posterior_$i.png

  #generate file of truth and credibility
  ./src/emulator_load_and_validate.py
  } &

done

#wait til all processes have finished
wait
echo "Finished closure test! Goodbye"
