#!/usr/bin/bash

declare -a bgpids

cleanup() {
    for pid in ${bgpids[@]}; do
        kill -9 $pid
    done
}

trap "cleanup" SIGINT SIGTERM

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

  #sed -i .bak "s/validation_pt=.*/validation_pt=$i/g" src/configurations.py
  #perform MCMC
  ./src/bayes_mcmc.py 3000 --nwalkers 70 --nburnsteps 1000

  #generate plots
  ./src/bayes_plot.py plots/diag_posterior.png

  #generate file of truth and credibility
  ./src/emulator_load_and_validate.py
  cd ..
  } &
  bgpids+=("$!")
done

#wait til all processes have finished
wait
echo "Finished closure test! Goodbye"
