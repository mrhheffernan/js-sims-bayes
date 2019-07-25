#!/usr/bin/bash

for i in {0..99}; do
    sed -i "s/validation=.*/validation=$i/g" src/configurations.py
    rm -f mcmc/*
    ./src/bayes_mcmc.py 2000 --nwalkers 100 --nburnsteps 500 || exit
    ./src/CR.py || exit
done
