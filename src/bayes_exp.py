#!/usr/bin/env python3
import logging
from configurations import *
import numpy as np

#get model calculations at VALIDATION POINTS
#print("Load calculations from " + f_obs_validation)
#Yexp_PseudoData = np.fromfile(f_obs_validation, dtype=bayes_dtype)

#get experimental data from file
print("Loading experimental data from " + dir_obs_exp)

for system in system_strs:
    print("System : " + system)
    for obs in list( obs_cent_list[system].keys() ):
        print("Obs : " + obs)
        expt_data = pd.read_csv('expt_data/saved_data/' + system + '/' + obs + '.dat', sep = ' ')
        x_expt = expt_data.iloc[:,0].values
        y_expt = expt_data.iloc[:,1].values
