#!/usr/bin/env python3
import logging
from configurations import *
import numpy as np

#get model calculations at VALIDATION POINTS
if validation:
    print("Load calculations from " + f_obs_validation)
    Y_exp_data = np.fromfile(f_obs_validation, dtype=bayes_dtype)

#get experimental data
else :
    print("Loading experimental data from " + dir_obs_exp)
    entry = np.zeros(1, dtype=np.dtype(bayes_dtype))
    #Y_exp_data = []

    for system in system_strs:
        for obs in list( obs_cent_list[system].keys() ):
            for idf in range(number_of_models_per_run):
                expt_data = pd.read_csv('expt_data/saved_data/' + system + '/' + obs + '.dat', sep = ' ', header=None)
                entry[system][obs]['mean'][:, idf] = expt_data.iloc[:,1].values
                entry[system][obs]['err'][:, idf] = expt_data.iloc[:,2].values
                entry[system][obs]['err'][:, idf] = 0.0

                #entry[system][obs]['mean'][idf, :] = expt_data.iloc[:,1].values
                #entry[system][obs]['err'][:, idf] = expt_data.iloc[:,2].values
                #entry[system][obs]['err'][idf, :] = 0.0

    #Y_exp_data.append(entry)
    Y_exp_data = entry
