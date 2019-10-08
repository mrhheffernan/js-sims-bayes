#!/usr/bin/env python3
#import logging
from configurations import *
import numpy as np


#get model calculations at VALIDATION POINTS
if validation:
    print("Load calculations from " + f_obs_validation)
    Y_exp_data_pre = np.fromfile(f_obs_validation, dtype=bayes_dtype)
    Y_exp_data=np.array([Y_exp_data_pre[validation_pt]])

#get experimental data
else :
    print("Loading experimental data from " + dir_obs_exp)
    entry = np.zeros(1, dtype=np.dtype(bayes_dtype) )
    #entry = np.zeros(1, dtype=np.dtype(calibration_bayes_dtype) )

    for system_str in system_strs:
        #print("system : " + system_str)
        expt = expt_for_system[system_str]
        path_to_data = 'HIC_experimental_data/' + system_str + '/' + expt_for_system[system_str] + '/'
        path_to_PHENIX = 'HIC_experimental_data/' + system_str + '/PHENIX/'
        for obs in list( obs_cent_list[system_str].keys() ):
            #print("Observable : " + obs)
        #for obs in list( calibration_obs_cent_list[system_str].keys() ):
            n_bins_bayes = len(obs_cent_list[system_str][obs]) # only using these bins for calibration. Files may contain more bins
            for idf in range(number_of_models_per_run):

                #for STAR identified yields we have the positively charged particles only, not the sum of pos. + neg.
                if (obs in STAR_id_yields.keys() and system_str == 'Au-Au-200'):

                    #for proton dN/dy use the PHENIX data rather than star,
                    #for all other observables use STAR
                    if (obs == 'dN_dy_proton'):
                        expt_data = pd.read_csv(path_to_PHENIX + obs + '_+.dat', sep = ' ', skiprows=2, escapechar='#')
                    else :
                        expt_data = pd.read_csv(path_to_data + obs + '_+.dat', sep = ' ', skiprows=2, escapechar='#')

                    #our model takes the sum of pi^+ and pi^-, k^+ and k^-, etc...
                    #the Au Au data are saved separately for particles and antiparticles
                    entry[system_str][obs]['mean'][:, idf] = expt_data['val'].iloc[:n_bins_bayes] * 2.0
                else :
                    expt_data = pd.read_csv(path_to_data + obs + '.dat', sep = ' ', skiprows=2, escapechar='#')
                    entry[system_str][obs]['mean'][:, idf] = expt_data['val'].iloc[:n_bins_bayes]

                try :
                    err_expt = expt_data['err'].iloc[:n_bins_bayes]
                except KeyError :
                    stat = expt_data['stat_err'].iloc[:n_bins_bayes]
                    sys = expt_data['sys_err'].iloc[:n_bins_bayes]
                    err_expt = np.sqrt(stat**2 + sys**2)

                if (obs in STAR_id_yields.keys() and system_str == 'Au-Au-200'):
                    err_expt *= np.sqrt(2.0)

                entry[system_str][obs]['err'][:, idf] = err_expt


                #print("Mean : " + str(entry[system_str][obs]['mean'][:, idf]) )
                #print("Error : " + str(entry[system_str][obs]['err'][:, idf]) )

    Y_exp_data = entry
