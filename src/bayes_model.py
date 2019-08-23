#!/usr/bin/env python3

from configurations import *
import logging
import numpy as np

bayes_dtype=[    (s, 
                  [(obs, [("mean",float_t,len(cent_list)),
                          ("err",float_t,len(cent_list))]) \
                    for obs, cent_list in obs_cent_list[s].items() ],
                  number_of_models_per_run                                                                     
                 ) \
                 for s in system_strs
            ]



logging.info("Loading model calculations from " + f_obs_main)
model_data = np.fromfile(f_obs_main, dtype=bayes_dtype)


# things to drop
delete_sets = set()
for pt in range(n_design_pts_main): # loop over all design points
    system_str = 'Pb-Pb-2760'
    for obs in active_obs_list[system_str]:
        values = np.array( model_data[system_str][pt, idf][obs]['mean'] )
        # delete Nan dataset
        isnan = np.isnan(values)
        if np.sum(isnan)>0:
            model_data[system_str][pt, idf][obs]['mean'][isnan] = \
                np.mean(values[np.logical_not(isnan)])

delete_sets = [
106,
109,
114,
119,
134,
144,
152,
165,
167,
170,
179,
190,
196,
28,
35,
72,
82,
87,
90,
67,
148,
176,
]

print(np.sort(list(delete_sets)))

trimed_model_data = np.delete(model_data, list(delete_sets), 0)

