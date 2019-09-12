#!/usr/bin/env python3

from configurations import *
#import logging
import numpy as np

print("Loading model calculations from " + f_obs_main)
model_data = np.fromfile(f_obs_main, dtype=bayes_dtype)
print("model_data.shape = " + str(model_data.shape))


if transform_multiplicities:
    print("Note : multiplicity related observables o = dN_dx or o = dET_dx  \n")
    print("will be transformed via o -> ln(1 + o) before taining emulator!")

# handle some Nans
for pt in range(n_design_pts_main): # loop over all design points
    system_str = system_strs[0]
    for obs in active_obs_list[system_str]:
        values = np.array( model_data[system_str][pt, idf][obs]['mean'])
        # delete Nan dataset
        isnan = np.isnan(values)
        if np.sum(isnan) > 0:
            print("WARNING : FOUND NAN IN MODEL DATA : (design pt , obs) = ( {:s} , {:s} )".format( str(pt), obs) )
            model_data[system_str][pt, idf][obs]['mean'][isnan] = np.mean(values[np.logical_not(isnan)])

        #transforming yield related observables
        is_mult = ('dN' in obs) or ('dET' in obs)
        if is_mult and transform_multiplicities:
            model_data[system_str][pt, idf][obs]['mean'] = np.log(1.0 + values)



if len(delete_design_pts_set) > 0 :
    print("Design points which will be deleted from training : " + str( np.sort( list(delete_design_pts_set) ) ) )
    trimmed_model_data = np.delete(model_data, list(delete_design_pts_set), 0)
else :
    print("No design points will be deleted from training")
    trimmed_model_data = model_data

#load the validation model calculations
if validation:
    if pseudovalidation:
        if crossvalidation:
            validation_data = model_data #don't omit testing points from validation set
        else :
            print("Using training model data set trimmed_model_data for pseudovalidation ")
            validation_data = trimmed_model_data

    else :
        print("Loading validation calculations from " + f_obs_validation)
        validation_data = np.fromfile(f_obs_validation, dtype=bayes_dtype)
        print("validation_data.shape = " + str(validation_data.shape))
