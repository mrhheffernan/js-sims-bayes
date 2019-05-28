#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, glob
import re
# Output data format
from file_format_event_average import *


#################################################################################
#### Try to figure out semi-automatically what observables to group together ####
#################################################################################

# This is the input:
# Specifies how observables are grouped according to these regular expression
regex_obs_to_group_list=[
(r'$\pi$/K/p dN/dy',"dN_dy_(pion|kaon|proton)"),
(r'$\pi$/K/p $\langle p_T \rangle$',"mean_pT_(pion|kaon|proton)"),
(r'$\Lambda/\Omega/\Xi$ dN/dy',"dN_dy_(Lambda|Omega|Xi)"),  
(r'$v_n\{2\}$',"v[2-5+]2") 
]

# This parts figures out how to group observables based on the regular expressions

obs_to_group={}
# Loop over observables to see which ones to group
for obs_name in obs_cent_list.keys():
    found_match=False
    for regex_id, (regex_label, regex_obs_to_group) in enumerate(regex_obs_to_group_list):
        r = re.compile(regex_obs_to_group)
        match=r.match(obs_name)
        # No match means nothing to group
        if (match is not None):
            if (found_match):
                print("Non-exclusive grouping. Can't work...")
                exit(1)
            else:
                found_match=True

                obs_to_group[obs_name]=(regex_id, regex_label)

    if (not found_match):
        obs_to_group[obs_name]=None

# Parse the previous list to make something useful out of it
final_obs_grouping = {}

#
for n, (key, value) in enumerate(obs_to_group.items()):

    if (value is None):
        newvalue=(n,key)
    else:
        newvalue=value

    final_obs_grouping.setdefault(newvalue, []).append(key)




##############
#### Plot ####
##############

def plot(calcs):

    #Loop over delta-f
    #for idf in range(0,5):

    #Loop over observables
    #for n, (obs, cent) in enumerate(obs_cent_list.items()):
    for n, ((regex_id, obs_name), obs_list) in enumerate(final_obs_grouping.items()):

        plt.subplot(nb_of_rows,nb_of_cols,n+1)
        plt.xlabel(r'Centrality (%)', fontsize=7)
        plt.ylabel(obs_name, fontsize=7)
        
        for obs in obs_list:

            cent=obs_cent_list[obs]
            mid_centrality=[(low+up)/2. for low,up in cent]
            mean_values=calcs['Pb-Pb-2760'][obs]['mean'][:,0][0]
            stat_uncert=calcs['Pb-Pb-2760'][obs]['stat_err'][:,0][0]


            plt.errorbar(mid_centrality, mean_values, yerr=stat_uncert)

    plt.tight_layout(True)
    #plt.savefig("obs.pdf")
    plt.show()



if __name__ == '__main__':
        results = []
        for file in glob.glob(sys.argv[1]):
                # Load calculations       
                calcs = np.fromfile(file, dtype=np.dtype(bayes_dtype))

                # Count how many observables to plot
                nb_obs=len(final_obs_grouping)
                # Decide how many columns we want the plot to have
                nb_of_cols=4
                # COunt how many rows needed
                nb_of_rows=int(np.ceil(nb_obs/nb_of_cols))
                # Prepare figure
                fig = plt.figure(figsize=(2*nb_of_cols,2*nb_of_rows))

                entry = plot(calcs)
