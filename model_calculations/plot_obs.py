#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os, glob
import re
# Output data format
from file_format_event_average import *

# Observables are grouped in plots according to this regular expression
regex_obs_to_group_list=[
(r'$\pi$/K/p dN/dy',"dN_dy_(pion|kaon|proton)"),
(r'$\pi$/K/p $\langle p_T \rangle$',"mean_pT_(pion|kaon|proton)"),
(r'$\Lambda/\Omega/\Xi$ dN/dy',"dN_dy_(Lambda|Omega|Xi)"),  
(r'$v_n\{2\}$',"v[2-5+]2") 
]

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

## Plot ID
#plot_id=1
#plot_id_dict={}
#for  (key, value) in final_obs_grouping.items():
#    for obs in value:
#        plot_id_dict[obs]=plot_id
#    plot_id+=1
#
#print(plot_id_dict)


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
    plt.savefig("miaw.pdf")
    plt.show()


#                info = calcs['Pb-Pb-2760']['dNch_deta']['mean'][:, idf]
#
#
#
#                # dNdeta
#                info = calcs['Pb-Pb-2760']['dNch_deta']['mean'][:, idf]
#                #info = calculate_dNdeta(res, 'ALICE', cenb, idf)
#                #entry['Pb-Pb-2760']['dNch_deta'][:, idf] = info['obs']
#                #if plot:
#                plt.subplot(2,4,1)
#                #plt.errorbar(info['cenM'], info['obs'], yerr=info['err'], fmt='ro-')
#                plt.errorbar(info['cenM'], info['obs'], yerr=info['err'])
#                plt.ylim(0,2000)
#                plt.xlabel(r'Centrality (%)', fontsize=7)
#                #plt.xlim(60,70)
#                #plt.ylim(65,73)
#                plt.ylabel(r'charged $dN/d\eta$', fontsize=7)

#               # dETdeta
#
#               cenb = np.array([[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10], [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20], [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30], [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40], [40, 45], [45, 50], [50, 55], [55, 60], [60, 65], [65, 70]])
#               info = calculate_dETdeta(res, 'ALICE', cenb, idf)
#               entry['Pb-Pb-2760']['dET_deta'][:,idf] = info['obs']
#               if plot:
#                       plt.subplot(2,4,2)
#                       #plt.errorbar(info['cenM'], info['obs'], yerr=info['err'], fmt='bo-')
#                       plt.errorbar(info['cenM'], info['obs'], yerr=info['err'])
#                       plt.ylim(0,2000)
#                       plt.xlabel(r'Centrality (%)', fontsize=7)
#                       plt.ylabel(r'$dE_T/d\eta$', fontsize=7)
#
#               # dN(pid)/dy
#               cenb = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])
#               info = calculate_dNdy(res, 'ALICE', cenb, idf)
#               for (s, _) in pi_K_p:
#                       entry['Pb-Pb-2760']['dN_dy'][s][:,idf] = info['obs'][s]
#               if plot:
#                       plt.subplot(2,4,3)
#                       for (s, _) in pi_K_p:
#                               #plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], fmt='o-', label=s)
#                               plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], label=s)
#                       plt.xlabel(r'Centrality (%)', fontsize=7)
#                       plt.ylabel(r'$dN/dy$', fontsize=7)
#                       plt.legend()
#
#               # dN(pid)/dy exotic
#
#               cenb = np.array([[0,10],[10,20],[20,40],[40,60]])
#               info = calculate_dNdy(res, 'ALICE', cenb, idf)
#               for s in ['Lambda', 'Omega','Xi']:
#                       entry['Pb-Pb-2760']['dN_dy-s'][s][:,idf] = info['obs'][s]
#               if plot:
#                       plt.subplot(2,4,4)
#                       for (s, _) in lambda_omega_xi:
#                               #plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], fmt='o-', label=s)
#                               plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], label=s)
#                       plt.xlabel(r'Centrality (%)', fontsize=7)
#                       plt.ylabel(r'$dN/dy$', fontsize=7)
#                       plt.legend()
#
#
#               # mean-pT
#               cenb = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])
#               info = calculate_mean_pT(res, 'ALICE', cenb, idf)
#               for (s, _) in pi_K_p:
#                       entry['Pb-Pb-2760']['mean_pT'][s][:,idf] = info['obs'][s]
#               if plot:
#                       plt.subplot(2,4,5)
#                       for (s, _) in pi_K_p:
#                               #plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], fmt='o-', label=s)
#                               plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], label=s)
#                       plt.ylim(0,2)
#                       plt.xlabel(r'Centrality (%)', fontsize=7)
#                       plt.ylabel(r'$\langle p_T\rangle$', fontsize=7)
#                       plt.legend()
#
#               # mean-pT-fluct
#               cenb = np.array([[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]])
#               info = calculate_mean_pT_fluct(res, 'ALICE', cenb, idf)
#               entry['Pb-Pb-2760']['pT_fluct'][:,idf] = info['obs']
#               if plot:
#                       plt.subplot(2,4,6)
#                       #plt.errorbar(info['cenM'], info['obs'], yerr=info['err'], fmt='o-')
#                       plt.errorbar(info['cenM'], info['obs'], yerr=info['err'])
#                       plt.xlabel(r'Centrality (%)', fontsize=7)
#                       plt.ylabel(r'$\delta p_T/p_T$', fontsize=7)
#                       #plt.legend()
#
#               # vn
#               cenb = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])
#               #cenb = np.array([[0,10],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80]])
#               info = calculate_vn(res, 'ALICE', cenb, idf)
#               entry['Pb-Pb-2760']['v22'][:,idf] = info['obs'][:, 1]
#               entry['Pb-Pb-2760']['v32'][:,idf]  = info['obs'][:6, 2]
#               entry['Pb-Pb-2760']['v42'][:,idf]  = info['obs'][:6, 3]
#               if plot:
#                       plt.subplot(2,4,7)
#                       for n, c in zip([1,2,3],'rgb'):
#                               #plt.errorbar(info['cenM'], info['obs'][:,n], yerr=info['err'][:,n], fmt=c+'o')
#                               plt.errorbar(info['cenM'], info['obs'][:,n], yerr=info['err'][:,n], label=str(n+1))
#                       plt.xlabel(r'Centrality (%)', fontsize=7)
#                       plt.ylabel(r'$v_n$', fontsize=7)
#                       plt.ylim(0,.15)
#                       plt.legend()
#
#               # vn-diff
#               """
#               cenb = np.array([[30,40]])
#               pTbins = np.array([[0,.2], [.2,.4], [.4,.6],[.6,.8],[.8,1.],
#                               [1.,1.2], [1.2,1.5], [1.5,2.], [2.,2.5], [2.5,3]])
#               info = calculate_diff_vn(res, 'ALICE', cenb, pTbins, idf, pid='chg')
#               entry['Pb-Pb-2760']['v22-d'][:,idf] = info['obs'][:, :, 1]
#               entry['Pb-Pb-2760']['v32-d'][:,idf]  = info['obs'][:, :, 2]
#               entry['Pb-Pb-2760']['v42-d'][:,idf]  = info['obs'][:, :, 3]
#               if plot:
#                       plt.subplot(2,4,7)
#                       for n in np.arange(1,4):
#                               plt.errorbar(info['pTM'], info['obs'][0,:,n], yerr=info['err'][0,:,n], fmt='o-', label=r"$v_{:d}$".format(n+1))
#                       plt.ylim(0,.3)
#                       plt.xlabel(r'$p_T$ [GeV]', fontsize=7)
#                       plt.ylabel(r'charged $v_{:d}$'.format(n+1), fontsize=7)
#                       plt.legend()
#               """



if __name__ == '__main__':
        results = []
        for file in glob.glob(sys.argv[1]):
                calcs = np.fromfile(file, dtype=np.dtype(bayes_dtype))
                nb_obs=len(final_obs_grouping)
                nb_of_cols=4
                nb_of_rows=int(np.ceil(nb_obs/nb_of_cols))
                fig = plt.figure(figsize=(2*nb_of_cols,2*nb_of_rows))
                entry = plot(calcs)
