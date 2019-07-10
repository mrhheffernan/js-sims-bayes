#!/usr/bin/env python3
import matplotlib.pyplot as plt
import dill
import numpy as np
import pandas as pd

from configurations import *
from calculations_file_format_event_average import *
from emulator import _Covariance

def main():
    for s in systems:

        observables = []
        for obs, cent_list in obs_cent_list.items():
            observables.append(obs)

        #smaller subset of observables for plotting
        observables = [
        'dNch_deta',
        'dET_deta',
        'dN_dy_pion',
        'dN_dy_kaon',
        'dN_dy_proton',
        'dN_dy_Lambda',
        'dN_dy_Omega',
        'dN_dy_Xi',
        'mean_pT_pion',
        'mean_pT_kaon',
        'mean_pT_proton',
        'pT_fluct',
        'v22',
        'v32',
        'v42'
        ]

        #load the dill'ed emulator from emulator file
        system_str = s[0]+"-"+s[1]+"-"+str(s[2])
        print("Loading dilled emulator from emulator/emu-" + system_str + '.dill' )
        emu = dill.load(open('emulator/emu-' + system_str + '.dill', "rb"))
        print("Number of principal components : " + str(emu.npc) )


        #get design points
        design_dir = 'design_pts'
        print("Reading in design points from " + str(design_dir))
        design = pd.read_csv(design_dir + '/design_points_main_PbPb-2760.dat')


        #get model calculations
        idf = 0
        model_file = 'model_calculations/obs.dat'
        print("Reading model calculated observables from " + model_file)
        model_data = np.fromfile(model_file, dtype=bayes_dtype)

        #make a plot
        fig, axes = plt.subplots(figsize=(10,6), ncols=5, nrows=3)

        for obs, ax in zip(observables, axes.flatten()):
            for pt in design.values:
                ipt = int(pt[0])
                pt = pt[1:]
                mean, cov = emu.predict(np.array([pt]), return_cov=True)
                y_emu = mean[obs][0]
                dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                if 'dN' in obs or 'dET' in obs  or 'fluct' in obs:
                    dy_emu = 2*y_emu*dy_emu
                    y_emu = y_emu ** 2
                ax.errorbar(y_emu, model_data[system_str][obs]['mean'][ipt,idf], alpha=0.1, fmt='r.', xerr=3*dy_emu)
                if ipt ==0:
                    fM = y_emu.max()
                    fm = y_emu.min()        
                else:  
                    fM = np.max([fM, y_emu.max()])
                    fm = np.min([fm, y_emu.min()])
            ax.annotate(obs, xy=(.05, .8), xycoords="axes fraction")
            ax.plot([fm,fM],[fm,fM],'k--', zorder=100)
            if ax.is_last_row():
                ax.set_xlabel("Emulated")
            if ax.is_first_col():
                ax.set_ylabel("Computed")
            ax.ticklabel_format(scilimits=(2,1))

        plt.tight_layout(True)

        plt.show()

if __name__ == "__main__":
    main()
