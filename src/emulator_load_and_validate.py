#!/usr/bin/env python3
import logging
logging.basicConfig(level=logging.DEBUG)
import dill
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from configurations import *
from calculations_file_format_event_average import *
from emulator import *

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
        system_str = "{:s}-{:s}-{:d}".format(*s)
        logging.info("Loading emulators from emulator/emu-" + system_str + '.dill' )
        emu = dill.load(open('emulator/emu-' + system_str + '.dill', "rb"))
        logging.info("NPC = " + str(emu.npc))
        logging.info("idf = " + str(emu.idf))


        #get VALIDATION points
        design_file = design_dir + \
               '/design_points_validation_{:s}{:s}-{:d}.dat'.format(*s)
        logging.info("Loading design points from " + design_file)
        design = pd.read_csv(design_file)
        design = design.drop("idx", axis=1)
        design = design.drop("projectiles", axis=1)
        design = design.drop("cross_section", axis=1)

        #get model calculations at VALIDATION POINTS
        logging.info("Load calculations from " + f_model_validations)
        model_data = np.fromfile(f_model_validations, dtype=bayes_dtype)
  
        #make a plot
        fig, axes = plt.subplots(figsize=(10,6), ncols=5, nrows=3)

        for obs, ax in zip(observables, axes.flatten()):
            Y_true = []
            Y_emu = []
            for ipt, pt in enumerate(design.values):
                mean, cov = emu.predict(np.array([pt]), return_cov=True)
                y_true = model_data[system_str][obs]['mean'][ipt,emu.idf]
                y_emu = mean[obs][0]
                dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                if 'dN' in obs or 'dET' in obs  or 'fluct' in obs:
                    dy_emu = 2*y_emu*dy_emu
                    y_emu = y_emu ** 2
                Y_true = np.concatenate([Y_true, y_true])               
                Y_emu = np.concatenate([Y_emu, y_emu])
            ym, yM = np.min(Y_emu), np.max(Y_emu)
            ax.hist2d(Y_emu, Y_true, bins=31, 
                      cmap='coolwarm', range=[(ym, yM),(ym, yM)])
            ax.plot([ym,yM],[ym,yM],'k--', zorder=100)

            ax.annotate(obs, xy=(.05, .8), xycoords="axes fraction", color='White')
            if ax.is_last_row():
                ax.set_xlabel("Emulated")
            if ax.is_first_col():
                ax.set_ylabel("Computed")
            ax.ticklabel_format(scilimits=(2,1))

        plt.tight_layout(True)

        plt.show()

if __name__ == "__main__":
    main()
