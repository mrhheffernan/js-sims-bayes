#!/usr/bin/env python3
import logging
import dill
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from configurations import *
from emulator import *
from calculations_load import validation_data, trimmed_model_data

def plot_residuals(system_str, emu, design, cent_bin, observables, nrows, ncols):
    """
    Plot a histogram of the percent difference between the emulator
    prediction and the model at design points in either training or validation sets.
    """

    print("Plotting emulator residuals")

    fig, axes = plt.subplots(figsize=(10,8), ncols=ncols, nrows=nrows)
    for obs, ax in zip(observables, axes.flatten()):
        Y_true = []
        Y_emu = []

        if crossvalidation:
            for pt in cross_validation_pts:
                params = design.iloc[pt].values
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true = validation_data[system_str][pt, idf][obs]['mean'][cent_bin]
                y_emu = mean[obs][0][cent_bin]
                is_mult = ('dN' in obs) or ('dET' in obs)
                if is_mult and transform_multiplicities:
                    y_emu = np.exp(y_emu) - 1.
                    y_true = np.exp(y_true) - 1.
                #dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                Y_true.append(y_true)
                Y_emu.append(y_emu)

        else :
            for pt, params in enumerate(design.values):
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true = validation_data[system_str][pt, idf][obs]['mean'][cent_bin]
                y_emu = mean[obs][0][cent_bin]
                is_mult = ('dN' in obs) or ('dET' in obs)
                if is_mult and transform_multiplicities:
                    y_emu = np.exp(y_emu) - 1.
                    y_true = np.exp(y_true) - 1.
                #dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                Y_true.append(y_true)
                Y_emu.append(y_emu)

        Y_true = np.array(Y_true)
        Y_emu = np.array(Y_emu)

        residuals = (Y_emu - Y_true) / Y_emu
        std_resid = np.sqrt( np.var(residuals) )
        bins = np.linspace(-0.5, 0.5, 31)
        ax.hist(residuals, bins = bins, density = True)
        ax.set_xlim(-0.5, 0.5)

        ax.set_title(obs)
        ax.annotate(" std : " + str( round(std_resid, 2) ), xy=(.05, .8), xycoords = "axes fraction")

    plt.tight_layout(True)
    plt.savefig('validation_plots/emulator_residuals.png')

    #plt.show()

def plot_residuals_corr(system_str, emu, design, cent_bin, observables):
    """
    Plot a histogram of the percent difference between the emulator
    prediction and the model at design points in either training or validation sets.
    """

    print("Plotting emulator residuals obs1 vs obs2 ")
    ncols = nrows = len(observables)

    fig, axes = plt.subplots(figsize=(30,30), ncols=ncols, nrows=nrows)
    bins = np.linspace(-0.5, 0.5, 31)
    for row, obs1 in enumerate(observables):
        for col, obs2 in enumerate(observables):
            residuals_1 = []
            residuals_2 = []
            for pt, params in enumerate(design.values):
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true1 = validation_data[system_str][pt, idf][obs1]['mean'][cent_bin]
                y_emu1 = mean[obs1][0][cent_bin]
                res_1 = (y_emu1 - y_true1) / y_emu1
                residuals_1.append(res_1)
                y_true2 = validation_data[system_str][pt, idf][obs2]['mean'][cent_bin]
                y_emu2 = mean[obs2][0][cent_bin]
                res_2 = (y_emu2 - y_true2) / y_emu2
                residuals_2.append(res_2)

            axes[row,col].scatter(residuals_1, residuals_2)
            axes[row,col].set_xlabel(obs1)
            axes[row,col].set_ylabel(obs2)
            #axes[row, col].set_title(obs2 + " vs " + obs1 + " residuals")
            #if axes[row,col].is_last_row():
            #    axes[row,col].set_xlabel("res : " + obs1)
            #if axes[row,col].is_first_col():
            #    axes[row,col].set_ylabel("res : " + obs2)
            #axes[row, col].set_xlim(-0.5, 0.5)
            #axes[row, col].hist2d(res_1, res_2, bins = [bins, bins], density = True)

    plt.savefig('validation_plots/emulator_residuals_corr.png')

def plot_scatter(system_str, emu, design, cent_bin, observables, nrows, ncols):
    """
    Plot a scatter plot of the emulator prediction vs the model prediction at
    design points in either training or testing set.
    """

    print("Plotting scatter plot of emulator vs model")

    fig, axes = plt.subplots(figsize=(10,8), ncols=ncols, nrows=nrows)
    for obs, ax in zip(observables, axes.flatten()):
        Y_true = []
        Y_emu = []

        if crossvalidation:
            for pt in cross_validation_pts:
                params = design.iloc[pt].values
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true = validation_data[system_str][pt, idf][obs]['mean'][cent_bin]
                y_emu = mean[obs][0][cent_bin]
                is_mult = ('dN' in obs) or ('dET' in obs)
                if is_mult and transform_multiplicities:
                    y_emu = np.exp(y_emu) - 1.
                    y_true = np.exp(y_true) - 1.
                #dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                Y_true.append(y_true)
                Y_emu.append(y_emu)

        else :
            for pt, params in enumerate(design.values):
                mean, cov = emu.predict(np.array([params]), return_cov = True)
                y_true = validation_data[system_str][pt, idf][obs]['mean'][cent_bin]
                y_emu = mean[obs][0][cent_bin]
                is_mult = ('dN' in obs) or ('dET' in obs)
                if is_mult and transform_multiplicities:
                    y_emu = np.exp(y_emu) - 1.
                    y_true = np.exp(y_true) - 1.
                #dy_emu = (np.diagonal(cov[obs, obs])**.5)[:,0]
                Y_true.append(y_true)
                Y_emu.append(y_emu)

        Y_true = np.array(Y_true)
        Y_emu = np.array(Y_emu)
        ym, yM = np.min(Y_emu), np.max(Y_emu)
        h = ax.hist2d(Y_emu, Y_true, bins=31, cmap='coolwarm', range=[(ym, yM),(ym, yM)])
        ym, yM = ym-(yM-ym)*.05, yM+(yM-ym)*.05
        ax.plot([ym,yM],[ym,yM],'k--', zorder=100)

        ax.annotate(obs, xy=(.05, .8), xycoords="axes fraction", color='White')
        if ax.is_last_row():
            ax.set_xlabel("Emulated")
        if ax.is_first_col():
            ax.set_ylabel("Computed")
        ax.ticklabel_format(scilimits=(2,1))

    plt.tight_layout(True)
    plt.savefig('validation_plots/emulator_vs_model.png')

    #plt.show()

def plot_model_stat_uncertainty(system_str, design, cent_bin, observables, nrows, ncols):
    """
    Plot the model uncertainty for all observables
    """
    print("Plotting model stat. uncertainty")

    fig, axes = plt.subplots(figsize=(10,8), ncols=ncols, nrows=nrows)
    for obs, ax in zip(observables, axes.flatten()):


        values = []
        errors = []
        rel_errors = []

        #note if transformation of multiplicities is turned on!!!
        for pt in range( n_design_pts_main - len(delete_design_pts_set) ):
            val = trimmed_model_data[system_str][pt, idf][obs]['mean'][cent_bin]
            err = trimmed_model_data[system_str][pt, idf][obs]['err'][cent_bin]
            values.append(val)
            errors.append(err)
            if val > 0.0:
                if np.isnan(err / val):
                    print("nan")
                else :
                    rel_errors.append(err / val)

        rel_errors = np.array(rel_errors)
        std = np.sqrt( np.var(rel_errors) )
        mean = np.mean(rel_errors)

        rel_errors = rel_errors[ rel_errors < (mean + 5.*std) ]

        ax.hist(rel_errors, 20)
        ax.set_title(obs)

        if ax.is_last_row():
            ax.set_xlabel("relative error")

    #plt.suptitle('Distribution of model statistical error')
    plt.tight_layout(True)
    plt.savefig('validation_plots/model_stat_errors.png')

def main():

    cent_bin = 3

    for s in systems:
        system_str = "{:s}-{:s}-{:d}".format(*s)

        observables = []
        for obs, cent_list in obs_cent_list[system_str].items():
            observables.append(obs)

        nrows = 3
        ncols = 3

        if pseudovalidation:
            #using training points as testing points
            design, design_max, design_min, labels = load_design(system = s, pset='main')
        else :
            design, design_max, design_min, labels = load_design(system = s, pset='validation')

        print("Validation design set shape : (Npoints, Nparams) =  ", design.shape)

        #load the dill'ed emulator from emulator file
        print("Loading emulators from emulator/emulator-" + system_str + '-idf-' + str(idf) + '.dill' )
        emu = dill.load(open('emulator/emulator-' + system_str + '-idf-' + str(idf) + '.dill', "rb"))
        print("NPC = " + str(emu.npc))
        print("idf = " + str(idf))

        #make a plot of the residuals ; percent difference between emulator and model
        plot_residuals(system_str, emu, design, cent_bin, observables, nrows, ncols)
        #make a scatter plot to check if residuals between different observables are correlated
        plot_residuals_corr(system_str, emu, design, cent_bin, observables)
        #make a scatter plot of emulator prediction vs model prediction
        plot_scatter(system_str, emu, design, cent_bin, observables, nrows, ncols)
        #make a histogram to check the model statistical uncertainty
        plot_model_stat_uncertainty(system_str, design, cent_bin, observables, nrows, ncols)


if __name__ == "__main__":
    main()
