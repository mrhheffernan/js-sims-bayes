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
        #'dET_deta',
        'dN_dy_pion',
        'dN_dy_kaon',
        'dN_dy_proton',
        #'dN_dy_Lambda',
        #'dN_dy_Omega',
        #'dN_dy_Xi',
        #'mean_pT_pion',
        #'mean_pT_kaon',
        #'mean_pT_proton',
        #'pT_fluct',
        'v22',
        'v32',
        #'v42'
        ]

        #load the dill'ed emulator from emulator file
        system_str = s[0]+"-"+s[1]+"-"+str(s[2])
        print("Loading dilled emulator from emulator/emu-" + system_str + '.dill' )
        emu = dill.load(open('emulator/emu-' + system_str + '.dill', "rb"))
        print("Number of principal components : " + str(emu.npc) )

        #test emulator at a series of points in parameter space
        test_pts = np.random.uniform(0.0, 0.2, size=15)
        X = test_pts.reshape(-1,1)

        mean, cov = emu.predict(X, return_cov=True)

        #get design points
        design_dir = 'design_pts'
        print("Reading in design points from " + str(design_dir))
        design = pd.read_csv(design_dir + '/design_points_main_PbPb-2760.dat')
        design_vals = design['etas_min'].values

        #get model calculations
        #for testing idf = 3
        idf = 3
        model_file = 'model_calculations/obs.dat'
        print("Reading model calculated observables from " + model_file)
        model_data = np.fromfile(model_file, dtype=bayes_dtype)

        #make a plot
        nrows = 2
        ncols = 3
        fig, axs = plt.subplots(nrows,ncols)
        fig.tight_layout()
        n = 0
        for obs in observables:
            y_emu = mean[obs]
            dy_emu = cov[obs, obs]
            Y = []
            Y_err = []
            for pt in range(n_design_pts):
                Y.append( model_data[system_str][obs]['mean'][pt,idf] )
                Y_err.append( model_data[system_str][obs]['stat_err'][pt,idf] )


            Y = np.array(Y)
            Y_err = np.array(Y_err)

            #take a particular centrality bin
            y_emu_bin = y_emu[:, 2]
            dy_emu_bin = dy_emu[:, 2, 2]
            Y_bin = Y[:, 2]
            Y_err_bin = Y_err[:,2]

            #print("obs = " + str(obs))
            #print("dy_emu_bin = " +str(dy_emu_bin))

            #plot the emulator prediction against model data for a observable
            x = n // ncols
            y = n - (x * ncols)

            axs[x,y].scatter(design_vals, Y_bin, label='model')
            axs[x,y].errorbar(design_vals, Y_bin, Y_err_bin, ls='none')
            axs[x,y].scatter(test_pts, y_emu_bin, label='emulator', color='r')
            axs[x,y].errorbar(test_pts, y_emu_bin, dy_emu_bin, color='r', ls='none')
            axs[x,y].set(xlabel=r'$(\eta / s)_{min}$')
            axs[x,y].set_title(obs)

            if (obs == 'v22'):
                axs[x,y].set_ylim(.05, 0.09)
            if (obs == 'v32'):
                axs[x,y].set_ylim(.02, 0.045)
            n +=1

        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
