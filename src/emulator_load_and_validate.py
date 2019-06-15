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

        #load the dill'ed emulator from emulator file
        system_str = s[0]+"-"+s[1]+"-"+str(s[2])
        print("Loading dilled emulator from emulator/emu-" + system_str + '.dill' )
        emu = dill.load(open('emulator/emu-' + system_str + '.dill', "rb"))
        print("Number of principal components : " + str(emu.npc) )

        #test emulator at a series of points in parameter space
        test_pts = np.random.uniform(0.0, 0.2, size=15)
        X = test_pts.reshape(-1,1)

        mean, cov = emu.predict(X, return_cov=True)

        #print(cov['dNch_deta', 'dNch_deta'])

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

        fig, axs = plt.subplots(3,5)
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
            y_emu_2030 = y_emu[:, 2]
            dy_emu_2030 = dy_emu[:, 2]

            Y_2030 = Y[:, 2]
            Y_err_2030 = Y[:,2]

            #plot the emulator prediction against model data for a observable
            x = n % 3
            y = n % 5
            axs[x,y].scatter(design_vals, Y_2030, label='model')
            axs[x,y].errorbar(design_vals, Y_2030, Y_err_2030, ls='none')
            axs[x,y].scatter(test_pts, y_emu_2030, label='emulator', color='r')
            #plt.errorbar(test_pts, y_emu_2030, 2*dy_emu_2030, ls='none', color='r')
            axs[x,y].set(xlabel=r'$(\eta / s)_{min}$', ylabel=obs)
            n +=1

        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
