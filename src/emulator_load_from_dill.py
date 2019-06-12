import matplotlib.pyplot as plt
import dill
import numpy as np
import pandas as pd

from configurations import *
from calculations_file_format_event_average import *

def main():
    for s in systems:
        #load the dill'ed emulator from emulator file
        system_str = s[0]+"-"+s[1]+"-"+str(s[2])
        emu = dill.load(open('emulator/emu-' + system_str + '.dill', "rb"))

        #test emulator at a series of points in parameter space
        test_pts = np.random.uniform(0.0, 0.2, size=15)
        X = test_pts.reshape(1,-1)
        predictions = [emu.predict_2(x.reshape(1,-1)) for x in X.T]

        predictions = np.array(predictions)
        print("predictions.shape = " + str(predictions.shape) )

        y_emu = []
        dy_emu = []

        for p in predictions:
            y, dy = p
            y_emu.append(y)
            dy_emu.append(dy)

        y_emu=np.array(y_emu)
        dy_emu=np.array(dy_emu)

        #get design points
        design_dir = 'design_pts'
        print("Reading in design points from " + str(design_dir))
        design = pd.read_csv(design_dir + '/design_points_main_PbPb-2760.dat')
        #design_vals = design['tau_fs'].values
        design_vals = design['etas_min'].values

        #get model calculations
        #for testing idf = 3
        idf = 3
        #obs = 'dET_deta'
        obs = 'v22'
        model_file = 'model_calculations/obs.dat'
        print("Reading model calculated observables from " + model_file)
        model_data = np.fromfile(model_file, dtype=bayes_dtype)

        Y = []
        Y_err = []

        for pt in range(n_design_pts):
            #Y.append(model_data[system_str][pt, idf][obs]['mean'])
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
        plt.scatter(design_vals, Y_2030, label='model')
        plt.errorbar(design_vals, Y_2030, Y_err_2030, ls='none')
        plt.scatter(test_pts, y_emu_2030, label='emulator', color='r')
        plt.errorbar(test_pts, y_emu_2030, 2*dy_emu_2030, ls='none', color='r')
        plt.legend()
        plt.xlabel(r'$(\eta / s)_{min}$')
        plt.ylabel(r'$v_{2,2}$')
        plt.show()

main()
