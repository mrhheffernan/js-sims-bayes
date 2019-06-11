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

        #use the emulator to approximate model output

        #a single point
        #test_pt = 1.0
        #X = [[test_pt]]
        #predictions = emu.predict_2(X)

        # a series of points
        test_pts = np.random.uniform(0.0, 0.2, size=20)
        X = test_pts.reshape(1,-1)
        predictions = [emu.predict_2(x.reshape(1,-1)) for x in X.T]

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

        #plot the emulator prediction against model data
        plt.scatter(design_vals, Y, label='model')
        plt.errorbar(design_vals, Y, Y_err, ls='none')
        plt.scatter(test_pts, y_emu, label='emulator', color='r')
        plt.errorbar(test_pts, y_emu, 2*dy_emu, ls='none', color='r')
        #plt.fill_between(
        #    test_pts, y_emu - 2*dy_emu, y_emu + 2*dy_emu,
        #    color=plt.cm.Blues(.6), alpha=.2, lw=0, zorder=0
        #)
        plt.legend()
        plt.xlabel(r'$(\eta / s)_{min}$')
        plt.ylabel(r'$v_{2,2}$')
        #plt.xlim(0.2,1.5)
        #plt.ylim(560,640)
        plt.show()
main()
