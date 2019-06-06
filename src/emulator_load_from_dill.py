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
        test_pts = np.random.uniform(0.3, 1.2, size=5)
        X = test_pts.reshape(1,-1)

        test_pt = 1.0
        X = [[test_pt]]
        predictions = emu.predict_2(X)

        y_emu = predictions[0]
        dy_emu = predictions[1]

        #get design points
        design_dir = 'design_pts'
        design = pd.read_csv(design_dir + '/design_points_main_PbPb-2760.dat')
        design_vals = design['tau_fs'].values

        #get model calculations
        #for testing idf = 3
        idf = 3
        obs = 'dET_deta'
        model_file = 'model_calculations/obs.dat'
        print("Reading model calculated observables from " + model_file)
        model_data = np.fromfile(model_file, dtype=bayes_dtype)
        Y = []
        for pt in range(n_design_pts):
            Y.append(model_data[system_str][pt, idf][obs]['mean'])

        #plot the emulator prediction against model data
        plt.scatter(design_vals, Y, label='model')
        plt.scatter(test_pt, predictions[0], label='emulator')
        plt.errorbar(test_pt, y_emu, yerr=2*dy_emu)
        plt.legend()
        plt.xlabel(r'$\tau_{fs}$')
        plt.ylabel(r'$dN_{ch}/d_\eta$')
        plt.show()
main()
