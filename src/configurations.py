#!/usr/bin/env python3
import os, logging
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from bins_and_cuts import obs_cent_list, obs_range_list

workdir = Path(os.getenv('WORKDIR', '.'))

# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'

#how many versions of the model are run, for instance
# 4 versions of delta-f with SMASH and a fifth model with UrQMD totals 5
number_of_models_per_run = 5

#the Collision systems
systems = [('Pb', 'Pb', 2760)]
system_strs = ['{:s}-{:s}-{:d}'.format(*s) for s in systems]

#the number of design points
n_design_pts_main = 50
n_design_pts_validation = 50

#runid = "run-19p-200d"
runid = "check_prior_3"

f_events_main = str(
    workdir/'model_calculations/{:s}/Events/main/'.format(runid))
f_events_validation = str(
    workdir/'model_calculations/{:s}/Events/validation/'.format(runid))
f_obs_main = str(
    workdir/'model_calculations/{:s}/Obs/main.dat'.format(runid))
f_obs_validation = str(
    workdir/'model_calculations/{:s}/Obs/validation.dat'.format(runid))

design_dir =  str(workdir/'design_pts') #folder containing design points

idf = 0 # the choice of viscous correction

validation = 0

bayes_dtype=[    (s,
                  [(obs, [("mean",float_t,len(cent_list)),
                          ("err",float_t,len(cent_list))]) \
                    for obs, cent_list in obs_cent_list[s].items() ],
                  number_of_models_per_run
                 ) \
                 for s in system_strs
            ]

# The active ones used in Bayes analysis (MCMC)
active_obs_list = {
   sys: list(obs_cent_list[sys].keys()) for sys in system_strs
}


#the EOS is useful if we want a transformed design on zeta/s, tau_Pi etc...
"""
def compute_cs2_soverh():
    e, p, s, t = np.fromfile('EOS/hrg_hotqcd_eos_binary.dat').reshape(-1,4).T
    cs2 = interp1d((t[1:]+t[:-1])/2., (p[1:]-p[:-1])/(e[1:]-e[:-1]), fill_value=0., bounds_error=False)
    soverh = interp1d(t, s/(e+p), fill_value=0., bounds_error=False)
    return cs2, soverh

cs2, soverh = compute_cs2_soverh()

def zetas(T, zmax, T0, width, asym):
    DeltaT = T - T0
    sign = 1 if DeltaT>0 else -1
    x = DeltaT/(width*(1.+asym*sign))
    return zmax/(1.+x**2)
zetas = np.vectorize(zetas)

def etas(T, T_k, alow, ahigh, etas_k):
    if T < T_k:
        y = etas_k + alow*(T-T_k)
    else:
        y = etas_k + ahigh*(T-T_k)
    if y > 0:
        return y
    else:
        return 0.
etas = np.vectorize(etas)

def tauPi(T, bPi, zmax, T0, width, asym, power):
    return bPi * 0.18743**power * zetas(T, zmax, T0, width, asym) / ( 1./3.-cs2(T) )**power * soverh(T)
tauPi = np.vectorize(tauPi)

def taupi(T, bpi, Tk, alow, ahigh, etas_k):
    return bpi * etas(T, Tk, alow, ahigh, etas_k) * soverh(T)
taupi = np.vectorize(taupi)

"""
"""
def tau_fs(e, tauR, alpha):
    e0 = 1.
    return tauR * (e/e0)**alpha
tau_fs = np.vectorize(tau_fs)
"""

# load design for other module
def load_design(system=('Pb','Pb',2760), pset='main'): # or validation
    design_file = design_dir + \
       '/design_points_{:s}_{:s}{:s}-{:d}.dat'.format(pset, *system)
    range_file = design_dir + \
       '/design_ranges_{:s}_{:s}{:s}-{:d}.dat'.format(pset, *system)
    print("Loading {:s} points from {:s}".format(pset, design_file) )
    print("Loading {:s} ranges from {:s}".format(pset, range_file) )
    with open(design_dir+'/design_labels_{:s}{:s}-{:d}.dat'.format(*system),\
              'r') as f:
         labels = [r""+line[:-1] for line in f]
    # design
    design = pd.read_csv(design_file)
    design = design.drop("idx", axis=1)
    design_range = pd.read_csv(range_file)
    design_max = design_range['max'].values
    design_min = design_range['min'].values
    return design, design_min, design_max, labels

# A spectially transformed design for the emulators
# 0    1        2       3             4
# norm trento_p sigma_k nucleon_width dmin3
#
# 5     6     7
# tau_R alpha eta_over_s_T_kink_in_GeV
#
# 8                             9                              10
# eta_over_s_low_T_slope_in_GeV eta_over_s_high_T_slope_in_GeV eta_over_s_at_kink,
#
# 11              12                        13
# zeta_over_s_max zeta_over_s_T_peak_in_GeV zeta_over_s_width_in_GeV
#
# 14                       15                      16
# zeta_over_s_lambda_asymm shear_relax_time_factor bulk_relax_time_factor
#
# 17                    18
# bulk_relax_time_power Tswitch



def transform_design(X):

    """
    t0 = tau_fs(20, X[:, 5], X[:, 6])
    t1 = tau_fs(4, X[:, 5], X[:, 6])

    X[:,5] = t0
    X[:,6] = t1


    tPi = tauPi(0.3,
                X[:, 16],
                X[:, 11], X[:, 12], X[:, 13], X[:, 14],
                X[:, 17])

    tpi = taupi(0.3,
                X[:, 15],
                X[:, 7], X[:, 8], X[:, 9], X[:, 10])


    e1 = etas(.15,
              X[:, 7], X[:, 8], X[:, 9], X[:, 10])
    e2 = etas(.2,
              X[:, 7], X[:, 8], X[:, 9], X[:, 10])
    e3 = etas(.3,
              X[:, 7], X[:, 8], X[:, 9], X[:, 10])
    e4 = etas(.4,
              X[:, 7], X[:, 8], X[:, 9], X[:, 10])

    z1 = zetas(.15,
               X[:, 11], X[:, 12], X[:, 13], X[:, 14])
    z2 = zetas(.2,
               X[:, 11], X[:, 12], X[:, 13], X[:, 14])
    z3 = zetas(.3,
               X[:, 11], X[:, 12], X[:, 13], X[:, 14])
    z4 = zetas(.4,
               X[:, 11], X[:, 12], X[:, 13], X[:, 14])

    X[:, 16] = tPi
    X[:, 15] = tpi
    X[:, 7] = e1
    X[:, 8] = e2
    X[:, 9] = e3
    X[:, 10] = e4
    X[:, 11] = z1
    X[:, 12] = z2
    X[:, 13] = z3
    X[:, 14] = z4
    """

    return X


def prepare_emu_design():
    design, design_max, design_min, labels = \
          load_design(system=('Pb','Pb',2760), pset='main')

    #not transforming design of any parameters right now
    design = transform_design(design.values)

    design_max = np.max(design, axis=0)
    design_min = np.min(design, axis=0)
    return design, design_max, design_min, labels
