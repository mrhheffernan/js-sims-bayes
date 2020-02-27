import streamlit as st

import numpy as np
import matplotlib
import dill
import matplotlib.pyplot as plt
from configurations import *
from emulator import Trained_Emulators, _Covariance
from bayes_exp import Y_exp_data
from bayes_plot import obs_tex_labels_2

short_names = {
                'norm' : 'N',
                'trento_p' : 'p',
                'nucleon_width' : 'w',
                'sigma_k' : 'sigma_k',
                'dmin3' : 'd_{min}^3',
                'tau_R' : 'tau_R',
                'alpha' : 'alpha',
                'eta_over_s_T_kink_in_GeV' : 'eta_Tk',
                'eta_over_s_low_T_slope_in_GeV' : 'eta_low',
                'eta_over_s_high_T_slope_in_GeV' : 'eta_high',
                'eta_over_s_at_kink' : 'eta_k',
                'zeta_over_s_max' : 'zeta_max',
                'zeta_over_s_T_peak_in_GeV' : 'zeta_Tc',
                'zeta_over_s_width_in_GeV' : 'zeta_w',
                'zeta_over_s_lambda_asymm' : 'zeta_asym',
                'shear_relax_time_factor' : 'b_pi',
                'Tswitch' : 'T_sw',
}


system_observables = {
                    'Pb-Pb-2760' : ['dET_deta', 'dN_dy_pion', 'dN_dy_proton', 'mean_pT_pion', 'mean_pT_proton', 'pT_fluct', 'v22', 'v32', 'v42'],
                    'Au-Au-200' : ['dN_dy_pion', 'dN_dy_kaon', 'mean_pT_pion', 'mean_pT_kaon', 'v22', 'v32']
                    }


system = 'Pb-Pb-2760'
Yexp = Y_exp_data

#load the design
design_file = SystemsInfo[system]["main_design_file"]
range_file = SystemsInfo[system]["main_range_file"]
design = pd.read_csv(design_file)
design = design.drop("idx", axis=1)
labels = design.keys()
design_range = pd.read_csv(range_file)
design_max = design_range['max'].values
design_min = design_range['min'].values
Names = labels
idf = 0

#load the emulator
emu = dill.load(open('emulator/emulator-' + 'Pb-Pb-2760' + '-idf-' + str(idf) + '.dill', "rb"))


observables = system_observables[system]
nobs = len(observables)
nrows = 3
fig, axes = plt.subplots(nrows=nrows, ncols= nobs // nrows, figsize=(6,6))


#update the parameters with slider HERE
#params = [getattr(self, 'tune'+name).get() for name in Names]

#params = MAP_params[system]


params = [14.128, 0.089, 1.054, 1.064, 4.227, 1.507, 0.113, 0.223, -1.585, 0.32, 0.056, 0.11, 0.16, 0.093, -0.084, 4.666, 0.136]
Yemu_mean, Yemu_cov = emu.predict( np.array( [params] ), return_cov=True )

for obs, ax in zip(observables, axes.flatten()):
    ax.cla()
    ax.set_title(obs_tex_labels_2[obs])
    xbins = np.array(obs_cent_list[system][obs])
    #centrality bins
    x = (xbins[:,0]+xbins[:,1])/2.
    #emulator prediction
    y_emu = Yemu_mean[obs][0]
    dy_emu = (np.diagonal(np.abs(Yemu_cov[obs, obs]))**.5)[:,0]
    ax.fill_between(x, y_emu-dy_emu, y_emu+dy_emu)
    #experiment
    exp_mean = Yexp[system][obs]['mean'][idf]
    exp_err = Yexp[system][obs]['err'][idf]
    ax.errorbar( x, exp_mean, exp_err, color='black', marker='v')

    if obs == 'dNch_deta':
        ax.set_ylim(0, 2e3)
    if obs == 'dN_dy_pion':
        ax.set_ylim(0, 2e3)
    if obs == 'dN_dy_proton':
        ax.set_ylim(0, 1e2)
    if obs == 'dET_deta':
        ax.set_ylim(0, 2.5e3)
    if obs == 'mean_pT_pion':
        ax.set_ylim(0.3, 0.7)
    if obs == 'mean_pT_proton':
        ax.set_ylim(0.75, 1.75)
    if obs == 'pT_fluct':
        ax.set_ylim(0, 0.04)
    if obs == 'v22':
        ax.set_ylim(0, 0.13)
    if obs == 'v32':
        ax.set_ylim(0.01, 0.04)
    if obs == 'v42':
        ax.set_ylim(0.005, 0.018)


plt.tight_layout(True)
st.pyplot(fig)

x = st.slider('x')
st.write(x, 'squared is', x * x)
