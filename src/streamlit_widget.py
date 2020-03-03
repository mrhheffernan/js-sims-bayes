import streamlit as st
import numpy as np
import time
import matplotlib
import altair as alt
import dill
import matplotlib.pyplot as plt
from configurations import *
from emulator import Trained_Emulators, _Covariance
from bayes_exp import Y_exp_data
from bayes_plot import obs_tex_labels_2

short_names = {
                'norm' : r'$N$',
                'trento_p' : r'$p$',
                'nucleon_width' : r'$w$',
                'sigma_k' : r'$\sigma_k$',
                'dmin3' : r'$d_{min}^3$',
                'tau_R' : r'$\tau_R$',
                'alpha' : r'$\alpha$',
                'eta_over_s_T_kink_in_GeV' : r'$T_{\eta, \rm{kink}}$',
                'eta_over_s_low_T_slope_in_GeV' : r'$a_{\rm{low}}$',
                'eta_over_s_high_T_slope_in_GeV' : r'$a_{\rm{high}}$',
                'eta_over_s_at_kink' : r'$(\eta/s)_{\rm{kink}}$',
                'zeta_over_s_max' : r'$(\zeta/s)_{\rm{max}}$',
                'zeta_over_s_T_peak_in_GeV' : r'T_{\zeta, c}',
                'zeta_over_s_width_in_GeV' : r'$w_{\zeta}$',
                'zeta_over_s_lambda_asymm' : r'$\lambda_{\zeta}$',
                'shear_relax_time_factor' : r'$b_{\pi}$',
                'Tswitch' : 'T_sw',
}


system_observables = {
                    'Pb-Pb-2760' : ['dET_deta', 'dN_dy_pion', 'dN_dy_proton', 'mean_pT_pion', 'mean_pT_proton', 'pT_fluct', 'v22', 'v32', 'v42'],
                    'Au-Au-200' : ['dN_dy_pion', 'dN_dy_kaon', 'mean_pT_pion', 'mean_pT_kaon', 'v22', 'v32']
                    }


system = 'Pb-Pb-2760'

@st.cache(persist=True)
def load_design(system):
    #load the design
    design_file = SystemsInfo[system]["main_design_file"]
    range_file = SystemsInfo[system]["main_range_file"]
    design = pd.read_csv(design_file)
    design = design.drop("idx", axis=1)
    labels = design.keys()
    design_range = pd.read_csv(range_file)
    design_max = design_range['max'].values
    design_min = design_range['min'].values
    return design, labels, design_max, design_min


@st.cache(allow_output_mutation=True)
def load_emu(system, idf):
    #load the emulator
    emu = dill.load(open('emulator/emulator-' + system + '-idf-' + str(idf) + '.dill', "rb"))
    return emu


@st.cache(persist=True)
def load_obs(system):
    observables = system_observables[system]
    nobs = len(observables)
    Yexp = Y_exp_data
    return observables, nobs, Yexp


@st.cache()
def emu_predict(params):
    start = time.time()
    Yemu_cov = 0
    Yemu_mean = emu.predict( np.array( [params] ), return_cov=False )
    #Yemu_mean, Yemu_cov = emu.predict( np.array( [params] ), return_cov=True )
    end = time.time()
    time_emu = end - start
    return Yemu_mean, Yemu_cov, time_emu

def make_plot_altair(Yemu_mean, Yemu_cov, Yexp, idf):
    for iobs, obs in enumerate(observables):
        xbins = np.array(obs_cent_list[system][obs])
        #centrality bins
        x = (xbins[:,0]+xbins[:,1])/2.
        #emulator prediction
        y_emu = Yemu_mean[obs][0]

        #FAKE ERROR BAR FOR TESTING
        dy_emu = y_emu * 0.1
        #dy_emu = (np.diagonal(np.abs(Yemu_cov[obs, obs]))**.5)[:,0]
        df_emu = pd.DataFrame({'cent': x, 'yl':y_emu - dy_emu, "yh":y_emu + dy_emu})
        chart_emu = alt.Chart(df_emu).mark_area().encode(x='cent', y='yl', y2='yh').properties(width=150,height=150)

        #experiment
        exp_mean = Yexp[system][obs]['mean'][idf]
        exp_err = Yexp[system][obs]['err'][idf]
        df_exp = pd.DataFrame({"cent": x, obs:exp_mean, "dy":exp_err})
        #chart_exp = alt.Chart(df_exp).mark_circle(color='Black').encode(x='cent', y=obs)
        chart_exp = alt.Chart(df_exp).mark_circle(color='Black').encode(
        x=alt.X('cent', axis=alt.Axis(title='cent')),
        y=alt.Y(obs, axis=alt.Axis(title=obs))
        )

        chart = alt.layer(chart_emu, chart_exp)

        if iobs == 0:
            charts0 = chart
        if iobs in [1, 2]:
            charts0 = alt.hconcat(charts0, chart)

        if iobs == 3:
            charts1 = chart
        if iobs in [4, 5]:
            charts1 = alt.hconcat(charts1, chart)

        if iobs == 6:
            charts2 = chart
        if iobs in [7, 8]:
            charts2 = alt.hconcat(charts2, chart)

    st.write(charts0)
    st.write(charts1)
    st.write(charts2)


system = 'Pb-Pb-2760'
idf_names = ['Grad', 'C.E. RTA', 'Pratt-McNelis', 'Pratt-Bernhard']
idf_name = st.selectbox('Viscous Correction',idf_names)
idf = idf_names.index(idf_name)

#load the design
design, labels, design_max, design_min = load_design(system)

#load the emu
emu = load_emu(system, idf)

#load the exp obs
observables, nobs, Yexp = load_obs(system)

#initialize parameters
params_0 = MAP_params[system][ idf_label_short[idf] ]

#updated params
params = []
for i_s, s_name in enumerate(short_names.keys()):
    #s = st.markdown(short_names[s_name])
    min = design_min[i_s]
    max = design_max[i_s]
    p = st.sidebar.slider(short_names[s_name], min_value=min, max_value=max, value=params_0[i_s])
    params.append(p)

#get emu prediction
Yemu_mean, Yemu_cov, time_emu = emu_predict(params)

make_plot_altair(Yemu_mean, Yemu_cov, Yexp, idf)
