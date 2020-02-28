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
    #Yemu_mean = emu.predict( np.array( [params] ), return_cov=False )
    Yemu_mean, Yemu_cov = emu.predict( np.array( [params] ), return_cov=True )
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
        #dy_emu = y_emu * 0.1
        dy_emu = (np.diagonal(np.abs(Yemu_cov[obs, obs]))**.5)[:,0]
        df_emu = pd.DataFrame({'cent': x, 'yl':y_emu - dy_emu, "yh":y_emu + dy_emu})
        chart_emu = alt.Chart(df_emu).mark_area().encode(x='cent', y='yl', y2='yh').properties(width=100,height=100)

        #chart_emu = alt.Chart(df_emu).mark_area().encode(
        #x=alt.X('cent', axis=alt.Axis(labels=False)),
        #y=alt.Y('yl', axis=alt.Axis(labels=False)),
        #y2=alt.Y2('yh', axis=alt.Axis(labels=False)),
        #)


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

#get emu prediction
params = [14.128, 0.089, 1.054, 1.064, 4.227, 1.507, 0.113, 0.223, -1.585, 0.32, 0.056, 0.11, 0.16, 0.093, -0.084, 4.666, 0.136]
Yemu_mean, Yemu_cov, time_emu = emu_predict(params)

make_plot_altair(Yemu_mean, Yemu_cov, Yexp, idf)

x = st.slider('x')
st.write(x, 'squared is', x * x)
