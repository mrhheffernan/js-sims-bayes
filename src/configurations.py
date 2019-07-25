import os
from pathlib import Path
import numpy as np

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
n_design_pts = 100

# Number of principal components to keep in the emulator
npca=4

f_events_main = str(workdir/'model_calculations/Events/main/')
f_events_validation = str(workdir/'model_calculations/Events/validation/')
f_obs_main = str(workdir/'model_calculations/Obs/main.dat')
f_obs_validation = str(workdir/'model_calculations/Obs/validation.dat')
design_dir =  str(workdir/'design_pts')

idf = 3
validation=5

""" 
# full bins
dNch_deta_cents = [[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]] #8 bins
dET_deta_cent=[[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10], [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20], [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30], [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40], [40, 45], [45, 50], [50, 55], [55, 60], [60, 65], [65, 70]] # 22 bins
dN_dy_cents = [[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]] # 8 bins
dN_dy_strange_cents=[[0,10],[10,20],[20,40],[40,60]] # 4 bins
mean_pt_cents=[[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]] # 8 bins
pT_fluct_cents=[[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]] #12 bins
vn_cents=[[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]] # 8 bins
"""

# more central bins
dNch_deta_cents = [[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60]] #8 bins
dET_deta_cent=[[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10], [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20], [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30], [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40], [40, 45], [45, 50], [50, 55], [55, 60]] # 22 bins
dN_dy_cents = [[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60]] # 8 bins
dN_dy_strange_cents=[[0,10],[10,20],[20,40],[40,60]] # 4 bins
mean_pt_cents=[[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60]] # 8 bins
pT_fluct_cents=[[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]] #12 bins
vn_cents=[[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60]] # 8 bins


# Observable name, data type, centralities
# totals 15 observable types
obs_cent_list={
    'Pb-Pb-2760': {
		'dNch_deta': dNch_deta_cents,
		'dET_deta': dET_deta_cent,
		'dN_dy_pion': dN_dy_cents,
		'dN_dy_kaon': dN_dy_cents,
		'dN_dy_proton': dN_dy_cents,
		'dN_dy_Lambda': dN_dy_strange_cents,
		'dN_dy_Omega': dN_dy_strange_cents,
		'dN_dy_Xi': dN_dy_strange_cents,
		'mean_pT_pion': mean_pt_cents,
		'mean_pT_kaon': mean_pt_cents,
		'mean_pT_proton': mean_pt_cents,
		'pT_fluct': pT_fluct_cents,
		'v22': vn_cents,
		'v32': vn_cents,
		'v42': vn_cents
    },
}

obs_range_list={
    'Pb-Pb-2760': {
		'dNch_deta': [0,2000],
		'dET_deta': [0,2200],
		'dN_dy_pion': [0,1700],
		'dN_dy_kaon': [0,400],
		'dN_dy_proton': [0,120],
		'dN_dy_Lambda': [0,40],
		'dN_dy_Omega': [0,2],
		'dN_dy_Xi': [0,10],
		'mean_pT_pion': [0,1],
		'mean_pT_kaon': [0,1.5],
		'mean_pT_proton': [0,2],
		'pT_fluct': [0,0.05],
		'v22': [0,0.16],
		'v32': [0,0.1],
		'v42': [0,0.1]
    },
}

bayes_dtype=[    (sstr, 
                  [(obs, [("mean",float_t,len(cent_list)), ("err",float_t,len(cent_list))])\
                    for obs, cent_list in obs_cent_list[sstr].items() ],
                  number_of_models_per_run
                 ) \
                 for sstr in system_strs 
            ]

# The active ones used in Bayes analysis (MCMC)
active_obs_list = {
   sys: list(obs_cent_list[sys].keys()) for sys in system_strs
}

def zetas(T, zmax, width, T0, asym):
    DeltaT = T - T0
    sign = 1 if DeltaT>0 else -1
    x = DeltaT/(width*(1.+asym*sign))
    return zmax/(1.+x**2) 
zetas = np.vectorize(zetas)

