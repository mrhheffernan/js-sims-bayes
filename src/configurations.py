import numpy as np


# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'



#### >>>>>>>>>>>> configuration for design.py >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# collision systems
systems = [('Pb', 'Pb', 2760)]
system_strs = ['{:s}-{:s}-{:d}'.format(*s) for s in systems]

# number of design points
n_design_pts_main = 50
n_design_pts_validation = 0



#### >>>>>>>>>>>>>> configuration for calculations_average_obs.py >>>>>>>>>>>>>>>>>>>>>>
# 4 versions of delta-f with SMASH and a fifth model with UrQMD totals 5
number_of_models_per_run = 5

# number of minimum-bias events for each design points 
# (be consistent with submit file)
number_of_events_per_design = 100

# folder of ebe information at each design points for main and validation
f_events_main = '/home/yingru/Documents/Research/Important/4195354'
f_events_validation = None

# output file to save the average obs results
f_obs_main = 'main_4195354.dat'
f_obs_validation = None

# experimental/pre-defined centrality cuts for all observables in different experiments
obs_cent_list = {
'Pb-Pb-2760': {
    'dNch_deta' : np.array([[0,5],  [5,10], [10,20],[20,30],
                            [30,40],[40,50],[50,60]]), #8 bins
    'dET_deta' : np.array([[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10],
                           [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20],
                           [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30],
                           [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40],
                           [40, 45], [45, 50], [50, 55], [55, 60]]), # 22 bins
    'dN_dy_pion' : np.array([[0,5],[5,10],[10,20],[20,30],
                             [30,40],[40,50],[50,60]]), # 8 bins
    'dN_dy_kaon' : np.array([[0,5],[5,10],[10,20],[20,30],
                             [30,40],[40,50],[50,60]]), # 8 bins
    'dN_dy_proton' : np.array([[0,5],[5,10],[10,20],[20,30],
                             [30,40],[40,50],[50,60]]), # 8 bins
    'dN_dy_Lambda' : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'dN_dy_Omega' : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'dN_dy_Xi' : np.array([[0,10],[10,20],[20,40],[40,60]]), # 4 bins
    'mean_pT_pion' : np.array([[0,5],[5,10],[10,20],[20,30],
                             [30,40],[40,50],[50,60]]), # 8 bins
    'mean_pT_kaon' : np.array([[0,5],[5,10],[10,20],[20,30],
                             [30,40],[40,50],[50,60]]), # 8 bins
    'mean_pT_proton' : np.array([[0,5],[5,10],[10,20],[20,30],
                             [30,40],[40,50],[50,60]]), # 8 bins
    'pT_fluct' : np.array([[0,5],[5,10],[10,15],[15,20],
                          [20,25],[25,30],[30,35],[35,40],
                          [40,45],[45,50],[50,55],[55,60]]), #12 bins
    'v22' : np.array([[0,5],[5,10],[10,20],[20,30],
                      [30,40],[40,50],[50,60]]), # 8 bins
    'v32' : np.array([[0,5],[5,10],[10,20],[20,30],
                      [30,40],[40,50],[50,60]]), # 8 bins
    'v42' : np.array([[0,5],[5,10],[10,20],[20,30],
                      [30,40],[40,50],[50,60]]), # 8 bins
    },
}



