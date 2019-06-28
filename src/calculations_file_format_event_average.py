# Data structure used to save hadronic observables
# after average over

from configurations import *

from collections.abc import Iterable

# Center-of-mass energies
sqrts_list=['Au-Au-200','Pb-Pb-2760','Pb-Pb-5020']


dNch_deta_cents = [[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]] #8 bins
dET_deta_cent=[[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10], [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20], [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30], [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40], [40, 45], [45, 50], [50, 55], [55, 60], [60, 65], [65, 70]] # 22 bins
dN_dy_cents = [[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]] # 8 bins
dN_dy_strange_cents=[[0,10],[10,20],[20,40],[40,60]] # 4 bins
mean_pt_cents=[[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]] # 8 bins
pT_fluct_cents=[[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]] #12 bins
vn_cents=[[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]] # 8 bins

#TEMPORARY FOR SINGLE IC TESTS
"""
dNch_deta_cents = [[0,100]]
dET_deta_cent = [[0,100]]
dN_dy_cents = [[0,100]]
dN_dy_strange_cents = [[0,100]]
mean_pt_cents = [[0,100]]
pT_fluct_cents = [[0,100]]
vn_cents = [[0,100]]
"""
#TEMPORARY FOR SINGLE IC TESTS

#TEMPORARY

dNch_deta_cents = [[0,100],[50,100]]
dET_deta_cent = [[0,100],[50,100]]
dN_dy_cents = [[0,100],[50,100]]
dN_dy_strange_cents = [[0,100],[50,100]]
mean_pt_cents = [[0,100],[50,100]]
pT_fluct_cents = [[0,100],[50,100]]
vn_cents = [[0,100],[50,100]]

#TEMPORARY

# Observable name, data type, centralities
# totals 15 observable types
obs_cent_list={
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
}

tmp_dtype=[]

for obs, cent_list in obs_cent_list.items():

    tmp_dtype.append(
         ( obs,
            [
            ("mean",float_t,len(cent_list)),
            ("stat_err",float_t,len(cent_list))
            ],
        )
    )

bayes_dtype=[(sqrts, tmp_dtype, number_of_models_per_run) for sqrts in sqrts_list]

#bayes_dtype=[
#('Au-Au-200',
#	[
#		('dNch_deta', float_t, 8),
#		('dET_deta', float_t, 22),
#		('dN_dy', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
#		('dN_dy-s', [(n, float_t, 4) for n in ['Lambda', 'Omega','Xi']], 1),
#		('mean_pT', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
#		('pT_fluct', float_t, 12),
#		('v22', float_t, 8),
#		('v32', float_t, 6),
#		('v42', float_t, 6),
#		('v22-d', float_t, [1,10]),
#		('v32-d', float_t, [1,10]),
#		('v42-d', float_t, [1,10]),
#	], 5),
#('Pb-Pb-2760',
#	[
#		('dNch_deta', float_t, 8),
#		('dET_deta', float_t, 22),
#		('dN_dy', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
#		('dN_dy-s', [(n, float_t, 4) for n in ['Lambda', 'Omega','Xi']], 1),
#		('mean_pT', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
#		('pT_fluct', float_t, 12),
#		('v22', float_t, 8),
#		('v32', float_t, 6),
#		('v42', float_t, 6),
#		('v22-d', float_t, [1,10]),
#		('v32-d', float_t, [1,10]),
#		('v42-d', float_t, [1,10]),
#	], 5),
#('Pb-Pb-5020',
#	[
#		('dNch_deta', float_t, 8),
#		('dET_deta', float_t, 22),
#		('dN_dy', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
#		('dN_dy-s', [(n, float_t, 4) for n in ['Lambda', 'Omega','Xi']], 1),
#		('mean_pT', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
#		('pT_fluct', float_t, 12),
#		('v22', float_t, 8),
#		('v32', float_t, 6),
#		('v42', float_t, 6),
#		('v22-d', float_t, [1,10]),
#		('v32-d', float_t, [1,10]),
#		('v42-d', float_t, [1,10]),
#	], 5)
#]
