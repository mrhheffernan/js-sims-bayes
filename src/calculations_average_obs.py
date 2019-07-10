#!/usr/bin/env python3

import numpy as np
import h5py
import sys, os, glob
# Input data format
from calculations_file_format_single_event import *
# Output data format
from calculations_file_format_event_average import *

def list2array(func):
        def func_wrapper(x, w):
                try:
                        x = np.array(x)
                        w = np.array(w)
                except:
                        raise ValueError("cannot interpret input as numpy array...")
                return func(x, w)
        return func_wrapper

def weighted_mean_std(x, w=None):
        if w is None:
                Neff = x.size
                mean = np.mean(x)
                std = np.std(x)/np.sqrt(Neff-1.+1e-9)
        else:
                Neff = np.sum(w)**2/np.sum(w**2)
                mean = np.average(x, weights=w)
                std = ( np.average((x-mean)**2, weights=w)/(Neff-1.+1e-9) ) **.5
        return mean, std

def calculate_dNdeta(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros_like(cenM)
        obs_err = np.zeros_like(cenM)
        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                obs[i], obs_err[i] = weighted_mean_std( ds[exp]['dNch_deta'][nl:nh, idf] )
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}


def calculate_dETdeta(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros_like(cenM)
        obs_err = np.zeros_like(cenM)
        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                obs[i], obs_err[i] = weighted_mean_std(ds[exp]['dET_deta'][nl:nh, idf])
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_dNdy(ds, exp, cen, idf):
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = {s: np.zeros_like(cenM) for (s, _) in species}
        obs_err = {s: np.zeros_like(cenM) for (s, _) in species}
        for (s, _) in species:
                for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                        obs[s][i], obs_err[s][i] = weighted_mean_std(ds[exp]['dN_dy'][s][nl:nh, idf])
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_mean_pT(ds, exp, cen, idf):
        #print("Calculating mean pT")
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = {s: np.zeros_like(cenM) for (s, _) in species}
        obs_err = {s: np.zeros_like(cenM) for (s, _) in species}
        for (s, _) in species:
                for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                        obs[s][i], obs_err[s][i] = weighted_mean_std(ds[exp]['mean_pT'][s][nl:nh, idf])
        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_mean_pT_fluct(ds, exp, cen, idf):

        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)
        obs = np.zeros_like(cenM)
        obs_err = np.zeros_like(cenM)
        for (s, _) in species:

                for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                        N = ds[exp]['pT_fluct_chg']['N'][nl:nh, idf]
                        sum_pT = ds[exp]['pT_fluct_chg']['sum_pT'][nl:nh, idf]
                        sum_pTsq = ds[exp]['pT_fluct_chg']['sum_pT2'][nl:nh, idf]


                        Npairs = .5*N*(N - 1)
                        #print("Npairs = " + str(Npairs))
                        M = sum_pT.sum() / N.sum()

                        # This is equivalent to the sum over pairs in Eq. (2).  It may be derived
                        # by using that, in general,
                        #
                        #   \sum_{i,j>i} a_i a_j = 1/2 [(\sum_{i} a_i)^2 - \sum_{i} a_i^2].
                        #
                        # That is, the sum over pairs (a_i, a_j) may be re-expressed in terms of
                        # the sum of a_i and sum of squares a_i^2.  Applying this to Eq. (2) and
                        # collecting terms yields the following expression.
                        C = (
                                .5*(sum_pT**2 - sum_pTsq) - M*(N - 1)*sum_pT + M**2*Npairs
                        ).sum() / Npairs.sum()

                        obs[i] = np.sqrt(C)/M
                        obs_err[i] = obs[i]*0.

        return {'Name': 'dNch_deta', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}


def calculate_vn(ds, exp, cen, idf):
        @list2array
        def obs_and_err(qn, m):
                w = m*(m-1.) # is this P_{M,2} in notation of Jonah's Thesis
                cn2 = (np.abs(qn)**2 - m)/w # is this is <2> in Jonah's thesis (p.27)
                avg_cn2, std_avg_cn2 = weighted_mean_std(cn2, w)
                vn = np.sqrt(avg_cn2)
                vn_err = std_avg_cn2/2./vn
                return vn, vn_err
        Ne = len(ds)
        cenM = np.mean(cen, axis=1)
        index = (cen/100.*Ne).astype(int)

        obs = np.zeros([len(cenM), Nharmonic])
        obs_err = np.zeros([len(cenM), Nharmonic])

        for i, (nl, nh) in enumerate(zip(index[:,0], index[:,1])):
                M = ds[exp]['flow']['N'][nl:nh, idf]+1e-10
                for n in range(Nharmonic):
                        Q = ds[exp]['flow']['Qn'][nl:nh, idf, n]
                        obs[i,n], obs_err[i,n] = obs_and_err(Q, M)
        return {'Name': 'vn', 'cenM': cenM, 'pTM' : None,
                        'obs': obs, 'err': obs_err}

def calculate_diff_vn(ds, exp, cenbins, pTbins, idf, pid='chg'):
        Ne = len(ds)
        pTbins = np.array(pTbins)
        cenbins = np.array(cenbins)
        cenM = np.mean(cenbins, axis=1)
        pTM = np.mean(pTbins, axis=1)
        Cindex = (cenbins/100.*Ne).astype(int)

        if pid == 'chg':
                obs = 'd_flow_chg'
                data = ds[exp][:,idf][obs]
        else:
                obs = 'd_flow_pid'
                data = ds[exp][:,idf][obs][s]

        # need soft flow within the same centrality bin first
        # only needs Ncen x [v2, v3]
        vnref = calculate_vn(ds, exp, cenbins, idf)

        # calculate hard vn
        vn = np.zeros([len(cenM), len(pTM), Nharmonic_diff])
        vn_err = np.zeros([len(cenM), len(pTM), Nharmonic_diff])
        for i, (nl, nh) in enumerate(Cindex):
                for j, (pl, ph) in enumerate(pTbins):
                        for n in range(Nharmonic_diff):
                                w = data['N'][nl:nh, j] * ds[exp]['flow']['N'][nl:nh, idf]
                                dn2 = (data['Qn'][nl:nh,j,n].conjugate() * ds[exp]['flow']['Qn'][nl:nh, idf, n]).real / w
                                avg_dn2, std_avg_dn2 = weighted_mean_std(dn2, w)
                                vn[i, j, n] = avg_dn2/vnref['obs'][i,n]
                                vn_err[i, j, n] = std_avg_dn2/vnref['obs'][i,n]
        return {'Name': 'vn2', 'cenM': cenM, 'pTM' : pTM,
                        'obs': vn, 'err': vn_err}

def load_and_compute(inputfile):
    entry = np.zeros(1, dtype=np.dtype(bayes_dtype))

    res_unsort = np.fromfile(inputfile, dtype=result_dtype)

    for idf in [0,3]:
        print("Computing observables for idf = " + str(idf) )
        res = np.array(sorted(res_unsort, key=lambda x: x['ALICE'][idf]['dNch_deta'], reverse=True))

        print("Result size is " + str(res.size))
        print("Number of events with no charged particles : " + str( (res_unsort['ALICE']['dNch_deta'][:, idf] < 0.1).sum() ) )

        # dNdeta
        tmp_obs='dNch_deta'
        cenb=np.array(obs_cent_list[tmp_obs])
        info = calculate_dNdeta(res, 'ALICE', cenb, idf)
        entry['Pb-Pb-2760'][tmp_obs]['mean'][:, idf] = info['obs']
        entry['Pb-Pb-2760'][tmp_obs]['stat_err'][:,idf] = info['err']

        # dETdeta
        tmp_obs='dET_deta'
        cenb=np.array(obs_cent_list[tmp_obs])
        info = calculate_dETdeta(res, 'ALICE', cenb, idf)
        entry['Pb-Pb-2760'][tmp_obs]['mean'][:,idf] = info['obs']
        entry['Pb-Pb-2760'][tmp_obs]['stat_err'][:,idf] = info['err']

        # dN(pid)/dy
        for s in ['pion','kaon','proton','Lambda', 'Omega','Xi']:
            cenb=np.array(obs_cent_list['dN_dy_'+s])
            info = calculate_dNdy(res, 'ALICE', cenb, idf)
            entry['Pb-Pb-2760']['dN_dy_'+s]['mean'][:,idf] = info['obs'][s]
            entry['Pb-Pb-2760']['dN_dy_'+s]['stat_err'][:,idf] = info['err'][s]


        # mean-pT
        for s in ['pion','kaon','proton']:
            cenb=np.array(obs_cent_list['mean_pT_'+s])
            info = calculate_mean_pT(res, 'ALICE', cenb, idf)
            entry['Pb-Pb-2760']['mean_pT_'+s]['mean'][:,idf] = info['obs'][s]
            entry['Pb-Pb-2760']['dN_dy_'+s]['stat_err'][:,idf] = info['err'][s]

        # mean-pT-fluct
        
        tmp_obs='pT_fluct'
        cenb=np.array(obs_cent_list[tmp_obs])
        info = calculate_mean_pT_fluct(res, 'ALICE', cenb, idf)
        entry['Pb-Pb-2760'][tmp_obs]['mean'][:,idf] = info['obs']
        entry['Pb-Pb-2760'][tmp_obs]['stat_err'][:,idf] = info['err']
        
        # vn
        for n in range(2,5):
            tmp_obs='v'+str(n)+'2'
            cenb=np.array(obs_cent_list[tmp_obs])
            info = calculate_vn(res, 'ALICE', cenb, idf)
            entry['Pb-Pb-2760'][tmp_obs]['mean'][:,idf] = info['obs'][:, n-1]
            entry['Pb-Pb-2760'][tmp_obs]['stat_err'][:,idf] = info['err'][:, n-1]


    return entry

if __name__ == '__main__':
    results = []
    res_dir = sys.argv[1]
    for i in range(100):
        filename = res_dir+"/{:d}.dat".format(i)
        print("-------working on " + filename+"-----------")
        entry = load_and_compute(filename)
        results.append(entry[0])
    results = np.array(results, dtype=np.dtype(bayes_dtype))
    results.tofile("obs.dat")
