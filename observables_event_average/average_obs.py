#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys, os, glob
# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'

# species (name, ID) for identified particle observables
species = [
	('pion', 211),
	('kaon', 321),
	('proton', 2212),
	('Lambda', 3122),
	('Sigma0', 3212),
	('Xi', 3312),
	('Omega', 3334),
	('phi', 333),
]
pi_K_p = [
	('pion', 211),
	('kaon', 321),
	('proton', 2212),
]
lambda_omega_xi = [
	('Lambda', 3122),
	('Omega', 3334),
	('Xi', 3312),
]

NpT = 10
Nharmonic = 8
Nharmonic_diff = 5

# results "array" (one element)
# to be overwritten for each event
result_dtype=[
('initial_entropy', float_t, 1),
('impact_parameter', float_t, 1),
('npart', float_t, 1),
('ALICE',
	[
		# 1) dNch/deta, eta[-0.5, 0.5], charged
		('dNch_deta', float_t, 1),
		# 2) dET/deta, eta[-0.6, 0.6]
		('dET_deta', float_t, 1),
		# 3.1) The Tmunu observables, eta[-0.6, 0.6]
		('Tmunu', float_t, 10),
		# 3.2) The Tmunu observables, eta[-0.5, 0.5], charged
		('Tmunu_chg', float_t, 10),
		# 4.1) identified particle yield
		('dN_dy', 	[(name, float_t, 1) for (name,_) in species], 1),
		# 4.2) identified particle <pT>
		('mean_pT', [(name, float_t, 1) for (name,_) in species], 1),
		# 5.1) pT fluct, pT[0.15, 2], eta[-0.8, 0.8], charged
		('pT_fluct_chg', [	('N', int_t, 1),
							('sum_pT', float_t, 1),
							('sum_pT2', float_t, 1)], 1),
		# 5.2) pT fluct, pT[0.15, 2], eta[-0.8, 0.8], pi, K, p
		('pT_fluct_pid', [	(name, [	('N', int_t, 1),
										('sum_pT', float_t, 1),
										('sum_pT2', float_t, 1)], 1	)
							  for (name,_) in pi_K_p	], 1),
		# 6) Q vector, pT[0.2, 5.0], eta [-0.8, 0.8], charged
		('flow', [	('N', int_t, 1),
					('Qn', complex_t, Nharmonic)], 1),
		# 7) Q vector, diff-flow eta[-0.8, 0.8], pi, K, p
		# It uses #6 as its reference Q vector
		('d_flow_chg', [('N', int_t, NpT),
						('Qn', complex_t, [NpT, Nharmonic_diff])], 1),
		('d_flow_pid', [(name, [('N', int_t, NpT),
								('Qn', complex_t, [NpT, Nharmonic_diff])], 1)
						for (name,_) in pi_K_p	], 1),
	], 5)
]

bayes_dtype=[
('Au-Au-200',
	[
		('dNch_deta', float_t, 8),
		('dET_deta', float_t, 22),
		('dN_dy', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
		('dN_dy-s', [(n, float_t, 4) for n in ['Lambda', 'Omega','Xi']], 1),
		('mean_pT', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
		('pT_fluct', float_t, 12),
		('v22', float_t, 8),
		('v32', float_t, 6),
		('v42', float_t, 6),
		('v22-d', float_t, [1,10]),
		('v32-d', float_t, [1,10]),
		('v42-d', float_t, [1,10]),
	], 5),
('Pb-Pb-2760',
	[
		('dNch_deta', float_t, 8),
		('dET_deta', float_t, 22),
		('dN_dy', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
		('dN_dy-s', [(n, float_t, 4) for n in ['Lambda', 'Omega','Xi']], 1),
		('mean_pT', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
		('pT_fluct', float_t, 12),
		('v22', float_t, 8),
		('v32', float_t, 6),
		('v42', float_t, 6),
		('v22-d', float_t, [1,10]),
		('v32-d', float_t, [1,10]),
		('v42-d', float_t, [1,10]),
	], 5),
('Pb-Pb-5020',
	[
		('dNch_deta', float_t, 8),
		('dET_deta', float_t, 22),
		('dN_dy', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
		('dN_dy-s', [(n, float_t, 4) for n in ['Lambda', 'Omega','Xi']], 1),
		('mean_pT', [(n, float_t, 8) for n in ['pion','kaon','proton']], 1),
		('pT_fluct', float_t, 12),
		('v22', float_t, 8),
		('v32', float_t, 6),
		('v42', float_t, 6),
		('v22-d', float_t, [1,10]),
		('v32-d', float_t, [1,10]),
		('v42-d', float_t, [1,10]),
	], 5)
]

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
		#print("cn2 = ")
		#print(cn2)
		avg_cn2, std_avg_cn2 = weighted_mean_std(cn2, w)
		vn = np.sqrt(avg_cn2)
		vn_err = std_avg_cn2/2./vn
		return vn, vn_err
	Ne = len(ds)
	cenM = np.mean(cen, axis=1)
	index = (cen/100.*Ne).astype(int)

	obs = np.zeros([len(cenM), Nharmonic])
	obs_err = np.zeros([len(cenM), Nharmonic])

	#print("Inside calculate_vn")
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

def load_and_compute(inputfile, plot=False):
	entry = np.zeros(1, dtype=bayes_dtype)

	res_unsort = np.fromfile(inputfile, dtype=result_dtype)

	for idf in range(0,5):
		res = np.array(sorted(res_unsort, key=lambda x: x['ALICE'][idf]['dNch_deta'], reverse=True))
		print("Result size is " + str(res.size))
		print("Number of events with no charged particles : " + str( (res_unsort['ALICE']['dNch_deta'][:,3]==0).sum() ) )
		# dNdeta
		cenb = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])
		info = calculate_dNdeta(res, 'ALICE', cenb, idf)
		entry['Pb-Pb-2760']['dNch_deta'][:, idf] = info['obs']
		if plot:
			plt.subplot(2,4,1)
			#plt.errorbar(info['cenM'], info['obs'], yerr=info['err'], fmt='ro-')
			plt.errorbar(info['cenM'], info['obs'], yerr=info['err'])
			plt.ylim(0,2000)
			plt.xlabel(r'Centrality (%)', fontsize=7)
			#plt.xlim(60,70)
			#plt.ylim(65,73)
			plt.ylabel(r'charged $dN/d\eta$', fontsize=7)

		# dETdeta

		cenb = np.array([[0, 2.5], [2.5, 5], [5, 7.5], [7.5, 10], [10, 12.5], [12.5, 15], [15, 17.5], [17.5, 20], [20, 22.5], [22.5, 25], [25, 27.5], [27.5, 30], [30, 32.5], [32.5, 35], [35, 37.5], [37.5, 40], [40, 45], [45, 50], [50, 55], [55, 60], [60, 65], [65, 70]])
		info = calculate_dETdeta(res, 'ALICE', cenb, idf)
		entry['Pb-Pb-2760']['dET_deta'][:,idf] = info['obs']
		if plot:
			plt.subplot(2,4,2)
			#plt.errorbar(info['cenM'], info['obs'], yerr=info['err'], fmt='bo-')
			plt.errorbar(info['cenM'], info['obs'], yerr=info['err'])
			plt.ylim(0,2000)
			plt.xlabel(r'Centrality (%)', fontsize=7)
			plt.ylabel(r'$dE_T/d\eta$', fontsize=7)

		# dN(pid)/dy
		cenb = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])
		info = calculate_dNdy(res, 'ALICE', cenb, idf)
		for (s, _) in pi_K_p:
			entry['Pb-Pb-2760']['dN_dy'][s][:,idf] = info['obs'][s]
		if plot:
			plt.subplot(2,4,3)
			for (s, _) in pi_K_p:
				#plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], fmt='o-', label=s)
				plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], label=s)
			plt.xlabel(r'Centrality (%)', fontsize=7)
			plt.ylabel(r'$dN/dy$', fontsize=7)
			plt.legend()

		# dN(pid)/dy exotic

		cenb = np.array([[0,10],[10,20],[20,40],[40,60]])
		info = calculate_dNdy(res, 'ALICE', cenb, idf)
		for s in ['Lambda', 'Omega','Xi']:
			entry['Pb-Pb-2760']['dN_dy-s'][s][:,idf] = info['obs'][s]
		if plot:
			plt.subplot(2,4,4)
			for (s, _) in lambda_omega_xi:
				#plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], fmt='o-', label=s)
				plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], label=s)
			plt.xlabel(r'Centrality (%)', fontsize=7)
			plt.ylabel(r'$dN/dy$', fontsize=7)
			plt.legend()


		# mean-pT
		cenb = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])
		info = calculate_mean_pT(res, 'ALICE', cenb, idf)
		for (s, _) in pi_K_p:
			entry['Pb-Pb-2760']['mean_pT'][s][:,idf] = info['obs'][s]
		if plot:
			plt.subplot(2,4,5)
			for (s, _) in pi_K_p:
				#plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], fmt='o-', label=s)
				plt.errorbar(info['cenM'], info['obs'][s], yerr=info['err'][s], label=s)
			plt.ylim(0,2)
			plt.xlabel(r'Centrality (%)', fontsize=7)
			plt.ylabel(r'$\langle p_T\rangle$', fontsize=7)
			plt.legend()

		# mean-pT-fluct
		cenb = np.array([[0,5],[5,10],[10,15],[15,20],[20,25],[25,30],[30,35],[35,40],[40,45],[45,50],[50,55],[55,60]])
		info = calculate_mean_pT_fluct(res, 'ALICE', cenb, idf)
		entry['Pb-Pb-2760']['pT_fluct'][:,idf] = info['obs']
		if plot:
			plt.subplot(2,4,6)
			#plt.errorbar(info['cenM'], info['obs'], yerr=info['err'], fmt='o-')
			plt.errorbar(info['cenM'], info['obs'], yerr=info['err'])
			plt.xlabel(r'Centrality (%)', fontsize=7)
			plt.ylabel(r'$\delta p_T/p_T$', fontsize=7)
			#plt.legend()

		# vn
		cenb = np.array([[0,5],[5,10],[10,20],[20,30],[30,40],[40,50],[50,60],[60,70]])
		#cenb = np.array([[0,10],[20,30],[30,40],[40,50],[50,60],[60,70],[70,80]])
		info = calculate_vn(res, 'ALICE', cenb, idf)
		entry['Pb-Pb-2760']['v22'][:,idf] = info['obs'][:, 1]
		entry['Pb-Pb-2760']['v32'][:,idf]  = info['obs'][:6, 2]
		entry['Pb-Pb-2760']['v42'][:,idf]  = info['obs'][:6, 3]
		if plot:
			plt.subplot(2,4,7)
			for n, c in zip([1,2,3],'rgb'):
				#plt.errorbar(info['cenM'], info['obs'][:,n], yerr=info['err'][:,n], fmt=c+'o')
				plt.errorbar(info['cenM'], info['obs'][:,n], yerr=info['err'][:,n], label=str(n+1))
			plt.xlabel(r'Centrality (%)', fontsize=7)
			plt.ylabel(r'$v_n$', fontsize=7)
			plt.ylim(0,.15)
			plt.legend()

		# vn-diff
		"""
		cenb = np.array([[30,40]])
		pTbins = np.array([[0,.2], [.2,.4], [.4,.6],[.6,.8],[.8,1.],
				[1.,1.2], [1.2,1.5], [1.5,2.], [2.,2.5], [2.5,3]])
		info = calculate_diff_vn(res, 'ALICE', cenb, pTbins, idf, pid='chg')
		entry['Pb-Pb-2760']['v22-d'][:,idf] = info['obs'][:, :, 1]
		entry['Pb-Pb-2760']['v32-d'][:,idf]  = info['obs'][:, :, 2]
		entry['Pb-Pb-2760']['v42-d'][:,idf]  = info['obs'][:, :, 3]
		if plot:
			plt.subplot(2,4,7)
			for n in np.arange(1,4):
				plt.errorbar(info['pTM'], info['obs'][0,:,n], yerr=info['err'][0,:,n], fmt='o-', label=r"$v_{:d}$".format(n+1))
			plt.ylim(0,.3)
			plt.xlabel(r'$p_T$ [GeV]', fontsize=7)
			plt.ylabel(r'charged $v_{:d}$'.format(n+1), fontsize=7)
			plt.legend()
		"""


		if plot:
			plt.tight_layout(True)
			plt.show()


	return entry

if __name__ == '__main__':
	results = []
	for file in glob.glob(sys.argv[1]):
		fig = plt.figure(figsize=(8,4))
		entry = load_and_compute(file, plot=True)
		results.append(entry[0])
	results = np.array(results, dtype=bayes_dtype)
	#print(results.shape)
	results.tofile("obs.dat")
