# Data structure used to save hadronic observables 
# (or quantities that can be used to compute hadronic observables)
# for each hydrodynamic event (oversamples or not)


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


