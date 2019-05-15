# Data structure used to save hadronic observables 
# after average over 


# fully specify numeric data types, including endianness and size, to
# ensure consistency across all machines
float_t = '<f8'
int_t = '<i8'
complex_t = '<c16'

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

