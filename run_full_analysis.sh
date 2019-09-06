#remove existing mcmc chains
rm mcmc/chain.hdf

#average the observables by centrality
#./src/calculations_average_obs.py

#train the emulator
./src/emulator.py --retrain --npc 6 --nrestarts 4

#perform MCMC
./src/bayes_mcmc.py 4000 --nwalkers 100 --nburnsteps 500

#generating plots of posteriors 
./src/bayes_plot.py plots/diag_posterior.png
./src/bayes_plot.py plots/viscous_posterior.png
./src/bayes_plot.py plots/obs_validation.png

open plots/diag_posterior.png
open plots/viscous_posterior.png
open plots/obs_validation.png
