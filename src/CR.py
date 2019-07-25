#!/usr/bin/env python3
import logging
from configurations import *
import numpy as np
from bayes_mcmc import Chain
import pandas as pd
import matplotlib.pyplot as plt
#get model calculations at VALIDATION POINTS
logging.info("Load calculations from " + f_obs_validation)
Yexp_PseudoData = np.fromfile(f_obs_validation, dtype=bayes_dtype)


design_file = design_dir + \
       '/design_points_validation_{:s}{:s}-{:d}.dat'.format(*systems[0])
logging.info("Loading design points from " + design_file)
design = pd.read_csv(design_file)
design = design.drop("idx", axis=1)
truth = design.values[validation]

chain = Chain()
data = chain.load().T[:-1]
ndims, nsamples = data.shape

def zetas(T, zmax, width, T0, asym):
    DeltaT = T - T0
    sign = 1 if DeltaT>0 else -1
    x = DeltaT/(width*(1.+asym*sign))
    return zmax/(1.+x**2) 
zetas = np.vectorize(zetas)

with open("validate/{:d}.dat".format(validation),'w') as f:
    for i in range(ndims):
        samples = data[i]
        H, xbins = np.histogram(samples, bins=21)
        x = (xbins[1:] + xbins[:-1])/2.
        m = np.median(samples)
        M = x[np.argmax(H)]
        l1 = np.quantile(samples, .2)
        l2 = np.quantile(samples, .1)
        h1 = np.quantile(samples, .8)
        h2 = np.quantile(samples, .9)
        f.write("{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n".format(
                 truth[i], m, M, l1, l2, h1, h2
                 )
               )

    for Ttest in [.155, .175, .2, .25, .35]:
        width=2./np.pi*data[-2]**4/data[-4]
        samples = zetas(Ttest, data[-4], width, data[-3], data[-1])
        H, xbins = np.histogram(samples, bins=21)
        x = (xbins[1:] + xbins[:-1])/2.
        m = np.median(samples)
        M = x[np.argmax(H)]

        l1 = np.quantile(samples, .2)
        l2 = np.quantile(samples, .1)
        h1 = np.quantile(samples, .8)
        h2 = np.quantile(samples, .9)

        width=2./np.pi*truth[-2]**4/truth[-4]
        tt = zetas(Ttest, truth[-4], width, truth[-3], truth[-1])

        #plt.hist(samples,100, normed=True)
        #plt.plot([tt,tt],[0,1])
        #plt.show()

        f.write("{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\t{:1.6f}\n".format(
                 tt, m, M, l1, l2, h1, h2
                 )
               )

