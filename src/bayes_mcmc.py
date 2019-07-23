#!/usr/bin/env python3
"""
Markov chain Monte Carlo model calibration using the `affine-invariant ensemble
sampler (emcee) <http://dfm.io/emcee>`_.

This module must be run explicitly to create the posterior distribution.
Run ``python -m src.mcmc --help`` for complete usage information.

On first run, the number of walkers and burn-in steps must be specified, e.g.
::

    python -m src.mcmc --nwalkers 500 --nburnsteps 100 200

would run 500 walkers for 100 burn-in steps followed by 200 production steps.
This will create the HDF5 file :file:`mcmc/chain.hdf` (default path).

On subsequent runs, the chain resumes from the last point and the number of
walkers is inferred from the chain, so only the number of production steps is
required, e.g. ::

    python -m src.mcmc 300

would run an additional 300 production steps (total of 500).

To restart the chain, delete (or rename) the chain HDF5 file.
"""

import argparse
from contextlib import contextmanager
import logging
import pandas as pd

import emcee
import h5py
import numpy as np
from scipy.linalg import lapack

import matplotlib.pyplot as plt
from configurations import *
from emulator import *

def mvn_loglike(y, cov):
    """
    Evaluate the multivariate-normal log-likelihood for difference vector `y`
    and covariance matrix `cov`:

        log_p = -1/2*[(y^T).(C^-1).y + log(det(C))] + const.

    The likelihood is NOT NORMALIZED, since this does not affect MCMC.  The
    normalization const = -n/2*log(2*pi), where n is the dimensionality.

    Arguments `y` and `cov` MUST be np.arrays with dtype == float64 and shapes
    (n) and (n, n), respectively.  These requirements are NOT CHECKED.

    The calculation follows algorithm 2.1 in Rasmussen and Williams (Gaussian
    Processes for Machine Learning).

    """
    # Compute the Cholesky decomposition of the covariance.
    # Use bare LAPACK function to avoid scipy.linalg wrapper overhead.
    L, info = lapack.dpotrf(cov, clean=False)

    if info < 0:
        raise ValueError(
            'lapack dpotrf error: '
            'the {}-th argument had an illegal value'.format(-info)
        )
    elif info < 0:
        raise np.linalg.LinAlgError(
            'lapack dpotrf error: '
            'the leading minor of order {} is not positive definite'
            .format(info)
        )

    # Solve for alpha = cov^-1.y using the Cholesky decomp.
    alpha, info = lapack.dpotrs(L, y)

    if info != 0:
        raise ValueError(
            'lapack dpotrs error: '
            'the {}-th argument had an illegal value'.format(-info)
        )

    return -.5*np.dot(y, alpha) - np.log(L.diagonal()).sum()


class LoggingEnsembleSampler(emcee.EnsembleSampler):
    def run_mcmc(self, X0, nsteps, status=None, **kwargs):
        """
        Run MCMC with logging every 'status' steps (default: approx 10% of
        nsteps).

        """
        logging.info('running %d walkers for %d steps', self.k, nsteps)

        if status is None:
            status = nsteps // 10

        for n, result in enumerate(
                self.sample(X0, iterations=nsteps, **kwargs),
                start=1
        ):
            if n % status == 0 or n == nsteps:
                af = self.acceptance_fraction
                logging.info(
                    'step %d: acceptance fraction: '
                    'mean %.4f, std %.4f, min %.4f, max %.4f',
                    n, af.mean(), af.std(), af.min(), af.max()
                )

        return result

def compute_cov(system, obs1, obs2, dy1, dy2, y1, y2):
    if obs1 == obs2:
        return np.diag(dy1)**2
    else:
        return np.zeros([dy1.size, dy2.size])

class Chain:
    """
    High-level interface for running MCMC calibration and accessing results.

    Currently all design parameters except for the normalizations are required
    to be the same at all beam energies.  It is assumed (NOT checked) that all
    system designs have the same parameters and ranges (except for the norms).

    """
    def __init__(self, validation=-1, path=workdir / 'mcmc' / 'chain.hdf'):
        self.path = path
        self.path.parent.mkdir(exist_ok=True)

        self._slices = {}
        self._expt_y = {}
        self._expt_cov = {}

        if validation < 0:
            Yexp = expdata
        else:
            #get model calculations at VALIDATION POINTS
            logging.info("Load calculations from " + f_obs_validation)
            model_data = np.fromfile(f_obs_validation, dtype=bayes_dtype)
            Yexp = model_data[validation]

        #get design range
        range_file = design_dir + \
               '/design_ranges_main_{:s}{:s}-{:d}.dat'.format(*systems[0])
        # range
        design_range = pd.read_csv(range_file)
        self.max = design_range['max'].values
        self.min = design_range['min'].values
        self.ndim = len(self.max)
        self.keys = design_range['param'].values
        self.labels = design_range['param'].values
        self.range = np.array([self.min, self.max]).T

        # load the dill'ed emulator from emulator file
        self.emulators = {}
        for s in system_strs:
            self.emulators[s] = dill.load(open('emulator/emu-' + s + '.dill', "rb"))

        logging.info("Pre-compute experimental covariance matrix")
        for s in system_strs:
            nobs = 0
            self._slices[s] = []
            for obs in active_obs_list[s]:
                try:
                    obsdata = Yexp[s][obs]['mean'][idf,:]
                except KeyError:
                    continue

                n = obsdata.size
                self._slices[s].append(
                        (obs, slice(nobs, nobs + n))
                  )
                nobs += n

            self._expt_y[s] = np.empty(nobs)
            self._expt_cov[s] = np.empty((nobs, nobs))

            for obs1, slc1 in self._slices[s]:
                self._expt_y[s][slc1] = Yexp[s][obs1]['mean'][idf,:]
                for obs2, slc2 in self._slices[s]:
                    self._expt_cov[s][slc1, slc2] = compute_cov(
                        s, obs1, obs2, 
                        Yexp[s][obs1]['err'][idf,:],
                        Yexp[s][obs2]['err'][idf,:],
                        Yexp[s][obs1]['mean'][idf,:],
                        Yexp[s][obs2]['mean'][idf,:]
                    )

    def _predict(self, X, **kwargs):
        """
        Call each system emulator to predict model output at X.

        """
        return { s: self.emulators[s].predict(X, **kwargs) \
                 for s in system_strs
               }

    def log_posterior(self, X):
        """
        Evaluate the posterior at `X`.

        `extra_std_prior_scale` is the scale parameter for the prior
        distribution on the model sys error parameter:

            prior ~ sigma^2 * exp(-sigma/scale)

        """
        X = np.array(X, copy=False, ndmin=2)

        lp = np.zeros(X.shape[0])

        inside = np.all((X > self.min) & (X < self.max), axis=1)
        lp[~inside] = -np.inf

        nsamples = np.count_nonzero(inside)

        if nsamples > 0:
            pred = self._predict(
                X[inside], return_cov=True
            )
            for sys in system_strs:
                nobs = self._expt_y[sys].size
                # allocate difference (model - expt) and covariance arrays
                dY = np.empty((nsamples, nobs))
                cov = np.empty((nsamples, nobs, nobs))

                Y_pred, cov_pred = pred[sys]

                # copy predictive mean and covariance into allocated arrays
                for obs1, slc1 in self._slices[sys]:
                    dY[:, slc1] = Y_pred[obs1] - self._expt_y[sys][slc1]
                    for obs2, slc2 in self._slices[sys]:
                        cov[:, slc1, slc2] = cov_pred[obs1, obs2]

                # add expt cov to model cov
                cov += self._expt_cov[sys]

                # compute log likelihood at each point
                lp[inside] += list(map(mvn_loglike, dY, cov))


        return lp

    def random_pos(self, n=1):
        """
        Generate `n` random positions in parameter space.

        """
        return np.random.uniform(self.min, self.max, (n, self.ndim))

    @staticmethod
    def map(f, args):
        """
        Dummy function so that this object can be used as a 'pool' for
        :meth:`emcee.EnsembleSampler`.

        """
        return f(args)

    def run_mcmc(self, nsteps, nburnsteps=None, nwalkers=None, status=None):
        """
        Run MCMC model calibration.  If the chain already exists, continue from
        the last point, otherwise burn-in and start the chain.

        """
        with self.open('a') as f:
            try:
                dset = f['chain']
            except KeyError:
                burn = True
                if nburnsteps is None or nwalkers is None:
                    logging.error(
                        'must specify nburnsteps and nwalkers to start chain'
                    )
                    return
                dset = f.create_dataset(
                    'chain', dtype='f8',
                    shape=(nwalkers, 0, self.ndim),
                    chunks=(nwalkers, 1, self.ndim),
                    maxshape=(nwalkers, None, self.ndim),
                    compression='lzf'
                )
            else:
                burn = False
                nwalkers = dset.shape[0]

            sampler = LoggingEnsembleSampler(
                nwalkers, self.ndim, self.log_posterior, pool=self
            )

            if burn:
                logging.info(
                    'no existing chain found, starting initial burn-in')
                # Run first half of burn-in starting from random positions.
                nburn0 = nburnsteps // 2
                sampler.run_mcmc(
                    self.random_pos(nwalkers),
                    nburn0,
                    status=status
                )
                logging.info('resampling walker positions')
                # Reposition walkers to the most likely points in the chain,
                # then run the second half of burn-in.  This significantly
                # accelerates burn-in and helps prevent stuck walkers.
                X0 = sampler.flatchain[
                    np.unique(
                        sampler.flatlnprobability,
                        return_index=True
                    )[1][-nwalkers:]
                ]
                sampler.reset()
                X0 = sampler.run_mcmc(
                    X0,
                    nburnsteps - nburn0,
                    status=status,
                    storechain=False
                )[0]
                sampler.reset()
                logging.info('burn-in complete, starting production')
            else:
                logging.info('restarting from last point of existing chain')
                X0 = dset[:, -1, :]

            sampler.run_mcmc(X0, nsteps, status=status)

            logging.info('writing chain to file')
            dset.resize(dset.shape[1] + nsteps, 1)
            dset[:, -nsteps:, :] = sampler.chain

    def open(self, mode='r'):
        """
        Return a handle to the chain HDF5 file.

        """
        return h5py.File(str(self.path), mode)

    @contextmanager
    def dataset(self, mode='r', name='chain'):
        """
        Context manager for quickly accessing a dataset in the chain HDF5 file.

        >>> with Chain().dataset() as dset:
                # do something with dset object

        """
        with self.open(mode) as f:
            yield f[name]

    def load(self, thin=1):
        """
        Read the chain from file.  If `keys` are given, read only those
        parameters.  Read only every `thin`'th sample from the chain.

        """
        ndim = self.ndim
        indices = slice(None)

        with self.dataset() as d:
            return np.array(d[:, ::thin, indices]).reshape(-1, ndim)

    def samples(self, n=1):
        """
        Predict model output at `n` parameter points randomly drawn from the
        chain.

        """
        with self.dataset() as d:
            X = np.array([
                d[i] for i in zip(*[
                    np.random.randint(s, size=n) for s in d.shape[:2]
                ])
            ])

        return self._predict(X)


def credible_interval(samples, ci=.9):
    """
    Compute the highest-posterior density (HPD) credible interval (default 90%)
    for an array of samples.

    """
    # number of intervals to compute
    nci = int((1 - ci)*samples.size)

    # find highest posterior density (HPD) credible interval
    # i.e. the one with minimum width
    argp = np.argpartition(samples, [nci, samples.size - nci])
    cil = np.sort(samples[argp[:nci]])   # interval lows
    cih = np.sort(samples[argp[-nci:]])  # interval highs
    ihpd = np.argmin(cih - cil)

    return cil[ihpd], cih[ihpd]


def main():
    parser = argparse.ArgumentParser(description='MCMC')

    parser.add_argument(
        'nsteps', type=int,
        help='number of steps'
    )
    parser.add_argument(
        '--nwalkers', type=int,
        help='number of walkers'
    )
    parser.add_argument(
        '--nburnsteps', type=int,
        help='number of burn-in steps'
    )
    parser.add_argument(
        '--status', type=int,
        help='number of steps between logging status'
    )
    parser.add_argument(
        '--validation', type=int, default=-1,
        help='the set of validation point to beused as PSEUDO-DATA, default(-1)'
    )

    args = parser.parse_args()
    Chain(
            validation=args.validation
          ).run_mcmc(
            nsteps=args.nsteps,
            nwalkers=args.nwalkers,
            nburnsteps=args.nburnsteps,
            status=args.status
          )


if __name__ == '__main__':
    main()
