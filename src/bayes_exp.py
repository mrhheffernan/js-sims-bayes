#!/usr/bin/env python3
import logging
from configurations import *
import numpy as np

#get model calculations at VALIDATION POINTS
logging.info("Load calculations from " + f_obs_validation)
Yexp_PseudoData = np.fromfile(f_obs_validation, dtype=bayes_dtype)

