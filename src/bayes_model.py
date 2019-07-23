#!/usr/bin/env python3

from configurations import *
import logging
import numpy as np

logging.info("Loading model calculations from " + f_obs_main)
model_data = np.fromfile(f_obs_main, dtype=bayes_dtype)

