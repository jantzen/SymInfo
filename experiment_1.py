# file: experiment_1.py

""" Symmetries known exactly from analysis of model.
"""

import numpy as np
from symmetries import *
from stochastic_systems import *
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import sys
import getopt
import pdb

def main(num_trials):
    r = np.array([1.5])
    k = np.array([500.])
    alpha = ([[1.]])
    sigma = np.array([0.1])
    gamma_A = 1.
    params_A = [r, k, alpha, sigma, gamma_A]
    gamma_B =  1.1
    params_B = [r, k, alpha, sigma, gamma_B]
    t_max = 2.
    num_times = 200
    init_x = np.array([5.])

    mi_A = []
    mi_B = []
    
    for ii in range(num_trials):
        data_A = np.round(random_time_intervention(init_x, params_A, t_max, lvsym,
                                          num_times=num_times), 2)
        data_B = np.round(random_time_intervention(init_x, params_B, t_max, lvsym,
                                          num_times=num_times), 2)
        mi_A.append(mutual_info_regression(data_A[:,0].reshape(-1,1),
                                           data_A[:,1]))
        mi_B.append(mutual_info_regression(data_B[:,0].reshape(-1,1),
                                           data_B[:,1]))

    mi_A = np.concatenate(mi_A)
    mi_B = np.concatenate(mi_B)

    mu_A = np.mean(mi_A)
    mu_B = np.mean(mi_B)

    std_A = np.std(mi_A)
    std_B = np.std(mi_B)
    test = ks_2samp(mi_A, mi_B)

    print(test)

    out = np.array([mu_A, std_A, mu_B, std_B, test[1]])

    np.savetxt('experiment1.txt', out)
    print("System A: mean = {}; std = {}".format(mu_A, std_A))
    print("System B: mean = {}; std = {}".format(mu_B, std_B))


if __name__ == "__main__":
    argv = sys.argv[1:]

    num_trials=100
    try:
        opts, args = getopt.getopt(argv, "n:")
    except:
        print("Option error.")
 
    for opt, arg in opts:
        if opt in ['-n']:
            num_trials = int(arg)
    main(num_trials)
