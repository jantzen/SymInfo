import matplotlib.pyplot as plt
import numpy as np
from stochastic_systems import *
from symmetries import *
from estimate_symmetries import *
from sklearn.feature_selection import mutual_info_regression
from stat_methods import *


def modelTests():
    m = FujikawaLogistic(2.1, .69, 10**3, 10**(8.9))
    #m = FujikawaLogistic(1.37, .73, 10**4, 10**(8.85))
    domain = [i for i in range(1,200)]
    vals = m.check_xs(domain)
    plt.plot(domain, np.log10(vals))
    plt.xlabel("Time")
    plt.ylabel("Log X (CFU/ml)")
    plt.savefig('test1.png')
    m = McCarthyPVA(29.4, 29.4)

    domain = [i for i in range(1, 51)]#np.linspace(0, 1.05*100, 100)
    vals, surs, recs = m.mod_check_xs(domain)


    figure, axis = plt.subplots(2, 2)

    axis[0,1].plot(domain, vals)
    axis[0,1].set_xlabel("Time")
    axis[0,1].set_ylabel("Population")
    # axis[0,0].scatter(domain, surs, s=10)
    axis[0,0].scatter(vals, surs, s=10)
    coef = np.polyfit(np.array(vals).reshape(-1), np.array(surs).reshape(-1), 1)
    axis[0,0].axline(xy1=(0,coef[1]), slope=coef[0], color='black')
    axis[0,0].set_ylabel("Survivability")
    # axis[0,0].set_xlabel("Population")
    # axis[1,0].scatter(domain, recs, s=10)
    axis[1,0].scatter(vals, recs, s=10)
    coef = np.polyfit(np.array(vals).reshape(-1), np.array(recs).reshape(-1), 1)
    axis[1,0].axline(xy1=(0,coef[1]), slope=coef[0], color='black')
    # axis[1,0].set_xlabel("Population")
    axis[1,0].set_ylabel("Recruitment")

    plt.savefig('test2.png')


def FujikawaLogisticSymmetry():
    # generate data illustrating dynamics for system of type A and B
    init_x = 10**3
    FL_A = FujikawaLogistic(2.1, .69, 10**3, 10**(8.9))
    FL_B = FujikawaLogistic(2.1, .69, 10**3, 10**(8.9))
    # mPVA_B = FujikawaLogistic(1.37, .73, 10**4, 10**(8.85))
    curve_A = []
    curve_B = []
    t_max = 30
    num_times = 40
    times = np.linspace(0., t_max, 30)
    for ii in range(num_times):
        FL_A._x = copy.copy(init_x)
        FL_B._x = copy.copy(init_x)
        curve_A.append(FL_A.check_xs(times).reshape(-1,1))
        curve_B.append(FL_B.check_xs(times).reshape(-1,1))
    tmp = np.concatenate(curve_A, axis=1)    
    curve_A = np.concatenate([times.reshape(-1,1), tmp], axis=1)
    np.save('./exp1_output/Fujikawa/curve_A.npy', curve_A)
    tmp = np.concatenate(curve_B, axis=1)    
    curve_B = np.concatenate([times.reshape(-1,1), tmp], axis=1)
    np.save('./exp1_output/Fujikawa/curve_B.npy', curve_B)

    # Demonstrate discrimination from a single trial
    # data_A = FL_A.random_time_int_loop_func(0, 30, flsym)
    # data_B = FL_B.random_time_int_loop_func(0, 30, flsym)
    data_A = np.round(FL_A.random_time_intervention(t_max, flsym, num_times=num_times), 2)
    data_B = np.round(FL_B.random_time_intervention(t_max, flsym, num_times=num_times), 2)
    mi_A = mutual_info_regression(data_A[:,0].reshape(-1,1), data_A[:,1])
    mi_B = mutual_info_regression(data_B[:,0].reshape(-1,1), data_B[:,1])

    # Test A
    pval_A = test_for_independence(data_A[:,0], data_A[:,1])
    pval_B = test_for_independence(data_B[:,0], data_B[:,1])

    np.savetxt('./exp1_output/Fujikawa/mi_A', mi_A)
    np.savetxt('./exp1_output/Fujikawa/mi_B', mi_B)
    np.savetxt('./exp1_output/Fujikawa/pval_A', pval_A)
    np.savetxt('./exp1_output/Fujikawa/pval_B', pval_B)

    print(mi_A)
    print(mi_B)
    print(pval_A)
    print(pval_B)

def McCarthyPVASymmetry():
    # generate data illustrating dynamics for system of type A and B
    init_x = 29.4
    MC_A = McCarthyPVA(29.4, 29.4)
    MC_B = McCarthyPVA(29.4, 29.4)

    curve_A = []
    curve_B = []
    t_max = 50
    num_times = 40
    times = np.linspace(0., t_max, 50)
    for ii in range(num_times):
        MC_A._x = copy.copy(init_x)
        MC_B._x = copy.copy(init_x)
        curve_A.append(MC_A.check_xs(times).reshape(-1,1))
        curve_B.append(MC_B.check_xs(times).reshape(-1,1))
    tmp = np.concatenate(curve_A, axis=1)    
    curve_A = np.concatenate([times.reshape(-1,1), tmp], axis=1)
    np.save('./exp1_output/McCarthyPVASymmetry/curve_A.npy', curve_A)
    tmp = np.concatenate(curve_B, axis=1)    
    curve_B = np.concatenate([times.reshape(-1,1), tmp], axis=1)
    np.save('./exp1_output/McCarthyPVASymmetry/curve_B.npy', curve_B)

    # Demonstrate discrimination from a single trial
    # data_A = MC_A.random_time_int_loop_func(0, 30, flsym)
    # data_B = MC_B.random_time_int_loop_func(0, 30, flsym)
    data_A = np.round(MC_A.random_time_intervention(t_max, mcsym, num_times=num_times), 2)
    data_B = np.round(MC_B.random_time_intervention(t_max, mcsym, num_times=num_times), 2)
    mi_A = mutual_info_regression(data_A[:,0].reshape(-1,1), data_A[:,1])
    mi_B = mutual_info_regression(data_B[:,0].reshape(-1,1), data_B[:,1])

    # Test A
    pval_A = test_for_independence(data_A[:,0], data_A[:,1])
    pval_B = test_for_independence(data_B[:,0], data_B[:,1])

    np.savetxt('./exp1_output/McCarthyPVASymmetry/mi_A', mi_A)
    np.savetxt('./exp1_output/McCarthyPVASymmetry/mi_B', mi_B)
    np.savetxt('./exp1_output/McCarthyPVASymmetry/pval_A', pval_A)
    np.savetxt('./exp1_output/McCarthyPVASymmetry/pval_B', pval_B)

    print(mi_A)
    print(mi_B)
    print(pval_A)
    print(pval_B)


McCarthyPVASymmetry()