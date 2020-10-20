
import numpy as np
import os
from tools import *


def run_stochastic(dataset, name, R, c_0=3.0, L_0=1.0, x_axe_threshold=1000,  
                   max_time=100, timestamps=[20, 20, 20, 20], show_legend=True,
                   inner_eps=1e-7, resid_eps=1e-6, M=1):
    
    print('STOCHASTIC METHODS: \t %s, \t file: %s, \t R = %f.' % 
          (name, dataset, R))
    output_name = 'output/%s_%d' % (name, R)
    cmd = ('./main --dataset=%s --experiment=stochastic ' +
           '--logging_time_period=1.0 --n_iters=100000000 ' + 
           '--output_name=%s --R=%d --max_time=%d --c_0=%g --L_0=%g ' +
           '--inner_eps=%g ') % \
           (dataset, output_name, R, max_time, c_0, L_0, inner_eps)
    
    print(('COMMAND: "%s"' % cmd), flush=True)
    
    os.system(cmd)

    print('PLOT RESULTS ... ', end=' ', flush=True)
    (iter_sgd, secs_sgd, func_sgd, data_sgd) = \
        read_dump(output_name + '_SGD.csv')
    (iter_svrg, secs_svrg, func_svrg, data_svrg) = \
        read_dump(output_name + '_SVRG.csv')
    (iter_snewton, secs_snewton, func_snewton, data_snewton) = \
        read_dump(output_name + '_Stoch_Contr_Newton.csv')
    (iter_svrnewton, secs_svrnewton, func_svrnewton, data_svrnewton) = \
        read_dump(output_name + '_Stoch_VR_Contr_Newton.csv')

    plot_results(
        [iter_sgd, iter_svrg, iter_snewton, iter_svrnewton], 
        [secs_sgd, secs_svrg, secs_snewton, secs_svrnewton],
        [func_sgd, func_svrg, func_snewton, func_svrnewton],
        [data_sgd, data_svrg, data_snewton, data_svrnewton],
        ['SGD', 'SVRG', 'SNewton', 'SVRNewton'],
        ['blue', 'tab:blue', 'red', 'tab:orange'],
        [':', '-', '-.', '-', ':', ':'],
        [3, 1, 3, 4],
        [0.8, 0.8, 1, 1],
        ('%s, D = %d' % (name, 2 * R)),
        x_axe_threshold=x_axe_threshold,
        timestamps=timestamps,
        filename=output_name + '.pdf',
        show_legend=show_legend,
        resid_eps=resid_eps,
        use_data_accesses=True,
        M=M
        )

    print('DONE.')
    print(('=' * 80), flush=True)


run_stochastic('data/covtype_bin_sc', 'covtype', R=10, c_0=3.0, L_0=100.0,
               x_axe_threshold=200, max_time=30,
               timestamps=[[20], [20], [20], [20]], 
               show_legend=True, M=581012)

run_stochastic('data/covtype_bin_sc', 'covtype', R=50, c_0=3.0, L_0=100.0,
               x_axe_threshold=200, max_time=30,
               timestamps=[[20], [20], [20], [20]], 
               show_legend=False, M=581012)

run_stochastic('data/covtype_bin_sc', 'covtype', R=250, c_0=3.0, L_0=100.0,
               x_axe_threshold=200, max_time=30,
               timestamps=[[20], [20], [20], [20]], 
               show_legend=False, M=581012)

# Extra Experiments:
"""
run_stochastic('data/YearPredictionMSD', 'YearPredictionMSD', R=10, 
               c_0=0.01, L_0=10000000.0,
               x_axe_threshold=200, max_time=100,
               timestamps=[[20], [20], [20], [20]], 
               show_legend=True, M=463715)

run_stochastic('data/YearPredictionMSD', 'YearPredictionMSD', R=50, 
               c_0=0.001, L_0=100000000.0,
               x_axe_threshold=200, max_time=100,
               timestamps=[[20], [20], [20], [20]], 
               show_legend=True, M=463715)

run_stochastic('data/YearPredictionMSD', 'YearPredictionMSD', R=250, 
               c_0=0.001, L_0=100000000.0,
               x_axe_threshold=200, max_time=100,
               timestamps=[[20], [20], [20], [20]], 
               show_legend=True, M=463715)

run_stochastic('data/mnist', 'mnist', R=10, c_0=0.01, L_0=1000000.0,
               x_axe_threshold=600, max_time=200,
               timestamps=[[50], [50], [50], [70]], 
               show_legend=True, M=60000, inner_eps=1e-5)

run_stochastic('data/mnist', 'mnist', R=50, c_0=0.01, L_0=1000000.0,
               x_axe_threshold=600, max_time=200,
               timestamps=[[30], [50], [50], [150]], 
               show_legend=False, M=60000, inner_eps=1e-5)

run_stochastic('data/mnist', 'mnist', R=250, c_0=0.01, L_0=1000000.0,
               x_axe_threshold=600, max_time=200,
               timestamps=[[50], [50], [50], [50]], 
               show_legend=False, M=60000, inner_eps=1e-5)

run_stochastic('data/higgs2m.txt', 'HIGGS2m', R=10, 
               c_0=3.0, L_0=10000.0,
               x_axe_threshold=200, max_time=100,
               timestamps=[[50], [50], [50], [50]], 
               show_legend=True, M=2000000)

run_stochastic('data/higgs2m.txt', 'HIGGS2m', R=50, 
               c_0=0.1, L_0=10000.0,
               x_axe_threshold=200, max_time=100,
               timestamps=[[50], [50], [40], [50]], 
               show_legend=False, M=2000000)

run_stochastic('data/higgs2m.txt', 'HIGGS2m', R=250, 
               c_0=0.1, L_0=10000.0,
               x_axe_threshold=200, max_time=100,
               timestamps=[[50], [50], [50], [50]], 
               show_legend=False, M=2000000)
"""

