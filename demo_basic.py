
import numpy as np
import os
from tools import *


def run_basic(dataset, name, R, x_axe_threshold=1000, max_time=20,
              timestamps=[5, 5, 5, 5, 5], show_legend=True,
              inner_eps=1e-9, resid_eps=1e-7):
    print('BASIC METHODS: \t %s, \t file: %s, \t R = %f.' % (name, dataset, R))
    output_name = 'output/%s_%d' % (name, R)
    cmd = ('./main --dataset=%s --experiment=basic ' +
           '--logging_time_period=0.1 --n_iters=10000 ' + 
           '--output_name=%s --R=%d --max_time=%d --c_0=3.0 --L_0=1.0 ' +
           '--inner_eps=%g ') % \
           (dataset, output_name, R, max_time, inner_eps)
    
    print(('COMMAND: "%s"' % cmd), flush=True)
    
    os.system(cmd)

    print('PLOT RESULTS ... ', end=' ', flush=True)
    (iter_gm, secs_gm, func_gm, data_gm) = \
        read_dump(output_name + '_GM.csv')
    (iter_fgm, secs_fgm, func_fgm, data_fgm) = \
        read_dump(output_name + '_FGM.csv')
    (iter_fw, secs_fw, func_fw, data_fw) = \
        read_dump(output_name + '_FW.csv')
    (iter_cnewton, secs_cnewton, func_cnewton, data_cnewton) = \
        read_dump(output_name + '_Contr_Newton.csv')
    (iter_anewton, secs_anewton, func_anewton, data_anewton) = \
        read_dump(output_name + '_Aggr_Newton.csv')

    plot_results(
        [iter_fw, iter_gm, iter_fgm, iter_cnewton, iter_anewton], 
        [secs_fw, secs_gm, secs_fgm, secs_cnewton, secs_anewton],
        [func_fw, func_gm, func_fgm, func_cnewton, func_anewton],
        [data_fw, data_gm, data_fgm, data_cnewton, data_anewton],
        ['Frank-Wolfe', 'Grad. Method', 'Fast Grad. Method', 'Contr. Newton', 
         'Aggr. Newton'],
        ['tab:gray', 'tab:blue', 'tab:green', 'red', 'black'],
        [':', '--', '-.', '-', ':'],
        [3, 3, 3, 3, 5],
        [1, 1, 1, 1, 1],
        ('%s, D = %d' % (name, 2 * R)),
        x_axe_threshold=x_axe_threshold,
        timestamps=timestamps,
        filename=output_name + '.pdf',
        show_legend=show_legend,
        resid_eps=resid_eps,
        )

    print('DONE.')
    print(('=' * 80), flush=True)


run_basic('data/w8a.txt', 'w8a', R=10,
          x_axe_threshold=200,
          timestamps=[[0.50], [0.3], [0.25], [4.6], [7]], 
          show_legend=True)

run_basic('data/w8a.txt', 'w8a', R=50, 
          x_axe_threshold=2000,
          timestamps=[[2.4, 5.1], [2.5, 5], [5.0], [4.5], [7]], 
          show_legend=False)

run_basic('data/w8a.txt', 'w8a', R=250,
          x_axe_threshold=4000,
          timestamps=[[10.0], [15.0], [15.0], [1.3], [15]],
          show_legend=False)

# Extra Experiments:
"""
run_basic('data/a9a.txt', 'a9a', R=10,
          x_axe_threshold=400,
          timestamps=[[0.50], [0.5], [0.5], [0.5], [0.5]], 
          show_legend=True)

run_basic('data/a9a.txt', 'a9a', R=50, 
          x_axe_threshold=3500,
          timestamps=[[4], [5], [5], [1], [1]], 
          show_legend=False)

run_basic('data/a9a.txt', 'a9a', R=250,
          x_axe_threshold=5000,
          timestamps=[[5.0], [10.0], [10.0], [1], [1]],
          show_legend=False)

run_basic('data/connect-4.txt', 'connect-4', R=10,
          x_axe_threshold=400,
          timestamps=[[2], [2], [4], [1], [1, 4]], 
          show_legend=True)

run_basic('data/connect-4.txt', 'connect-4', R=50, 
          x_axe_threshold=700,
          timestamps=[[4], [5], [5], [1], [1]], 
          show_legend=False)

run_basic('data/connect-4.txt', 'connect-4', R=250,
          x_axe_threshold=1000,
          timestamps=[[5.0], [10.0], [10.0], [1], [5]],
          show_legend=False)

run_basic('data/mnist', 'mnist', R=10,
          x_axe_threshold=10000,
          max_time=200,
          timestamps=[[150], [150], [50], [150], [150]], 
          show_legend=True,
          inner_eps=1e-5,
          resid_eps=1e-6)

run_basic('data/mnist', 'mnist', R=50, 
          x_axe_threshold=10000,
          max_time=200,
          timestamps=[[100], [100], [100], [100], [100]], 
          show_legend=False,
          inner_eps=1e-5,
          resid_eps=1e-6)

run_basic('data/mnist', 'mnist', R=250,
          x_axe_threshold=10000,
          max_time=200,
          timestamps=[[100], [100], [100], [100], [100]],
          show_legend=False,
          inner_eps=1e-5,
          resid_eps=1e-6)
"""

