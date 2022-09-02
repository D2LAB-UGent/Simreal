import simreal
from simreal.utils.Visualizations import graph_adr_training, graph_adr_params, animate_adr_params, animate_adr_sequential, graph_adr_sequential, graph_adr_sequantial_matrix

from utils import graph_adr_stages

seeds = [1084389005, 1549325168, 1781137871, 2010844016, 2128474801]
seed = seeds[0]
LOG_DIR_ACRO = f'src/logs/acrobot-baseline'
init_params_acro = {'l1': .22, 'l2': .27, 'm1': 3.5, 'm2': .9}


if __name__ == '__main__':

    graph_adr_training(LOG_DIR_ACRO, seed=seed, time=False)

    graph_adr_params(LOG_DIR_ACRO, seed, init_params_acro, show_std=True)

    animate_adr_params(LOG_DIR_ACRO, seed, init_params_acro, param_x='l2', param_y='m1', c_base='time')
    
    animate_adr_sequential(LOG_DIR_ACRO, seed, init_params_acro, param_x='l1', param_y='l2')
    
    graph_adr_sequential(LOG_DIR_ACRO, seed, init_params_acro, param_x='l2', param_y='m1', c_base='time')
    
    graph_adr_stages(LOG_DIR_ACRO, seed)

    graph_adr_sequantial_matrix(LOG_DIR_ACRO, seed, init_params_acro)