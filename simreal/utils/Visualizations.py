import pickle
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import rc
import matplotlib.animation as animation
import numpy as np
import time
import os
import matplotlib.tri as tri


def open_logs(LOG_DIR):
    with open("{}/train_logs.pkl".format(LOG_DIR), 'rb') as file:
        train_log_data = pickle.load(file)
    with open("{}/eval_logs.pkl".format(LOG_DIR), 'rb') as file:
        eval_log_data = pickle.load(file)

    df_train = pd.DataFrame(train_log_data)
    df_eval = pd.DataFrame(eval_log_data)
    return df_train, df_eval


def graph_eval_train(LOG_DIR, show_loss=True, time=False):
    df_train , df_eval = open_logs(LOG_DIR)

    eval_means = np.array([])
    eval_std = np.array([])
    for ev in df_eval['online_return'].tolist():
        eval_means = np.append(eval_means, ev[0].mean)
        eval_std = np.append(eval_std, ev[0].std)

    if time == False:
        train_times = pd.DataFrame(df_train['accumulated_trained_timesteps'].tolist())
        eval_times = pd.DataFrame(df_eval['accumulated_trained_timesteps'].tolist())
    elif time == True:
        train_times = pd.DataFrame(df_train['train_time'].tolist())
        eval_times = pd.DataFrame(df_eval['eval_time'].tolist())

    train_returns = pd.DataFrame(df_train['episode_return'].tolist())

    actor_loss = pd.DataFrame(df_train['actor_loss'].tolist())
    critic_loss = pd.DataFrame(df_train['critic_loss'].tolist())
    alpha_loss = pd.DataFrame(df_train['alpha_loss'].tolist())

    # actor_grad_norm = pd.DataFrame(df_train['actor_grad_norm'].tolist())
    # critic_grad_norm = pd.DataFrame(df_train['critic_grad_norm'].tolist())
    # alpha = pd.DataFrame(df_train['alpha'].tolist())

    fig = plt.figure()

    if show_loss:
        ax1 = plt.subplot2grid((2, 1), (0, 0))
    else:
        ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(train_times, train_returns, label = "train_return", color='C0')
    ax1.plot(eval_times, eval_means, label = "eval_return", color='C1')
    ax1.fill_between(np.transpose(np.asarray(eval_times))[0], eval_means+eval_std, eval_means-eval_std, facecolor='C1', alpha=0.5)
    ax1.set_title('Train and Evaluation Returns')
    if not show_loss:
        if time == False:
            ax1.set_xlabel('iteration')
        elif time == True:
            ax1.set_xlabel('time [sec]')
    ax1.set_ylabel('Return')
    ax1.legend(loc=4)

    if show_loss:
        ax2 = plt.subplot2grid((2,1), (1,0))
        ax2.plot(train_times, actor_loss, label = "actor_loss")
        ax2.plot(train_times, critic_loss, label="critic_loss")
        ax2.plot(train_times, alpha_loss, label="alpha_loss")
        ax2.set_title('Losses')
        ax2.set_xlabel('time step')
        ax2.legend(loc=1)

    plt.show()

def graph_eval_train_PPO(LOG_DIR, time=False, save_name=None):
    df_train, df_eval = open_logs(LOG_DIR)

    if time == False:
        train_times = pd.DataFrame(np.transpose(df_train['accumulated_trained_timesteps'].tolist())[0])
        eval_times = pd.DataFrame(df_eval['accumulated_trained_timesteps'].tolist())
    elif time == True:
        train_times = pd.DataFrame(np.transpose(df_train['train_time'].tolist())[0])
        eval_times = pd.DataFrame(df_eval['eval_time'].tolist())

    train_means = np.array([])
    train_std = np.array([])
    for it in df_train['return'].tolist():
        train_means = np.append(train_means, it[0].mean)
        train_std = np.append(train_std, it[0].std)

    eval_means = np.array([])
    eval_std = np.array([])
    for ev in df_eval['online_return'].tolist():
        eval_means = np.append(eval_means, ev[0].mean)
        eval_std = np.append(eval_std, ev[0].std)

    fig = plt.figure()

    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(train_times, train_means, label = "train_return", color='C0')
    ax1.fill_between(np.transpose(np.asarray(train_times))[0], train_means + train_std, train_means - train_std,
                     facecolor='C0', alpha=0.5)
    ax1.plot(eval_times, eval_means, label = "eval_return", color='C1')
    ax1.fill_between(np.transpose(np.asarray(eval_times))[0], eval_means+eval_std, eval_means-eval_std, facecolor='C1', alpha=0.5)
    ax1.set_title('Train and Evaluation Returns')
    ax1.set_ylabel('Return')
    ax1.set_ylim(-600, 50)
    ax1.legend(loc=4)

    if time == False:
        ax1.set_xlabel('iteration')
    elif time==True:
        ax1.set_xlabel('time [sec]')

    if save_name != None:
        plt.savefig(save_name)

    plt.show()

def graph_params(LOG_DIR, init_params, show_std = True):
    df_train, df_eval = open_logs(LOG_DIR)
    train_times = pd.DataFrame(df_train['accumulated_trained_timesteps'].tolist())

    means, stds = dict(), dict()
    param_batches = df_train['param_batch'].tolist()
    for param_batch in param_batches:
        param_batch = param_batch[0]

        PARAM_LIST = dict()
        for param_episode in param_batch:
            # print(param_episode)
            for param_dict in param_episode:
                for key, value in param_dict.items():
                    if key in PARAM_LIST.keys():
                        PARAM_LIST[key] = PARAM_LIST[key] + [value]
                    else:
                        PARAM_LIST[key] = [value]

        for key, value in PARAM_LIST.items():
            if key in means.keys():
                means[key] = means[key] + [np.mean(value)]
            else:
                means[key] = [np.mean(value)]
            if key in stds.keys():
                stds[key] = stds[key] + [np.std(value)]
            else:
                stds[key] = [np.std(value)]

    fig = plt.figure()

    ax1 = plt.subplot2grid((1, 1), (0, 0))
    for key, value in means.items():
        normalized_mean = np.array(value)/init_params[key]
        ax1.plot(train_times, normalized_mean, label=key)
    if show_std:
        for key, value in stds.items():
            normalized_mean = np.array(means[key])/init_params[key]
            normalized_std = np.array(value)/init_params[key]
            ax1.fill_between(np.transpose(np.array(train_times))[0], normalized_mean + normalized_std,
                             normalized_mean - normalized_std, alpha=0.5)
    ax1.set_title('Parameter plot')
    ax1.set_xlabel('time step')
    ax1.legend(loc=4)

    plt.show()

def get_sequential_data(LOG_DIR, init_params, param_x, param_y):
    df_train, df_eval = open_logs(LOG_DIR)
    train_times = pd.DataFrame(df_train['accumulated_trained_timesteps'].tolist())

    total_dict, param_dict_old = dict(), dict()
    param_batch_old = None
    param_batches = df_train['param_batch'].tolist()

    for param_batch in param_batches:
        param_batch = param_batch[0]

        if param_batch_old != param_batch:
            PARAM_LIST = dict()
            for param_episode in param_batch:
                for param_dict in param_episode:

                    add_to_total = False
                    for key, value in param_dict.items():
                        if not key in param_dict_old:
                            add_to_total = True
                        elif param_dict_old[key] != value:
                            add_to_total = True
                    param_dict_old = param_dict
                    if add_to_total:
                        for key, value in param_dict.items():
                            if key in PARAM_LIST.keys():
                                PARAM_LIST[key] = PARAM_LIST[key] + [value]
                            else:
                                PARAM_LIST[key] = [value]

            for key, value in PARAM_LIST.items():
                if key in total_dict.keys():
                    total_dict[key] = total_dict[key] + value
                else:
                    total_dict[key] = value

        param_batch_old = param_batch

    dataX = np.array(total_dict[param_x])
    dataY = np.array(total_dict[param_y])

    #normalise data
    dataX = dataX/float(init_params[param_x])
    dataY = dataY/float(init_params[param_y])

    return dataX, dataY


def animate_params(LOG_DIR, init_params, param_x, param_y, c_base='time',save = False):
    dataX, dataY = get_sequential_data(LOG_DIR, init_params, param_x, param_y)

    #compute colors
    dataC = []
    for i in range(len(dataX)):
        if c_base == 'distance':
            c = np.linalg.norm(np.array([dataX[i], dataY[i]]) - np.array([1., 1.]))
        elif c_base == 'time':
            c = -i
        dataC += [c]
    dataC = np.array(dataC)

    def animate(num):
        data = np.hstack((dataX[:num, np.newaxis], dataY[:num, np.newaxis]))
        art.set_offsets(data)
        art.set_color(cmap(norm(dataC[:num])))  # update colors using the colorbar and its normalization defined below
        return art,

    fig = plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0))

    cmap = matplotlib.cm.plasma
    norm = matplotlib.colors.Normalize(vmin=np.min(dataC), vmax=np.max(dataC))
    # cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')

    ax.scatter(1., 1., s=100, c='g')
    ax.set_xlim(0.50, 1.50)
    ax.set_ylim(0.50, 1.50)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title('Param plot: {} vs {}'.format(param_x, param_y))
    ax.set_aspect('equal')
    art = ax.scatter([], [], c=[])

    if save:
        frames_to_save = 300
        ani = animation.FuncAnimation(fig, animate, frames=frames_to_save, interval=5, blit=True, repeat=True, save_count=frames_to_save)
        rc('animation', html='html5')
        ani.save('Animation-{}-vs-{}-{}.gif'.format(param_x, param_y, int(time.time())), writer='imagemagick', fps=30)
    else:
        ani = animation.FuncAnimation(fig, animate, interval=5, blit=True, repeat=True)
        plt.show()

def animate_sequential(LOG_DIR, init_params, param_x, param_y, save = False):
    dataX, dataY = get_sequential_data(LOG_DIR, init_params, param_x, param_y)

    def animate(num):
        data = np.hstack((dataX[:num, np.newaxis], dataY[:num, np.newaxis]))
        if len(data) != 0:
            data_present = data[-1]
            data_past = data[:-1]
            art_past.set_offsets(data_past)
            art_present.set_offsets(data_present)
            art_past.set_color('y')
            art_present.set_color('b')
        return art_present, art_past,

    fig = plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0))

    ax.scatter(1., 1., s=100, c='k')
    ax.set_xlim(0.50, 1.50)
    ax.set_ylim(0.50, 1.50)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title('Param plot: {} vs {}'.format(param_x, param_y))
    ax.set_aspect('equal')
    art_past = ax.scatter([], [], c=[])
    art_present = ax.scatter([], [], c=[], s=100)

    if save:
        frames_to_save = 300
        ani = animation.FuncAnimation(fig, animate, frames=frames_to_save, interval=5, blit=True, repeat=True, save_count=frames_to_save)
        rc('animation', html='html5')
        ani.save('Animation-sequential-{}-vs-{}-{}.gif'.format(param_x, param_y, int(time.time())), writer='imagemagick', fps=30)
    else:
        ani = animation.FuncAnimation(fig, animate, interval=5, blit=True, repeat=True)
        plt.show()


def plot_cem(params_logs, elite_logs, mean_logs, std_logs, Fs_logs, heatmap=False):

    # determin axes dimensions
    l1_min, l1_max, l2_min, l2_max, m1_min, m1_max, m2_min, m2_max = [], [], [], [], [], [], [], []
    for i, params in enumerate(params_logs):
        params = np.array(params)
        l1, l2, m1, m2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]
        l1_min += [min(l1)]
        l1_max += [max(l1)]
        l2_min += [min(l2)]
        l2_max += [max(l2)]
        m1_min += [min(m1)]
        m1_max += [max(m1)]
        m2_min += [min(m2)]
        m2_max += [max(m2)]

    delta_l = max(max(l1_max)-min(l1_min), max(l2_max)-min(l2_min))
    delta_m = max(max(m1_max)-min(m1_min), max(m2_max)-min(m2_min))

    for params, elites, mean, std in zip(params_logs, elite_logs, mean_logs, std_logs):
        params = np.array(params)
        l1, l2, m1, m2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

        elites = np.array(elites)
        l1_elite, l2_elite, m1_elite, m2_elite = elites[:, 0], elites[:, 1], elites[:, 2], elites[:, 3]

        std_mul = 3
        plt.figure()

        ax1 = plt.subplot2grid((1, 2), (0, 0))
        ellipse_l = Ellipse(xy=(mean[0], mean[1]), width=std_mul*std[0], height=std_mul*std[1],
                          edgecolor='r', fc='r', lw=0, alpha=0.5)
        ax1.add_patch(ellipse_l)

        ax1.set_aspect('equal')
        ax1.scatter(l1, l2, c='y', s=10)
        ax1.scatter(l1_elite, l2_elite, c='b', s=15)
        ax1.scatter(mean[0], mean[1], c='r', s=20)
        ax1.set_xlabel('l1 [m]')
        ax1.set_xbound(mean[0]-delta_l/2, mean[0]+delta_l/2)
        ax1.set_ylabel('l2 [m]')
        ax1.set_ybound(mean[1]-delta_l/2, mean[1]+delta_l/2)
        ax1.set_title('Lengths')

        ax2 = plt.subplot2grid((1, 2), (0, 1))
        ellipse_m = Ellipse(xy=(mean[2], mean[3]), width=std_mul*std[2], height=std_mul*std[3],
                          edgecolor='r', fc='r', lw=0, alpha=0.5)
        ax2.add_patch(ellipse_m)

        # ax2.set_aspect('equal')
        ax2.scatter(m1, m2, c='y', s=10)
        ax2.scatter(m1_elite, m2_elite, c='b', s=15)
        ax2.scatter(mean[2], mean[3], c='r', s=20)
        ax2.set_xlabel('m1 [kg]')
        ax2.set_xbound(mean[2]-delta_m/2, mean[2]+delta_m/2)
        ax2.set_ylabel('m2 [kg]')
        ax2.set_ybound(mean[3]-delta_m/10, mean[3]+delta_m/10)
        ax2.set_title('Masses')

        plt.show()

    if heatmap:
        for params, mean, Fs in zip(params_logs, mean_logs, Fs_logs):
            params = np.array(params)
            l1, l2, m1, m2 = params[:, 0], params[:, 1], params[:, 2], params[:, 3]

            # filter outliers
            outliers = np.where(Fs > 2*np.mean(Fs))
            l1 = np.delete(l1, outliers)
            l2 = np.delete(l2, outliers)
            m1 = np.delete(m1, outliers)
            m2 = np.delete(m2, outliers)
            Fs = np.delete(Fs, outliers)

            fig = plt.figure()

            ax1 = plt.subplot2grid((1, 2), (0, 0))

            # Create grid values first.
            xi = np.linspace(min(l1_min), max(l1_max), 1000)
            yi = np.linspace(min(l2_min), max(l2_max), 1000)
            # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
            triang = tri.Triangulation(l1, l2)
            interpolator = tri.LinearTriInterpolator(triang, Fs)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)

            ax1.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
            cntr1 = ax1.contourf(xi, yi, zi, levels=14, cmap="RdBu")
            fig.colorbar(cntr1, ax=ax1)
            ax1.plot(l1, l2, 'ko', ms=1)


            ax1.set_aspect('equal')
            ax1.set_xlabel('l1 [m]')
            ax1.set_xbound(mean[0] - delta_l / 2, mean[0] + delta_l / 2)
            ax1.set_ylabel('l2 [m]')
            ax1.set_ybound(mean[1] - delta_l / 2, mean[1] + delta_l / 2)
            ax1.set_title('Lengths')

            ax2 = plt.subplot2grid((1, 2), (0, 1))

            # Create grid values first.
            xi = np.linspace(min(m1_min), max(m1_max), 1000)
            yi = np.linspace(min(m2_min), max(m2_max), 1000)
            # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
            triang = tri.Triangulation(m1, m2)
            interpolator = tri.LinearTriInterpolator(triang, Fs)
            Xi, Yi = np.meshgrid(xi, yi)
            zi = interpolator(Xi, Yi)

            ax2.contour(xi, yi, zi, levels=14, linewidths=0.5, colors='k')
            cntr1 = ax2.contourf(xi, yi, zi, levels=14, cmap="RdBu")
            fig.colorbar(cntr1, ax=ax2)
            ax2.plot(m1, m2, 'ko', ms=1)

            # ax2.set_aspect('equal')
            ax2.set_xlabel('m1 [kg]')
            ax2.set_xbound(mean[2] - delta_m / 2, mean[2] + delta_m / 2)
            ax2.set_ylabel('m2 [kg]')
            ax2.set_ybound(mean[3] - delta_m / 10, mean[3] + delta_m / 10)
            ax2.set_title('Masses')

            plt.show()

def get_eval_train(LOG_DIR, time=False):
    df_train, df_eval = open_logs(LOG_DIR)

    if time == False:
        train_times = pd.DataFrame(np.transpose(df_train['accumulated_trained_timesteps'].tolist())[0])
        eval_times = pd.DataFrame(df_eval['accumulated_trained_timesteps'].tolist())
    elif time == True:
        train_times = pd.DataFrame(np.transpose(df_train['train_time'].tolist())[0])
        eval_times = pd.DataFrame(df_eval['eval_time'].tolist())

    train_means = np.array([])
    train_std = np.array([])
    for it in df_train['return'].tolist():
        train_means = np.append(train_means, it[0].mean)
        train_std = np.append(train_std, it[0].std)

    eval_means = np.array([])
    eval_std = np.array([])
    for ev in df_eval['online_return'].tolist():
        eval_means = np.append(eval_means, ev[0].mean)
        eval_std = np.append(eval_std, ev[0].std)

    return train_times, eval_times, train_means, train_std, eval_means, eval_std


def graph_adr_training(LOG_DIR, seed, time=False, save_name=None):
    l = []
    for iteration in os.listdir(LOG_DIR):
        if iteration.startswith('PPO'):
            l += [int(iteration[len('PPO-'):])]

    for i in range(np.max(l)+1):
        iteration_dir = f'{LOG_DIR}/PPO-{i}/0/{seed}'

        train_time, eval_time, train_mean, train_std, eval_mean, eval_std = get_eval_train(LOG_DIR=iteration_dir, time=time)

        if i == 0:
            train_times = np.array(np.transpose(train_time))[0]
            eval_times = np.array(np.transpose(eval_time))[0]
            train_means = train_mean
            train_stds = train_std
            eval_means = eval_mean
            eval_stds = eval_std
        else:
            if time == False:
                train_times = np.append(train_times, np.array(np.transpose(train_time))[0])
                eval_times = np.append(eval_times, np.array(np.transpose(eval_time))[0])
            elif time == True:
                train_times = np.append(train_times, train_times[-1] + np.array(np.transpose(train_time))[0])
                eval_times = np.append(eval_times, eval_times[-1] + np.array(np.transpose(eval_time))[0])
            train_means = np.append(train_means, train_mean)
            train_stds = np.append(train_stds, train_std)
            eval_means = np.append(eval_means, eval_mean)
            eval_stds = np.append(eval_stds, eval_std)

    fig = plt.figure()

    ax1 = plt.subplot2grid((1, 1), (0, 0))
    ax1.plot(train_times, train_means, label = "train_return", color='C0')
    ax1.fill_between(train_times, train_means + train_stds, train_means - train_stds,
                     facecolor='C0', alpha=0.5)
    ax1.plot(eval_times, eval_means, label = "eval_return", color='C1')
    ax1.fill_between(eval_times, eval_means+eval_stds, eval_means-eval_stds, facecolor='C1', alpha=0.5)
    ax1.set_title('Train and Evaluation Returns')
    ax1.set_ylabel('Return')
    ax1.set_ylim(-600, 50)
    ax1.legend(loc=4)

    if time == False:
        ax1.set_xlabel('iteration')
    elif time==True:
        ax1.set_xlabel('time [sec]')

    if save_name != None:
        plt.savefig(save_name)

    plt.show()


def get_params(LOG_DIR, init_params):
    df_train, df_eval = open_logs(LOG_DIR)
    train_times = pd.DataFrame(df_train['accumulated_trained_timesteps'].tolist())

    means, stds = dict(), dict()
    param_batches = df_train['param_batch'].tolist()
    for param_batch in param_batches:
        param_batch = param_batch[0]

        PARAM_LIST = dict()
        for param_episode in param_batch:
            # print(param_episode)
            for param_dict in param_episode:
                for key, value in param_dict.items():
                    if key in PARAM_LIST.keys():
                        PARAM_LIST[key] = PARAM_LIST[key] + [value]
                    else:
                        PARAM_LIST[key] = [value]

        for key, value in PARAM_LIST.items():
            if key in means.keys():
                means[key] = means[key] + [np.mean(value)]
            else:
                means[key] = [np.mean(value)]
            if key in stds.keys():
                stds[key] = stds[key] + [np.std(value)]
            else:
                stds[key] = [np.std(value)]

    return means, stds, train_times


def graph_adr_params(LOG_DIR, seed, init_params, show_std=True):
    l = []
    for iteration in os.listdir(LOG_DIR):
        if iteration.startswith('PPO'):
            l += [int(iteration[len('PPO-'):])]

    for i in range(np.max(l)+1):
        iteration_dir = f'{LOG_DIR}/PPO-{i}/0/{seed}'
        mean, std, train_time = get_params(iteration_dir, init_params)

        if i == 0:
            train_times = np.array(np.transpose(train_time))[0]
            means = mean
            stds = std
        else:
            train_times = np.append(train_times, np.array(np.transpose(train_time))[0])
            for key in means.keys():
                means[key] = np.append(means[key], mean[key])
                stds[key] = np.append(stds[key], std[key])
    fig = plt.figure()

    ax1 = plt.subplot2grid((1, 1), (0, 0))
    for key, value in means.items():
        normalized_mean = np.array(value)/init_params[key]
        ax1.plot(train_times, normalized_mean, label=key)
    if show_std:
        for key, value in stds.items():
            normalized_mean = np.array(means[key])/init_params[key]
            normalized_std = np.array(value)/init_params[key]
            ax1.fill_between(np.transpose(np.array(train_times))[0], normalized_mean + normalized_std,
                             normalized_mean - normalized_std, alpha=0.5)
    ax1.set_title('Parameter plot')
    ax1.set_xlabel('time step')
    ax1.legend(loc=4)

    plt.show()


def animate_adr_params(LOG_DIR, seed, init_params, param_x, param_y, c_base='time', save=False):
    l = []
    for iteration in os.listdir(LOG_DIR):
        if iteration.startswith('PPO'):
            l += [int(iteration[len('PPO-'):])]

    for i in range(np.max(l)+1):
        iteration_dir = f'{LOG_DIR}/PPO-{i}/0/{seed}'
        datax, datay = get_sequential_data(iteration_dir, init_params, param_x, param_y)

        if i == 0:
            dataX = datax
            dataY = datay
        else:
            dataX = np.append(dataX, datax)
            dataY = np.append(dataY, datay)

    # compute colors
    dataC = []
    for i in range(len(dataX)):
        if c_base == 'distance':
            c = np.linalg.norm(np.array([dataX[i], dataY[i]]) - np.array([1., 1.]))
        elif c_base == 'time':
            c = -i
        dataC += [c]
    dataC = np.array(dataC)

    def animate(num):
        print(f'{num}/{len(dataX)}')
        data = np.hstack((dataX[:num, np.newaxis], dataY[:num, np.newaxis]))
        art.set_offsets(data)
        art.set_color(
            cmap(norm(dataC[:num])))  # update colors using the colorbar and its normalization defined below
        return art,

    fig = plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0))

    cmap = matplotlib.cm.plasma
    norm = matplotlib.colors.Normalize(vmin=np.min(dataC), vmax=np.max(dataC))
    # cb1 = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation='vertical')

    ax.scatter(1., 1., s=100, c='g')
    ax.set_xlim(0.50, 1.50)
    ax.set_ylim(0.50, 1.50)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title('Param plot: {} vs {}'.format(param_x, param_y))
    ax.set_aspect('equal')
    art = ax.scatter([], [], c=[])

    if save:
        frames_to_save = 300
        ani = animation.FuncAnimation(fig, animate, frames=frames_to_save, interval=5, blit=True, repeat=True,
                                      save_count=frames_to_save)
        rc('animation', html='html5')
        ani.save('Animation-{}-vs-{}-{}.gif'.format(param_x, param_y, int(time.time())), writer='imagemagick',
                 fps=30)
    else:
        ani = animation.FuncAnimation(fig, animate, interval=5, blit=True, repeat=True)
        plt.show()


def animate_adr_sequential(LOG_DIR, seed, init_params, param_x, param_y, save=False):
    l = []
    for iteration in os.listdir(LOG_DIR):
        if iteration.startswith('PPO'):
            l += [int(iteration[len('PPO-'):])]

    for i in range(np.max(l)+1):
        iteration_dir = f'{LOG_DIR}/PPO-{i}/0/{seed}'
        datax, datay = get_sequential_data(iteration_dir, init_params, param_x, param_y)

        if i == 0:
            dataX = datax
            dataY = datay
        else:
            dataX = np.append(dataX, datax)
            dataY = np.append(dataY, datay)

    def animate(num):
        print(f'{num}/{len(dataX)}')
        data = np.hstack((dataX[:num, np.newaxis], dataY[:num, np.newaxis]))
        if len(data) != 0:
            data_present = data[-1]
            data_past = data[:-1]
            art_past.set_offsets(data_past)
            art_present.set_offsets(data_present)
            art_past.set_color('y')
            art_present.set_color('b')
        return art_present, art_past,

    fig = plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0))

    ax.scatter(1., 1., s=100, c='k')
    ax.set_xlim(0.50, 1.50)
    ax.set_ylim(0.50, 1.50)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title('Param plot: {} vs {}'.format(param_x, param_y))
    ax.set_aspect('equal')
    art_past = ax.scatter([], [], c=[])
    art_present = ax.scatter([], [], c=[], s=100)

    if save:
        frames_to_save = 300
        ani = animation.FuncAnimation(fig, animate, frames=frames_to_save, interval=5, blit=True, repeat=True, save_count=frames_to_save)
        rc('animation', html='html5')
        ani.save('Animation-sequential-{}-vs-{}-{}.gif'.format(param_x, param_y, int(time.time())), writer='imagemagick', fps=30)
    else:
        ani = animation.FuncAnimation(fig, animate, interval=5, blit=True, repeat=True)
        plt.show()


def graph_adr_sequential(LOG_DIR, seed, init_params, param_x, param_y, c_base='time', save=False):
    l = []
    for iteration in os.listdir(LOG_DIR):
        if iteration.startswith('PPO'):
            l += [int(iteration[len('PPO-'):])]

    for i in range(np.max(l)+1):
        iteration_dir = f'{LOG_DIR}/PPO-{i}/0/{seed}'
        datax, datay = get_sequential_data(iteration_dir, init_params, param_x, param_y)

        if i == 0:
            dataX = datax
            dataY = datay
        else:
            dataX = np.append(dataX, datax)
            dataY = np.append(dataY, datay)

    # compute colors
    dataC = []
    for i in range(len(dataX)):
        if c_base == 'distance':
            c = np.linalg.norm(np.array([dataX[i], dataY[i]]) - np.array([1., 1.]))
        elif c_base == 'time':
            c = -i
        dataC += [c]
    dataC = np.array(dataC)

    fig = plt.figure()
    ax = plt.subplot2grid((1, 1), (0, 0))

    ax.scatter(1., 1., s=100, c='k')
    ax.scatter(dataX, dataY, c=dataC)
    ax.set_xlim(0.50, 1.50)
    ax.set_ylim(0.50, 1.50)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title('Param plot: {} vs {}'.format(param_x, param_y))
    ax.set_aspect('equal')
    plt.show()


def graph_adr_sequantial_matrix(LOG_DIR, seed, init_params, c_base='time'):
    n_rows = len(init_params)
    n_cols = len(init_params)

    keys = [key for key in init_params.keys()]

    l = []
    for iteration in os.listdir(LOG_DIR):
        if iteration.startswith('PPO'):
            l += [int(iteration[len('PPO-'):])]

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(12, 10))

    for p in range(n_rows):
        for q in range(n_cols):
            ax = axes[q][p]
            x_param = keys[p]
            y_param = keys[q]

            for i in range(np.max(l) + 1):
                iteration_dir = f'{LOG_DIR}/PPO-{i}/0/{seed}'
                datax, datay = get_sequential_data(iteration_dir, init_params, x_param, y_param)

                if i == 0:
                    dataX = datax
                    dataY = datay
                else:
                    dataX = np.append(dataX, datax)
                    dataY = np.append(dataY, datay)

            # compute colors
            dataC = []
            for i in range(len(dataX)):
                if c_base == 'distance':
                    c = np.linalg.norm(np.array([dataX[i], dataY[i]]) - np.array([1., 1.]))
                elif c_base == 'time':
                    c = -i
                dataC += [c]
            dataC = np.array(dataC)

            ax.scatter(1., 1., s=10, c='k')
            ax.scatter(dataX, dataY, c=dataC, s=1)

            if p == 1:
                ax.set_xlim(0.1, 2.5)
            else:
                ax.set_xlim(0.1, 2.5)

            if q == 1:
                ax.set_ylim(0.1, 2.5)
            else:
                ax.set_ylim(0.1, 2.5)

            ax.set_xlabel(x_param)
            ax.set_ylabel(y_param)
            # ax.set_aspect('equal')

    plt.show()

