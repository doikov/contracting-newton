
import matplotlib.pyplot as plt
import numpy as np


def read_dump(filename):
    data = np.loadtxt(filename, delimiter=',')
    iters = data[0, :]
    secs = data[1, :]
    func = data[2, :]
    data_accesses = data[3, :]
    return (iters, secs, func, data_accesses)


def plot_results(iterations, seconds, functions, data_accesses, labels, colors, 
                 linestyles, linewidths, alphas, title,
                 x_axe_threshold, timestamps=[], filename=None, 
                 show_legend=True, use_data_accesses=False, M=1, 
                 resid_eps=1e-7):
    
    f_star = min([np.min(f) for f in functions])
    fig, ax = plt.subplots(figsize=(6, 5))
    delta_0 = 0.0125
    delta_1 = 0.015
    divisor = M if use_data_accesses else 1
    
    if use_data_accesses:
        iterations_to_use = data_accesses
    else:
        iterations_to_use = iterations
    
    for plot_round in range(2):
    
        for i, iters in enumerate(iterations_to_use):
            secs = seconds[i]
            resid = functions[i] - f_star

            mask1 = iters / divisor < x_axe_threshold
            mask2 = resid > resid_eps
            k = len(mask2)
            for ii in range(k):
                if not mask2[ii]:
                    mask2[ii] = 1
                    break
                
            mask = mask1 & mask2

            iters_ = iters[mask] / divisor
            resid_ = resid[mask]
            secs_ = secs[mask]
            
            if plot_round == 0:
                # Plot graphs.
                ax.semilogy(iters_, resid_, label=labels[i], color=colors[i], 
                            linestyle=linestyles[i], linewidth=linewidths[i], 
                            alpha=alphas[i])
            else:
                # Plot timestamps.
                for timestamp in timestamps[i]:
            
                    diff = np.abs(secs_ - timestamp)
                    j = np.where(diff == np.amin(diff))[0][0]
                    ax.plot([iters_[j]], [resid_[j]], 'o', color=colors[i], 
                            markersize=10)

                    props = dict(boxstyle='square', facecolor='white', 
                                 alpha=1.0, ec=colors[i], lw=1)
                    textstr = '%.2f' % secs_[j]
                    if textstr[-2:] == '00':
                        textstr = textstr[:-3]
                    elif textstr[-1:] == '0':
                        textstr = textstr[:-1]
                    textstr += 's'
                    
                    axis_to_data = ax.transAxes + ax.transData.inverted()
                    data_to_axis = axis_to_data.inverted()
                    coords = data_to_axis.transform([iters_[j], resid_[j]])
                    ax.text(coords[0] + delta_0, 
                            coords[1] + delta_1, 
                            textstr, fontsize=14, transform=ax.transAxes,
                            verticalalignment='bottom', bbox=props)
    
    if show_legend:
        plt.legend(fontsize=16)
    if use_data_accesses:
        plt.xlabel('Epochs', fontsize=16)
    else:
        plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Func. residual', fontsize=16)
    plt.tick_params(labelsize=14)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    
    if filename is not None:
        plt.savefig(filename)

