#%%
# 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def get_regressors(trials, Xid = True, regressors = ["target_1", "target_2", "rule_cue"], zscore = True):
    X, num_trials = [], trials.shape[0]
    if Xid:
        X.append(trials["trial_id"].values[:, None]) # trial number
    for reg in regressors:
        Xreg = trials[reg].values.astype(int)-1
        X1hot = np.zeros((num_trials, np.amax(Xreg)+1))
        X1hot[np.arange(num_trials), Xreg] = 1
        X.append(X1hot)

    X = np.concatenate(X, axis = 1) # concatenate everything
    if zscore:
        X = (X - np.mean(X, axis = 0, keepdims = True)) / (1e-3+np.std(X, axis = 0, keepdims = True)) # z-score regressors
    return X


def plot_vars(vars, avg_rates, thresh = 5, fname = "vars", vars_ctrl = None):

    keep_cells = (avg_rates > thresh)
    plt.figure()
    if vars_ctrl is not None:
        plt.scatter(vars_ctrl[keep_cells, 0], vars_ctrl[keep_cells, 1], marker = ".", color = "r", s = 20)
    plt.scatter(vars[keep_cells, 0], vars[keep_cells, 1], marker = ".", color = "k", s = 20)
    plt.savefig(f"figs/{fname}_unique.pdf", bbox_inches = "tight") # clearly a lot of neurons really care about these stimuli
    plt.show()
    plt.close()


def plot_vars_by_time(times, vars_by_bin, modulated, modulated_swap):
    plt.figure(figsize = (5, 4))
    plt.axvline(0.0, color = "k")
    for i in range(2):
        for j in range(3):
            mod = modulated_swap if j == 2 else modulated
            plt.plot(times, vars_by_bin[i][j, mod, :].mean(0), color = plt.get_cmap("tab10")(j), ls = ["-", "--"][i], lw = 2, label = "rule="+["fwd", "bwd"][i]+", pred="+["fwd","bwd","swp"][j])
    plt.xlabel("time (s)")
    plt.ylabel("predictive accuracy")
    plt.legend()
    plt.savefig("figs/fwd_bwd_vars.pdf", bbox_inches = "tight")
    plt.legend(ncol = 3, loc = "upper center", bbox_to_anchor = (0.5, 1.2))
    plt.show()
    plt.close()



def plot_var_pairs(vars, thresh = 0.1):

    ntot, ncat = vars.shape
    fig, axs = plt.subplots(1, 6, figsize = (12,3))
    nax = -1
    for i in range(ncat):
        for j in range(i+1, ncat):
            nax += 1
            v1, v2 = vars[:, i], vars[:, j]
            corr = pearsonr(v1, v2)[0]
            n1, n2 = np.sum(v1 > thresh), np.sum(v2 > thresh)
            n12, nb = np.sum((v1 > thresh) & (v2 > thresh)), n1*n2/ntot
            print(i, j, corr, n1, n2, n12, nb)
            axs[nax].scatter(v1, v2)
    plt.show()
    plt.close()


def plot_pair_modulation(tot_counts, asymms, env, title = None):

    abs_asymms = np.abs(asymms)
    putative_conns = np.where(abs_asymms > env(tot_counts))[0]

    plt.figure()
    plt.scatter(tot_counts, asymms, marker = ".", color = "k", alpha = 0.5, s = 20)
    plt.scatter(tot_counts[putative_conns], asymms[putative_conns], marker = ".", color = "r", alpha = 1.0, s = 30)

    xs = np.linspace(0, np.amax(tot_counts), 100)
    plt.plot(xs, env(xs), color = "r", linestyle = "-")
    plt.plot(xs, -env(xs), color = "r", linestyle = "-")
    plt.ylim(-np.amax(abs_asymms)*1.05, np.amax(abs_asymms)*1.05)
    plt.xlim(4, np.amax(tot_counts)*1.05)
    plt.xlabel("log(counts)")
    plt.ylabel("correlogram asymmetry")
    plt.title(title)
    #plt.savefig("figs/test_asymms.pdf", bbox_inches = "tight")
    plt.show()
    plt.close()

    return
                    
def get_modulated_pairs(spikes,vars, env, kept, dt = 10/1000, var_thresh = 0.1, count_thresh = 100, plot = True):

    ntot, ncat = vars.shape
    all_pairs = []
    all_ns = []
    for i in range(ncat):
        for j in range(ncat): # for each pair of neuron groups
            if i != j:
                pairs, fwds, bwds = [], [], []
                # find neurons tuned to each feature
                inds1, inds2 = kept[vars[:, i] > var_thresh], kept[vars[:, j] > var_thresh]


                n1, n2, n12 = len(inds1), len(inds2), np.sum([(i in inds2) for i in inds1])
                all_ns.append([n1, n2, n12, ntot])

                for n1 in inds1:
                    for n2 in inds2:
                        # for all pairs of unique neurons
                        if (n1 not in inds2) and (n2 not in inds1):
                            count_fwd, count_bwd = 0,0
                            for trial in range(spikes.shape[0]): # for each trial

                                spk1, spk2 = spikes[trial, n1], spikes[trial, n2] # (spikes, )

                                # find all spike pairs that fall within 10 ms
                                count_fwd += (np.searchsorted(spk2, spk1, side="left") - np.searchsorted(spk2, spk1-dt, side="left")).sum()

                                count_bwd += (np.searchsorted(spk2, spk1+dt, side="right") - np.searchsorted(spk2, spk1,  side="right")).sum()

                            if count_fwd + count_bwd > count_thresh: # minimum number of spike pairs in interval
                                pairs.append((n1, n2)) # neuron pair
                                fwds.append(count_fwd) # spike2s within 10 ms after spike1
                                bwds.append(count_bwd) # spike2s within 10 ms before spike1

                fwds = np.array(fwds)
                bwds = np.array(bwds)

                asymms = np.log(fwds + 1) - np.log(bwds + 1)
                tot_counts = np.log(fwds + bwds + 1)

                # plot degree of asymmetry
                if plot: plot_pair_modulation(tot_counts, asymms, env, title = f"{i}, {j}")

                putative_conns = np.where(asymms > env(tot_counts))[0]
                for conn in putative_conns:
                    print(i, j, pairs[conn])

                all_pairs.append(len(putative_conns))
                
    return np.array(all_pairs), np.array(all_ns)

#%%
