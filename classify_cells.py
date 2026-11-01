
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from load_data import get_data, get_binned_activity
from utils import var_explained
from scipy.stats import pearsonr

#%% load data

trials, spikes = get_data()
num_trials, num_cells = spikes.shape




"""
We want to identify neurons with (i) 'tuning' for each stimulus 1, (ii) tuning for each stimulus 2, (iii) tuning for rule cue, (iv) tuning for rule cue and stimulus 1, (v) tuning for rule cue and stimulus 2
to do so, we are going to fit a multiple regression model to the average activity within some period of interest.
We first identify the 'stimulus' tuning after stim 2, and we then look at variance explained by the 'true' or 'reversed' stims over time on forward and reverse trials.
The question is whether we see the reversal in our _encoding_ model that Tian et al. see in their _decoding_ model.
"""

#%% first extract some 'training' activity

# average activity from 100 to 500 ms after stimulus 2 offset (this is before the rule cue comes on)
activity = get_binned_activity(trials, spikes, "stimulus_2_off", 0.1, 0.5, 1, mode = "rates")[..., 0] # (trials, cells)

def get_regressors(trials, Xid = True, regressors = ["target_1", "target_2", "rule_cue"], normalise = True):
    X = []
    if Xid:
        X.append(trials["trial_id"].values[:, None]) # trial number
    for reg in regressors:
        Xreg = trials["target_1"].values-1

    Xstim1 = np.zeros((num_trials, 6)) # one-hot encoding of first target
    Xstim1[np.arange(num_trials), trials["target_1"].values-1] = 1
    Xstim2 = np.zeros((num_trials, 6)) # one-hot encoding of second-target
    Xstim2[np.arange(num_trials), trials["target_2"].values.astype(int)-1] = 1
    Xrule = np.zeros((num_trials, 2)) # one-hot encoding of rule (fwd/bwd)
    Xrule[np.arange(num_trials), trials["rule_cue"].values.astype(int)-1] = 1

    X = np.concatenate([Xid, Xstim1, Xstim2, Xrule], axis = 1) # concatenate everything
    if normalise: X = (X - np.mean(X, axis = 0, keepdims = True)) / (1e-3+np.std(X, axis = 0, keepdims = True)) # z-score regressors

    return X

print("rate quantiles:", np.quantile(activity.mean(0), [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
avg_rates = activity.mean(0)
#activity = activity[:, keep_cells] # at least 5 hz
Ymean, Ystd = np.mean(activity, axis = 0, keepdims = True), np.std(activity, axis = 0, keepdims = True) # mean/std across trials for each cell
Y = (activity - Ymean) / (Ystd + 1e-3)
print(Y.shape)
Yctrl = Y[np.random.choice(Y.shape[0], Y.shape[0], replace = False), :]

#%% fit encodings models to this data
model = ElasticNet(alpha = 0.1, l1_ratio = 0.0, fit_intercept = True)

# first fit full model
reg = model.fit(X, Y)
pred = reg.predict(X)
tot_var_explained = var_explained(Y, pred) # total variance explained by this model

ctrl_var_explained = var_explained(Yctrl, model.fit(X, Yctrl).predict(X)) # total variance explained after shuffling trials

fig, axs = plt.subplots(1, 2, figsize = (8,4))
axs[0].hist(tot_var_explained, bins = np.linspace(0, 1, 21))
axs[1].hist(ctrl_var_explained, bins = np.linspace(0, 1, 21))
plt.savefig("tot_var.pdf")
plt.close()

# now fit without stim1/stim2
X_m_stim1 = np.concatenate([Xid, Xstim2, Xrule], axis = 1)
X_m_stim2 = np.concatenate([Xid, Xstim1, Xrule], axis = 1)
partial_vars, partial_vars_ctrl = [], []
for partial_X in [X_m_stim1, X_m_stim2]: # fit each partial model
    partial_X = (partial_X - np.mean(partial_X, axis = 0, keepdims = True)) / np.std(partial_X, axis = 0, keepdims = True)
    partial_vars.append(var_explained(Y, model.fit(partial_X, Y).predict(partial_X))) # quantify variance explained
    partial_vars_ctrl.append(var_explained(Yctrl, model.fit(partial_X, Yctrl).predict(partial_X))) # quantify ctrl variance explained

uvars = tot_var_explained[:, None] - np.array(partial_vars).T # what is the unique variance explained by stim 1 and stim 2?
uvars_ctrl = ctrl_var_explained[:, None] - np.array(partial_vars_ctrl).T # what is the unique variance explained by stim 1 and stim 2?
sum_vars = np.sum(uvars, 1)
keep_cells = (avg_rates > 5)
plt.figure()
plt.scatter(uvars_ctrl[keep_cells, 0], uvars_ctrl[keep_cells, 1], marker = ".", color = "r", s = 20)
plt.scatter(uvars[keep_cells, 0], uvars[keep_cells, 1], marker = ".", color = "k", s = 20)
plt.savefig("test_uvars.pdf", bbox_inches = "tight") # clearly a lot of neurons really care about these stimuli
plt.close()

fig, axs = plt.subplots(1, 2, figsize = (8,4))
axs[0].hist(sum_vars, bins = np.linspace(0, 0.5, 21))
axs[1].hist(np.sum(uvars_ctrl, 1), bins = np.linspace(0, 0.5, 21))
plt.savefig("sum_uvars.pdf")
plt.show()
plt.close()

#%% now apply this model over time from 250 ms before rule cue on to 750 after rule cue on
# the question is whether the 'variance explained' by the stimulus cues will swap


test_activity = get_binned_activity(trials, spikes, "rule_cue_on", -0.25, 0.75, 10, mode = "rates") # 100 ms bins
times = np.linspace(-0.25, 0.75, 11)
times = times[1:]/2 + times[:-1]/2
Ytest = (test_activity - Ymean[..., None] ) / (Ystd[..., None]+1e-3) # trials by cells by bins

reg = model.fit(X, Y)
pred_fwd = reg.predict(X) # predicted activity from real regressor
rev_X = np.concatenate([Xid, Xstim2, Xstim1, Xrule], axis = 1)
rev_X = (rev_X - np.mean(rev_X, axis = 0, keepdims = True)) / (1e-3 + np.std(rev_X, axis = 0, keepdims = True))
pred_bwd = reg.predict(rev_X) # predicted activity from reverse regressors

rule_cues = np.array(trials["rule_cue"])
vars_by_bin = [] # fwd/bwd stim by fwd/bwd regressors by neurons by bins
for irule, rule in enumerate([1.0, 2.0]):
    Ytest_rule = Ytest[rule_cues == rule, ...]
    new_vars = np.zeros((2, Ytest_rule.shape[1], Ytest_rule.shape[-1]))
    for ipred, pred in enumerate([pred_fwd, pred_bwd]):
        pred_rule = pred[rule_cues == rule, ...]
        for ibin in range(Ytest.shape[-1]):
            #new_vars[ipred, :, ibin] = var_explained(Ytest_rule[..., ibin], pred_rule)
            for n in range(Ytest_rule.shape[1]):
                new_vars[ipred, n, ibin] = pearsonr(Ytest_rule[:, n, ibin], pred_rule[:, n])[0]
    vars_by_bin.append(new_vars)

#%%
modulated_cells = ( sum_vars <= 0.10) & (sum_vars >= 0.05) # cells with unique variance explained by the stimuli
modulated_cells = ( sum_vars <= 1.0) & (sum_vars >= 0.05) & (avg_rates > 5) # cells with unique variance explained by the stimuli and some minimum firing rate

#fig, axs = plt.subplots(1,2, figsize = (8,4))
plt.figure(figsize = (5, 4))
plt.axvline(0.0, color = "k")
for i in range(2):
    #axs[i].plot(vars_by_bin[i][:, modulated_cells, :].mean(1).T)
    for j in range(2):
        plt.plot(times, vars_by_bin[i][j, modulated_cells, :].mean(0), color = plt.get_cmap("tab10")(j), ls = ["-", "--"][i], lw = 2, label = "rule="+["fwd", "bwd"][i]+", pred="+["fwd","bwd"][j])
plt.xlabel("time (s)")
plt.ylabel("predictive accuracy")
plt.legend()
plt.savefig("fwd_bwd_vars.pdf", bbox_inches = "tight")
plt.show()
plt.close()


# %% now we want to _regress out_ stim 1/2 during the swap period on fwd trials, and then find cells where the residuals are tuned on the bwd trials.
# swap is 0.1-0.3 after rule cue


swap_activity = get_binned_activity(trials, spikes, "rule_cue_on", 0.15, 0.25, 1, mode = "rates")[..., 0] # (trials, cells)
swap_avg_rates = swap_activity.mean(0)

Ymean, Ystd = np.mean(swap_activity, axis = 0, keepdims = True), np.std(swap_activity, axis = 0, keepdims = True) # mean/std across trials for each cell
Y = (swap_activity - Ymean) / (Ystd + 1e-3)
print(Y.shape)

X1, Y1 = X[rule_cues == 1.0, :], Y[rule_cues == 1.0, :]
X2, Y2, X2_m_stim1, X2_m_stim2 = [arr[rule_cues == 2.0, :] for arr in [X, Y, X_m_stim1, X_m_stim2]]

# define model
model = ElasticNet(alpha = 0.1, l1_ratio = 0.0, fit_intercept = True)

# fit on forward trials
reg = model.fit(X1, Y1)

# compute bwd residuals
Y2_res = Y2 - model.predict(X2)

# now we basically repeat the process from above on the Y2 residuals
reg = model.fit(X2, Y2_res)
pred_res = reg.predict(X) # predict on all trials
pred_res2 = reg.predict(X2) # just swap trials
tot_swap_vars = var_explained(Y2_res, pred_res2) # total variance explained by this model
partial_swap_vars = []
for partial_X in [X2_m_stim1, X2_m_stim2]: # fit each partial model
    partial_X = (partial_X - np.mean(partial_X, axis = 0, keepdims = True)) / (np.std(partial_X, axis = 0, keepdims = True) + 1e-3)
    partial_swap_vars.append(var_explained(Y2_res, model.fit(partial_X, Y2_res).predict(partial_X))) # quantify variance explained

uswapvars = tot_swap_vars[:, None] - np.array(partial_swap_vars).T # what is the unique variance explained by stim 1 and stim 2?
sum_swap_vars = np.sum(uswapvars, 1)
keep_cells = (swap_avg_rates > 5)
plt.figure()
plt.scatter(uswapvars[keep_cells, 0], uswapvars[keep_cells, 1], marker = ".", color = "k", s = 20)
plt.savefig("test_swap_uvars.pdf", bbox_inches = "tight") # clearly a lot of neurons really care about these stimuli
plt.show()
plt.close()

# %% and now compute variance explained over time by these as well


test_swap_activity = get_binned_activity(trials, spikes, "rule_cue_on", -0.25, 0.75, 10, mode = "rates") # 100 ms bins
Ytest_swap = (test_swap_activity - Ymean[..., None] ) / (Ystd[..., None]+1e-3) # trials by cells by bins

swap_vars_by_bin = [] # fwd/bwd stim by fwd/bwd regressors by neurons by bins
for irule, rule in enumerate([1.0, 2.0]):
    Ytest_rule = Ytest_swap[rule_cues == rule, ...]
    new_vars = np.zeros((Ytest_rule.shape[1], Ytest_rule.shape[-1]))
    pred_rule = pred_res[rule_cues == rule, ...]
    for ibin in range(Ytest_rule.shape[-1]):
        for n in range(Ytest_rule.shape[1]):
            new_vars[n, ibin] = pearsonr(Ytest_rule[:, n, ibin], pred_rule[:, n])[0]
    swap_vars_by_bin.append(new_vars)

modulated_swap_cells = (sum_swap_vars > 0.05) & (swap_avg_rates > 5)
plt.figure(figsize = (5, 4))
plt.axvline(0.0, color = "k")
for i in range(2): # fwd/bwd trials
    plt.plot(times, swap_vars_by_bin[i][modulated_swap_cells, :].mean(0), color = plt.get_cmap("tab10")(2), ls = ["-", "--"][i], lw = 2, label = "rule="+["fwd", "bwd"][i]+", pred=swap")
plt.xlabel("time (s)")
plt.ylabel("predictive accuracy")
plt.legend()
plt.savefig("fwd_bwd_swap_vars.pdf", bbox_inches = "tight")
plt.show()
plt.close()


#%% try to understand which cells are the same and which are different

keep_active = np.where((avg_rates > 5) | (swap_avg_rates > 5))[0]
all_uvars = np.concatenate([uvars, uswapvars], axis = 1)[keep_active, :]

thresh = 0.1
ntot = len(all_uvars)
fig, axs = plt.subplots(1, 6, figsize = (12,3))
nax = -1
for i in range(4):
    for j in range(i+1, 4):
        nax += 1
        v1, v2 = all_uvars[:, i], all_uvars[:, j]
        corr = pearsonr(v1, v2)[0]
        n1, n2 = np.sum(v1 > thresh), np.sum(v2 > thresh)
        n12, nb = np.sum((v1 > thresh) & (v2 > thresh)), n1*n2/ntot
        print(i, j, corr, n1, n2, n12, nb)
        axs[nax].scatter(v1, v2)
plt.show()
plt.close()




# %% now try to see if these neurons are connected!

dt = 10 / 1000 # s
env = lambda x: np.exp( -(0.8*x - 4.8) )+ 0.06

for i in range(4):
    for j in range(i+1,4): # for each pair of neuron groups
        pairs, fwds, bwds = [], [], []
        # find neurons tuned to each feature
        inds1, inds2 = keep_active[all_uvars[:, i] > thresh], keep_active[all_uvars[:, j] > thresh]

        nax += 1
        for n1 in inds1:
            for n2 in inds2:
                # for all pairs of unique neurons
                if (n1 not in inds2) and (n2 not in inds1):
                    #print(n1, n2)
                    count_fwd, count_bwd = 0,0
                    for trial in range(spikes.shape[0]): # for each trial

                        spk1, spk2 = spikes[trial, n1], spikes[trial, n2] # (spikes, )

                        # find all spike pairs that fall within 10 ms
                        count_fwd += (
                            np.searchsorted(spk2, spk1,      side="left")
                            - np.searchsorted(spk2, spk1-dt, side="left")
                        ).sum()

                        count_bwd += (
                            np.searchsorted(spk2, spk1+dt, side="right")
                            - np.searchsorted(spk2, spk1,  side="right")
                        ).sum()

                    if count_fwd + count_bwd > 100:
                        pairs.append((n1, n2)) # neuron pair
                        fwds.append(count_fwd) # spike2s within 10 ms after spike1
                        bwds.append(count_bwd) # spike2s within 10 ms before spike1

        fwds = np.array(fwds)
        bwds = np.array(bwds)

        asymms = np.log(fwds + 1) - np.log(bwds + 1)
        tot_counts = np.log(fwds + bwds + 1)

        abs_asymms = np.abs(asymms)
        putative_conns = np.where(abs_asymms > env(tot_counts))[0]

        # plot degree of asymmetry

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
        plt.title(f"$i$, $j$")
        #plt.savefig("test_asymms.pdf", bbox_inches = "tight")
        plt.show()
        plt.close()

        print(i, j, len(putative_conns))
        for conn in putative_conns:
            print(pairs[conn])


# %%
