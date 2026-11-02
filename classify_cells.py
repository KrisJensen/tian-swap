
#%%
import numpy as np
from sklearn.linear_model import ElasticNet
from load_data import get_data, get_binned_activity
from utils import var_explained
from scipy.stats import pearsonr
from classify_utils import get_regressors, plot_vars, plot_vars_by_time, plot_var_pairs, get_modulated_pairs

#%% load data

trials, spikes = get_data(sub = "subject_G", sesh = 26)
num_trials, num_cells = spikes.shape
rule_cues = np.array(trials["rule_cue"])


#%% first extract some 'training' activity

# get regressors
X = get_regressors(trials) # all regressors
X_bwd = get_regressors(trials, regressors=["target_2", "target_1", "rule_cue"]) # swap 12
X_m_stim1 = get_regressors(trials, regressors=["target_2", "rule_cue"]) # without 1
X_m_stim2 = get_regressors(trials, regressors=["target_1", "rule_cue"]) # without 2


# average activity from 100 to 500 ms after stimulus 2 offset (this is before the rule cue comes on)
activity = get_binned_activity(trials, spikes, "stimulus_2_off", 0.1, 0.5, 1, mode = "rates")[..., 0] # (trials, cells)
avg_rates = activity.mean(0)
Ymean, Ystd = np.mean(activity, axis = 0, keepdims = True), np.std(activity, axis = 0, keepdims = True) # mean/std across trials for each cell
Y = (activity - Ymean) / (Ystd + 1e-3)
print(Y.shape)
Yctrl = Y[np.random.choice(Y.shape[0], Y.shape[0], replace = False), :]

#%% fit encodings models to this data
model = ElasticNet(alpha = 0.1, l1_ratio = 0.0, fit_intercept = True)

# first fit full model
reg = model.fit(X, Y)
pred = reg.predict(X)
pred_bwd = reg.predict(X_bwd)

# variance explained
tot_var_explained = var_explained(Y, pred) # total variance explained by this model
ctrl_var_explained = var_explained(Yctrl, model.fit(X, Yctrl).predict(X)) # total variance explained after shuffling trials

fig, axs = plt.subplots(1, 2, figsize = (8,4))
axs[0].hist(tot_var_explained, bins = np.linspace(0, 1, 21))
axs[1].hist(ctrl_var_explained, bins = np.linspace(0, 1, 21))
plt.savefig("figs/tot_var.pdf")
plt.show()
plt.close()

# now fit without stim1/stim2
partial_vars, partial_vars_ctrl = [], []
for partial_X in [X_m_stim1, X_m_stim2]: # fit each partial model
    partial_vars.append(var_explained(Y, model.fit(partial_X, Y).predict(partial_X))) # quantify variance explained
    partial_vars_ctrl.append(var_explained(Yctrl, model.fit(partial_X, Yctrl).predict(partial_X))) # quantify ctrl variance explained

uvars = tot_var_explained[:, None] - np.array(partial_vars).T # what is the unique variance explained by stim 1 and stim 2?
uvars_ctrl = ctrl_var_explained[:, None] - np.array(partial_vars_ctrl).T # what is the unique variance explained by stim 1 and stim 2?

plot_vars(uvars, avg_rates, thresh = 5, fname = "vars", vars_ctrl = uvars_ctrl)



# %% now we want to _regress out_ stim 1/2 during the swap period on fwd trials, and then find cells where the residuals are tuned on the bwd trials.
# swap is 0.1-0.3 after rule cue


swap_activity = get_binned_activity(trials, spikes, "rule_cue_on", 0.15, 0.25, 1, mode = "rates")[..., 0] # (trials, cells)
swap_avg_rates = swap_activity.mean(0)
Ymean_swap, Ystd_swap = np.mean(swap_activity, axis = 0, keepdims = True), np.std(swap_activity, axis = 0, keepdims = True) # mean/std across trials for each cell
Y_swap = (swap_activity - Ymean_swap) / (Ystd_swap + 1e-3)


X1, Y1 = X[rule_cues == 1.0, :], Y_swap[rule_cues == 1.0, :]
X2, Y2, X2_m_stim1, X2_m_stim2 = [arr[rule_cues == 2.0, :] for arr in [X, Y_swap, X_m_stim1, X_m_stim2]]

# fit a model only on forward trials and compute residuals on backward trials
reg1 = model.fit(X1, Y1)
Y2_res = Y2 - reg1.predict(X2)

# now we basically repeat the process from above on the Y2 residuals
reg2 = model.fit(X2, Y2_res)
pred_res = reg2.predict(X) # predict on all trials
pred_res2 = reg2.predict(X2) # just swap trials
tot_swap_vars = var_explained(Y2_res, pred_res2) # total variance explained by this model
partial_swap_vars = []
for partial_X in [X2_m_stim1, X2_m_stim2]: # fit each partial model
    partial_swap_vars.append(var_explained(Y2_res, model.fit(partial_X, Y2_res).predict(partial_X))) # quantify variance explained

uswapvars = tot_swap_vars[:, None] - np.array(partial_swap_vars).T # what is the unique variance explained by stim 1 and stim 2?
plot_vars(uswapvars, swap_avg_rates, thresh = 5, fname = "swap_vars")


#%% now apply this model over time from 250 ms before rule cue on to 750 after rule cue on
# the question is whether the 'variance explained' by the stimulus cues will swap

tmin, tmax, num_bins = -0.25, 0.75, 10
test_activity = get_binned_activity(trials, spikes, "rule_cue_on", tmin, tmax, num_bins, mode = "rates") # 100 ms bins
Ymean_test, Ystd_test = np.mean(test_activity, axis = 0, keepdims = True), np.std(test_activity, axis = 0, keepdims = True) # mean/std across trials for each cell
Ytest = (test_activity - Ymean_test) / (Ystd_test+1e-3) # trials by cells by bins

times = np.linspace(tmin, tmax, num_bins+1)
times = times[1:]/2 + times[:-1]/2

#%% compute and plot variance explained by different predictors
vars_by_bin = [] # fwd/bwd stim by fwd/bwd/swp regressors by neurons by bins
for irule, rule in enumerate([1.0, 2.0]):
    Ytest_rule = Ytest[rule_cues == rule, ...]
    new_vars = np.zeros((3, Ytest_rule.shape[1], Ytest_rule.shape[-1]))
    for ipred, Ypred in enumerate([pred, pred_bwd, pred_res]):
        Ypred_rule = Ypred[rule_cues == rule, ...]
        for ibin in range(Ytest_rule.shape[-1]):
            for n in range(Ytest_rule.shape[1]):
                new_vars[ipred, n, ibin] = pearsonr(Ytest_rule[:, n, ibin], Ypred_rule[:, n])[0]
    vars_by_bin.append(new_vars)


modulated = (uvars.sum(1) >= 0.05) & (avg_rates > 5) # cells with unique variance explained by the stimuli and some minimum firing rate
modulated_swap = (uswapvars.sum(1) >= 0.05) & (swap_avg_rates > 5)

plot_vars_by_time(times, vars_by_bin, modulated, modulated_swap)


#%% try to understand which cells are the same and which are different

keep_active = np.where((avg_rates > 5) | (swap_avg_rates > 5))[0]
all_uvars = np.concatenate([uvars, uswapvars], axis = 1)[keep_active, :]
plot_var_pairs(all_uvars)

# %% now try to see which neurons are connected!

env = lambda x: np.exp( -(0.8*x - 4.8) )+ 0.06
get_modulated_pairs(spikes, all_uvars, env, keep_active)

# %%
