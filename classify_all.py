
#%%
import numpy as np
from sklearn.linear_model import ElasticNet
from load_data import get_data, get_binned_activity
from utils import var_explained
from classify_utils import get_regressors, get_modulated_pairs

#%% load data
all_pairs, all_ns = [], []
for sesh in range(5, 39):
    print("sesh:", sesh)
    trials, spikes = get_data(sub = "subject_G", sesh = 5)
    num_trials, num_cells = spikes.shape
    rule_cues = np.array(trials["rule_cue"])
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

    # fit encodings models to this data
    model = ElasticNet(alpha = 0.1, l1_ratio = 0.0, fit_intercept = True)

    # first fit full model
    reg = model.fit(X, Y)
    pred = reg.predict(X)
    pred_bwd = reg.predict(X_bwd)

    # variance explained
    tot_var_explained = var_explained(Y, pred) # total variance explained by this model

    # now fit without stim1/stim2
    partial_vars, partial_vars_ctrl = [], []
    for partial_X in [X_m_stim1, X_m_stim2]: # fit each partial model
        partial_vars.append(var_explained(Y, model.fit(partial_X, Y).predict(partial_X))) # quantify variance explained

    uvars = tot_var_explained[:, None] - np.array(partial_vars).T # what is the unique variance explained by stim 1 and stim 2?

    # now we want to _regress out_ stim 1/2 during the swap period on fwd trials, and then find cells where the residuals are tuned on the bwd trials.

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

    # now try to see which neurons are connected!
    keep_active = np.where((avg_rates > 5) | (swap_avg_rates > 5))[0]
    all_uvars = np.concatenate([uvars, uswapvars], axis = 1)[keep_active, :]

    env = lambda x: np.exp( -(0.8*x - 4.8) )+ 0.06
    new_pairs, new_ns = get_modulated_pairs(spikes, all_uvars, env, keep_active, plot = False)
    all_pairs.append(new_pairs)
    all_ns.append(new_ns)

# %%
