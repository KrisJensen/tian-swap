
#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from load_data import get_data, get_binned_activity
from utils import var_explained
from classify_utils import get_regressors, get_jitter_pairs, shuffle_spikes

#%% load data
# Keep the epoch and test parameters explicit; alpha is uncorrected for each tail.
subject = "subject_G"
t0, t1 = -1.7, 0.74
jitter_params = dict(var_thresh = 0.1, min_spikes = 100, lag = (0.0005, 0.006), jitter = 0.020, n_jitter = 201, alpha = 0.01, seed = 0)
shuffle_conditions = ("target_1", "target_2", "rule_cue")
shuffle_seed = 0

# Store separate counts and shared tested-pair denominators for both connection signs.
all_exc, all_inh, all_ns, all_exc_shuff, all_inh_shuff, all_ns_shuff = [[] for _ in range(6)]
all_connections = []
for sesh in range(5, 39):
    print("sesh:", sesh)
    trials, spikes = get_data(sub = subject, sesh = sesh)
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

    # Compare the same epoch in the original and condition-matched trial-shuffled data.
    epoch_spikes, shuffled_spikes = shuffle_spikes(
        trials, spikes, "rule_cue_on", t0, t1,
        conditions = shuffle_conditions, seed = shuffle_seed)

    # Use each dataset's own tested-pair counts and retain the individual unit-pair results.
    for dataset, data, exc_list, inh_list, ns_list in [
        ("real", epoch_spikes, all_exc, all_inh, all_ns),
        #("shuffled", shuffled_spikes, all_exc_shuff, all_inh_shuff, all_ns_shuff),
    ]:
        print("testing jitter connections:", dataset)
        exc, inh, new_ns, details = get_jitter_pairs(
            data, all_uvars, keep_active, t0, t1, **jitter_params)
        exc_list.append(exc)
        inh_list.append(inh)
        ns_list.append(new_ns)
        all_connections.extend(dict(row, subject = subject, session = sesh, dataset = dataset) for row in details)
        print(exc, inh, new_ns[:, -1])

# %%

# Retain session-level counts and a table of unit indices, uncorrected p-values, and effect sizes.
exc_epoch, inh_epoch, ns_epoch = np.array(all_exc), np.array(all_inh), np.array(all_ns)
exc_shuff, inh_shuff, ns_shuff = np.array(all_exc_shuff), np.array(all_inh_shuff), np.array(all_ns_shuff)
connections = pd.DataFrame(all_connections)

import pickle
pickle.dump([(exc_epoch, inh_epoch, ns_epoch), (exc_shuff, inh_shuff, ns_shuff)], open("./results.p", "rb"))

# Sum session-specific independence expectations before calculating overlap enrichment.
observed = ns_epoch[:, :, 2].sum(axis = 0) # true pairs
expected = (ns_epoch[:, :, 0] * ns_epoch[:, :, 1] / ns_epoch[:, :, 3]).sum(axis = 0) # expected in each session, then sum
enrichment = observed / expected # compute ratio
frac1, frac2, frac3, frac4 = enrichment[[0, 8, 2, 4]] # X1/X2, S1/S2, X1/S2, X2/S1

print(frac1, frac2, frac3, frac4)

# %%


# Pool detected and tested pairs before reporting separate excitatory and inhibitory percentages.
summaries = []
for dataset, exc, inh, ns in [("real", exc_epoch, inh_epoch, ns_epoch),
                             ("shuffled", exc_shuff, inh_shuff, ns_shuff)]:
    tested = ns[:, :, 4].sum(axis = 0)
    for kind, counts in [("excitatory", exc), ("inhibitory", inh)]:
        pairtots = counts.sum(axis = 0)
        summary = dict(dataset = dataset, kind = kind)
        for name, indices in [("predicted", [1, 5]), ("pred_rev", [6, 10]),
                              ("control", [2, 4]), ("ctrl_rev", [7, 9]),
                              ("within", [0, 3, 8, 11])]:
            denominator = tested[indices].sum()
            summary[name] = 100 * pairtots[indices].sum() / denominator if denominator else np.nan
        summaries.append(summary)
connection_rates = pd.DataFrame(summaries)
print("Putative connection percentages (uncorrected):")
print(connection_rates.to_string(index = False))

# %%
