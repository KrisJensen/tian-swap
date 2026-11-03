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


def shuffle_spikes(trials, spikes, ref_event, t0, t1,
                   conditions = ("target_1", "target_2", "rule_cue"), seed = None):
    """Return original and trial-shuffled spike arrays, aligned to ref_event.

    Inputs follow get_data; both outputs retain spikes.shape with times in
    [t0, t1) seconds relative to ref_event. Each unit is independently permuted
    within conditions. Singleton groups remain unchanged. Assumes all trials
    fully cover the requested window; seed controls reproducibility.
    """
    # Align and crop before shuffling so variable event times do not smear responses.
    refs = trials[ref_event].values - trials["trial_start"].values
    aligned = np.empty_like(spikes)
    for trial, ref in enumerate(refs):
        for cell in range(spikes.shape[1]):
            times = spikes[trial, cell] - ref
            aligned[trial, cell] = times[(times >= t0) & (times < t1)]

    # Use positional row indices and a separate permutation for every unit.
    rng = np.random.default_rng(seed)
    shuffled = np.empty_like(aligned)
    for inds in trials.groupby(list(conditions), sort = False).indices.values():
        for cell in range(spikes.shape[1]):
            shuffled[inds, cell] = aligned[rng.permutation(inds), cell]
    return aligned, shuffled


def lag_count(pre, post, lag):
    """Count post-minus-pre lags in [lag[0], lag[1]) across sorted trial arrays."""
    count = 0
    for a, b in zip(pre, post):
        upper = np.searchsorted(b, a + lag[1], side = "left")
        lower = np.searchsorted(b, a + lag[0], side = "left")
        count += int((upper - lower).sum())
    return count


def jitter_null_counts(pre, post, t0, t1, lag, jitter, n_jitter, seed):
    """Sample interval-jitter pair counts for cropped per-trial spike arrays.

    Each target spike contributes the number of reference lag windows covering
    its uniform jittered position. Sample those counts directly, preserving the
    original null distribution, including overlapping reference windows.
    """
    # Split each trial at fixed jitter-bin boundaries and reference-window edges.
    n_bins = int(np.ceil((t1 - t0) / jitter))
    edges = np.minimum(t0 + np.arange(n_bins + 1) * jitter, t1)
    groups = {}
    for a, b in zip(pre, post):
        if not a.size or not b.size:
            continue
        starts = np.clip(a + lag[0], t0, t1)
        ends = np.clip(a + lag[1], t0, t1)
        cuts = np.unique(np.concatenate([edges, starts, ends]))
        lengths = np.diff(cuts)
        midpoints = cuts[:-1] + lengths / 2
        coverage = (np.searchsorted(starts, midpoints, side = "right") -
                    np.searchsorted(ends, midpoints, side = "right"))
        bins = np.clip(np.floor((midpoints - t0) / jitter).astype(int), 0, n_bins - 1)

        # Bin duration at coverage k gives the probability that a target adds k pairs.
        n_values = int(coverage.max()) + 1
        mass = np.bincount(bins * n_values + coverage, weights = lengths,
                           minlength = n_bins * n_values).reshape(n_bins, n_values)
        target_bins = np.clip(np.floor((b - t0) / jitter).astype(int), 0, n_bins - 1)
        counts = np.bincount(target_bins, minlength = n_bins)
        occupied = np.flatnonzero((counts > 0) & (mass[:, 1:].sum(axis = 1) > 0))

        # Pool only exactly identical distributions; no probability rounding is used.
        for index in occupied:
            probabilities = mass[index] / mass[index].sum()
            probabilities = probabilities[:np.flatnonzero(probabilities)[-1] + 1]
            key = tuple(probabilities)
            groups[key] = groups.get(key, 0) + int(counts[index])

    # Sum independent bin contributions without generating or sorting spike trains.
    rng = np.random.default_rng(seed)
    null = np.zeros(n_jitter, dtype = int)
    for key, count in groups.items():
        probabilities = np.array(key)
        support = np.flatnonzero(probabilities)
        if support.size == 1:
            null += count * support[0]
        elif support.size == 2:
            low, high = support
            null += count * low + (high - low) * rng.binomial(
                count, probabilities[high], size = n_jitter)
        else:
            null += rng.multinomial(count, probabilities, size = n_jitter) @ np.arange(len(key))
    return null


def jitter_connection(pre, post, t0, t1, lag = (.0005, .006),
                      jitter = .020, n_jitter = 999, alpha = .01, seed = 0):
    """Test a directed pair of sorted per-trial spike arrays, with times in seconds.

    Return uncorrected one-sided p-values, excitation/inhibition flags, counts,
    excess target spikes per reference spike, and the jitter-null counts.
    The reference train must contain a spike with its full lag window observed.
    """
    # Keep reference spikes whose entire positive-lag window is recorded.
    pre = [a[(a >= t0) & (a <= t1 - lag[1])] for a in pre]
    post = [b[(b >= t0) & (b < t1)] for b in post]
    n_pre = sum(map(len, pre))
    assert n_pre > 0

    # Sample the same interval-jitter null directly in count space.
    # Seeds remain reproducible, but draws differ from spike-time resampling.
    observed = lag_count(pre, post, lag)
    null = jitter_null_counts(pre, post, t0, t1, lag, jitter, n_jitter, seed)

    # Apply alpha independently to each tail without multiple-comparison correction.
    p_exc = float((1 + np.count_nonzero(null >= observed)) / (n_jitter + 1))
    p_inh = float((1 + np.count_nonzero(null <= observed)) / (n_jitter + 1))
    expected = float(null.mean())
    excess = observed - expected
    return {
        "excitatory": bool((p_exc <= alpha) and (excess > 0)),
        "inhibitory": bool((p_inh <= alpha) and (excess < 0)),
        "p_exc": p_exc, "p_inh": p_inh,
        "n_pre": n_pre, "observed": observed, "expected": expected,
        "excess_per_spike": excess / n_pre, "null": null,
    }


def get_jitter_pairs(spikes, vars, kept, t0, t1, var_thresh = .1, min_spikes = 100,
                     lag = (.0005, .006), jitter = .020, n_jitter = 999,
                     alpha = .01, seed = 0):
    """Test directed pairs in sorted trial-by-unit spikes using interval jitter.

    vars has one row per kept unit and one column per category. Return excitatory
    counts, inhibitory counts, rows [n1, n2, overlap, active_units, tested_pairs],
    and unique directed-pair results. Times are seconds relative to the same event.
    Pair IDs are zero-based column indices in spikes.
    """
    # Select units using marginal spike counts, which trial shuffling preserves.
    spike_counts = np.zeros(spikes.shape[1], dtype = int)
    reference_counts = np.zeros_like(spike_counts)
    for cell in kept:
        spike_counts[cell] = sum(np.count_nonzero((s >= t0) & (s < t1))
                                 for s in spikes[:, cell])
        reference_counts[cell] = sum(np.count_nonzero((s >= t0) & (s <= t1 - lag[1]))
                                     for s in spikes[:, cell])
    eligible = spike_counts >= min_spikes
    groups = [kept[vars[:, i] > var_thresh] for i in range(vars.shape[1])]

    # Preserve the existing exclusion of units belonging to both compared groups.
    category_pairs, ns = [], []
    for i, inds1 in enumerate(groups):
        for j, inds2 in enumerate(groups):
            if i != j:
                source = np.setdiff1d(inds1, inds2)
                target = np.setdiff1d(inds2, inds1)
                pairs = [(int(a), int(b)) for a in source for b in target
                         if eligible[a] and eligible[b] and reference_counts[a] > 0]
                category_pairs.append(pairs)
                ns.append([len(inds1), len(inds2), np.intersect1d(inds1, inds2).size,
                           vars.shape[0], len(pairs)])

    # Test each physical directed pair once and reuse its result across categories.
    unique_pairs = sorted({pair for pairs in category_pairs for pair in pairs})
    print("jitter pairs:", len(unique_pairs), flush = True)
    results, details = {}, []
    for index, (pre, post) in enumerate(unique_pairs):
        result = jitter_connection(
            spikes[:, pre], spikes[:, post], t0, t1, lag = lag, jitter = jitter,
            n_jitter = n_jitter, alpha = alpha,
            seed = np.random.SeedSequence([seed, pre, post]),
        )
        results[pre, post] = {key: value for key, value in result.items() if key != "null"}
        details.append(dict(pre = pre, post = post, **results[pre, post]))
        if (index + 1) % 100 == 0:
            print("jitter pair:", index + 1, "of", len(unique_pairs), flush = True)

    # Use the same tested-pair denominator for the excitatory and inhibitory calls.
    exc_counts = [sum(results[pair]["excitatory"] for pair in pairs)
                  for pairs in category_pairs]
    inh_counts = [sum(results[pair]["inhibitory"] for pair in pairs)
                  for pairs in category_pairs]
    return np.array(exc_counts), np.array(inh_counts), np.array(ns), details


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
    """Count connections between categories in vars using spikes and kept unit indices.

    Return counts and rows [n1, n2, overlap, active_units, tested_pairs],
    where tested pairs pass count_thresh and the existing overlap exclusion.
    """

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
                                count_bwd += (np.searchsorted(spk2, spk1, side="left") - np.searchsorted(spk2, spk1-dt, side="left")).sum()

                                count_fwd += (np.searchsorted(spk2, spk1+dt, side="right") - np.searchsorted(spk2, spk1,  side="right")).sum()

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
                all_ns[-1].append(len(pairs)) # pairs passing the spike-count threshold
                
    return np.array(all_pairs), np.array(all_ns)

#%%
