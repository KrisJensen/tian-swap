
#%%

import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from load_data import get_data, get_binned_activity
#%%

trials, spikes = get_data()
num_trials, num_cells = spikes.shape

#%%

events = ["trial_start", "fixation_on", "stimulus_1_on", "stimulus_1_off","stimulus_2_on", "stimulus_2_off", "rule_cue_on", "rule_cue_off", "fixation_off"]
fix_on_t = trials["fixation_on"].values

event_boundaries = []
for i in range(len(events)-1):
    print("\n",events[i], events[i+1])
    e1t, e2t = trials[events[i]].values, trials[events[i+1]].values
    duration = e2t-e1t
    print(np.nanmean(duration), np.nanstd(duration), len(duration), np.sum(np.isnan(duration)), np.nanmin(duration), np.nanmax(duration))
    event_boundaries.append(np.nanmedian(e2t-fix_on_t))

trial_dur = trials["fixation_off"].values - trials["fixation_on"].values
print(np.nanmean(trial_dur), np.nanstd(trial_dur), len(trial_dur), np.sum(np.isnan(trial_dur)), np.nanmin(trial_dur), np.nanmax(trial_dur))

tot_dur = trials["fixation_off"].values[-1] - trials["trial_start"].values[0]
rec_dur = np.sum(trial_dur[~np.isnan(trial_dur)])

print(rec_dur / 60, tot_dur / 60)

# %% bin all our spikes from fixation_on to rule_cue_off + 500 ms -- which is approximately fixation_on + 2.5 seconds

spike_counts = get_binned_activity(trials, spikes, "fixation_on", 0.0, 2.5, 51, mode = "counts")

tot_counts_per_neuron = np.sum(spike_counts, axis = (0,-1))
print(tot_counts_per_neuron)
print(np.quantile(tot_counts_per_neuron, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
thresh = 1000
spike_counts = spike_counts[:, tot_counts_per_neuron > thresh, :]
print(spike_counts.shape)

#%% plot raster plot

reverse = trials["rule_cue"].values == 2
print(np.sum(reverse), np.sum(~reverse))

mean_counts = np.mean(spike_counts, axis = 0) # cells by time
mean, std = np.mean(mean_counts, axis = 1, keepdims=True), np.std(mean_counts, axis = 1, keepdims=True) # mean and std

fig, axs = plt.subplots(1, 2, figsize = (8, 4))
for icond, cond in enumerate([~reverse, reverse]):

    cond_counts = (spike_counts[cond, :, :].mean(0) - mean ) / std

    axs[icond].imshow(cond_counts, cmap = "coolwarm", aspect = "auto", extent = [0, 2.5, 0, cond_counts.shape[0]], vmin = np.quantile(cond_counts, 0.01), vmax = np.quantile(cond_counts, 0.99))
    for time in event_boundaries[:-1]:
        axs[icond].axvline(time, color = "k", linestyle = "--")
    axs[icond].set_title("Reverse" if icond == 1 else "Forward")
    axs[icond].set_xticks(event_boundaries[1:-1], events[2:-1], rotation = 45, ha = "right")
    axs[icond].set_ylabel("neuron")
plt.tight_layout()
plt.savefig(f"test_raster.pdf", bbox_inches = "tight")
plt.close()


#%% systematic computation of all cross-correlograms

asymms = []
counts = []
fwds, bwds = [], []
pairs = []
thresh = 10 # ms
dt = thresh / 1000 # s

for n1 in range(spikes.shape[1]):
    print(n1, "of", spikes.shape[1]-1)
    for n2 in range(spike_counts.shape[1]):
        if n1 != n2:
            count_fwd, count_bwd = 0,0
            for trial in range(spikes.shape[0]):

                spk1, spk2 = spikes[trial, n1], spikes[trial, n2] # (spikes, )

                count_fwd += (
                    np.searchsorted(spk2, spk1,      side="left")
                    - np.searchsorted(spk2, spk1-dt, side="left")
                ).sum()

                count_bwd += (
                    np.searchsorted(spk2, spk1+dt, side="right")
                    - np.searchsorted(spk2, spk1,  side="right")
                ).sum()

            if count_fwd + count_bwd > 100:
                pairs.append((n1, n2))
                fwds.append(count_fwd)
                bwds.append(count_bwd)

fwds = np.array(fwds)
bwds = np.array(bwds)

asymms = np.log(fwds + 1) - np.log(bwds + 1)
tot_counts = np.log(fwds + bwds + 1)

                
#%%

env = lambda x: np.exp( -(0.8*x - 4.8) )+ 0.06
abs_asymms = np.abs(asymms)

#putative_conns = np.where((asymms > 0.1) & (tot_counts > 8))[0]
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
plt.savefig("test_asymms.pdf", bbox_inches = "tight")
plt.close()

#%% plot example putative connection

def get_diffs(n1, n2, thresh = 50):

    spike_diffs = []
    for trial in range(spikes.shape[0]):
        spk1, spk2 = spikes[trial, n1], spikes[trial, n2]
        diffs = (spk1[:, None] - spk2[None, :]).flatten()*1000 # ms
        spike_diffs.append(diffs[np.abs(diffs) < thresh])
    return np.concatenate(spike_diffs)

for index in putative_conns:
    n1, n2 = pairs[index]
    print(n1, n2)
    spike_diffs = get_diffs(n1, n2, thresh = 50)
    plt.figure()
    plt.hist(spike_diffs, bins = np.linspace(-50, 50, 51))
    plt.title(f"n1: {n1}, n2: {n2}, asymm: {asymms[index]:.2f}, tot_counts: {tot_counts[index]:.2f}")
    plt.axvline(0, color = "k", linestyle = "-")
    plt.savefig(f"putative/spike_diffs_{n1}_{n2}.pdf", bbox_inches = "tight")
    plt.close()

    count_fwd, count_bwd = np.sum((spike_diffs > 0) & (spike_diffs <= 10)), np.sum((spike_diffs < 0) & (spike_diffs >= -10))
    print(n1, n2, count_fwd, count_bwd, np.log(count_fwd + 1) - np.log(count_bwd + 1), np.log(count_fwd + count_bwd + 1))

