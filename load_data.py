
import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet

#%%

def get_data(sub = "subject_G", sesh = 27):
    """load the data for one session of one subject"""
    base = f"./data/{sub}/session_{sesh:03d}/"

    trials = pd.read_csv(f"{base}trials.csv")

    spikes = loadmat(
        f"{base}spikes.mat",
        squeeze_me=False
    )["spikes"] # trials by cells of spikestimes

    is_length2 = np.where(trials["trial_length_code"] == 2)[0]
    trials = trials.iloc[is_length2]
    spikes = spikes[is_length2]
    num_trials, num_cells = spikes.shape

    for trial in range(num_trials):
        for cell in range(num_cells):
            spikes[trial, cell] = np.sort(spikes[trial, cell][:, 0]) # sort and remove zeros
            #assert(np.all(np.diff(spikes[trial, cell][:, 0]) >= 0)), f"spikes {trial}, {cell} not sorted"

    return trials, spikes


def get_binned_activity(trials, spikes, ref_event, t0, t1, bin_count, mode = "rates"):
    """
    get 'bin_count' bins of activity from t0 to t1 after 'ref_event' for each trial
    return either raw spikes or firing rates normalised by bin width.
    """
    assert mode in ["rates", "counts"]
    num_trials, num_cells = spikes.shape

    bins = np.linspace(t0, t1, bin_count+1)
    ref_times = trials[ref_event].values - trials["trial_start"].values

    activity = np.zeros((num_trials, num_cells, bin_count))
    for trial, ref in enumerate(ref_times):
        for cell in range(num_cells):
            spike_counts = np.histogram(spikes[trial, cell] - ref, bins=bins)[0]

            if mode == "counts":
                activity[trial, cell, :] = spike_counts
            elif mode == "rates":
                activity[trial, cell, :] = spike_counts / np.diff(bins)

    return activity
