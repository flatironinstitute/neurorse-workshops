---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Group Project: Analyzing hippocampal place cells with Pynapple and NeMoS

<div class="render-all">
    
In this tutorial we will learn how to use more advanced applications of pynapple: signal processing and decoding. We'll apply these methods to demonstrate and visualize some well-known physiological properties of hippocampal activity, specifically phase presession of place cells and sequential coordination of place cell activity during theta oscillations.

</div>

## Objectives

<div class="render-all">
    
For part 1 of this notebook, we will be using Pynapple to achieve the following objectives:
1. Load in and get a feel for the data set   
2. Identify and extract theta oscillations in the LFP
3. Identify place cells using 1D tuning curves
4. Visualize phase precession using 2D tuning curves
5. Use Baysian decoding to reconstruct spatial sequences from population activity

For part 2, we will by applying NeMoS to explore the dataset further by:
1. Visualize speed vs. position encoding
2. Create a design matrix using a basis set to simplify speed and position parameter space
3. Fit a Poisson GLM to neural activity with speed and position as predictors
4. Evaluate the model's predicted tuning curves and compare to the real data

</div>

```{code-cell} ipython3
:tags: [render-all]

# suppress warnings
import warnings
warnings.simplefilter("ignore")

# imports
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy import signal
import seaborn as sns
import tqdm
import pynapple as nap
import workshop_utils

# necessary for animation
import nemos as nmo
plt.style.use(nmo.styles.plot_style)
```

## Part 1: Using Pynapple to identify phase precession and hippocampal sequences
### Fetching the data

<div class="render-all">  
    
The data set we'll be looking at is from the manuscript [Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences](https://www.science.org/doi/10.1126/science.aad1935). In this study, the authors collected electrophisiology data in rats across multiple sites in layer CA1 of hippocampus to extract the LFP alongside spiking activity of many simultaneous pyramidal units. In each recording session, data were collected while the rats explored a novel environment (a linear or circular track), as well as during sleep before and after exploration. In our following analyses, we'll focus on the exploration period of a single rat and recording session.

The full dataset for this study can be accessed on [DANDI](https://dandiarchive.org/dandiset/000044/0.210812.1516). Since the file size of a recording session can be large from the LFP saved for each recorded channel, we'll use a smaller file that contains the spiking activity and the LFP from a single, representative channel, which is hosted on [OSF](https://osf.io/2dfvp). This smaller file, like the original data, is saved as an [NWB](https://www.nwb.org) file.

If you ran the workshop setup script, you should have this file downloaded already. If not, the function we'll use to fetch it will download it for you. This function is called `fetch_data`, and can be imported from the `workshop_utils` module. This function will give us the file path to where the data is stored. We can then use the pynapple function `load_file` to load in the data, which is able to handle the NWB file type.

</div>

```{code-cell} ipython3
:tags: [render-all]

# fetch file path
path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
# load data with pynapple
data = nap.load_file(path)
print(data)
```

<div class="render-all">  
    
This returns a dictionary of pynapple objects that have been extracted from the NWB file. Let's explore each of these objects.

</div>

:::{admonition} Note
:class: note render-all
We will ignore the object `theta_phase` because we will be computing this ourselves later on in the exercise.
:::


#### units

<div class="render-all">  
    
The `units` field is a [`TsGroup`](pynapple.TsGroup): a collection of [`Ts`](pynapple.Ts) objects containing the spike times of each unit, where the "Index" is the unit number or key. Each unit has the following metadata:
- **rate**: computed by pynapple, is the average firing rate of the neuron across all recorded time points.
- **location**, **shank**, and **cell_type**: variables saved and imported from the original data set.

</div>

```{code-cell} ipython3
:tags: [render-all]

data["units"]
```

<div class="render-all">  

We can access the spike times of a single unit by indexing the `TsGroup` by its unit number. For example, to access the spike times of unit 1:

</div>

```{code-cell} ipython3
:tags: [render-all]

data["units"][1]
```

#### rem, nrem, and forward_ep

<div class="render-all">  

The next three objects; `rem`, `nrem`, and `forward_ep`; are all [`IntervalSet`](pynapple.IntervalSet) objects containing time windows of REM sleep, nREM sleep, and forward runs down the linear maze, respectively. 

</div>

```{code-cell} ipython3
:tags: [render-all]

data["rem"]
```

```{code-cell} ipython3
:tags: [render-all]

data["nrem"]
```

```{code-cell} ipython3
:tags: [render-all]

data["forward_ep"]
```

<div class="render-all"> 

All intervals in `forward_ep` occur in the middle of the session, while `rem` and `nrem` both contain sleep epochs that occur before and after exploration. 
    
The following plot demonstrates how each of these labelled epochs are organized across the session.

</div>

```{code-cell} ipython3
:tags: [render-all]

t_start = data["nrem"].start[0]
fig,ax = plt.subplots(figsize=(10,2), constrained_layout=True)
sp1 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="blue", alpha=0.1) for iset in data["rem"]];
sp2 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="green", alpha=0.1) for iset in data["nrem"]];
sp3 = [ax.axvspan((iset.start[0]-t_start)/60, (iset.end[0]-t_start)/60, color="red", alpha=0.1) for iset in data["forward_ep"]];
ax.set(xlabel="Time within session (minutes)", title="Labelled time intervals across session", yticks=[])
ax.legend([sp1[0],sp2[0],sp3[0]], ["REM sleep","nREM sleep","forward runs"]);
```

#### eeg

<div class="render-all">  

The `eeg` object is a [`TsdFrame`](pynapple.TsdFrame) containing an LFP voltage trace for a single representative channel in CA1.

</div>

```{code-cell} ipython3
:tags: [render-all]

data["eeg"]
```

<div class="render-all">  

Despite having a single column, this [`TsdFrame`](pynapple.TsdFrame) is still a 2D object. We can represent this as a 1D `Tsd` by indexing into the first column.

</div>

```{code-cell} ipython3
:tags: [render-all]

data["eeg"][:,0]
```

#### position

<div class="render-all">  

The final object, `position`, is a [`Tsd`](pynapple.Tsd) containing the linearized position of the animal, in centimeters, recorded during the exploration window.

</div>

```{code-cell} ipython3
:tags: [render-all]

data["position"]
```

<div class="render-all">  

Positions that are not defined, i.e. when the animal is at rest, are filled with `NaN`.

This object additionally contains a [`time_support`](pynapple.Tsd.time_support) attribute, which gives the time interval during which positions are recorded (including points recorded as `NaN`).

</div>

```{code-cell} ipython3
:tags: [render-all]

data["position"].time_support
```

<div class="render-all">  

Let's visualize the first 300 seconds of position data and overlay `forward_ep` intervals.

</div>

```{code-cell} ipython3
:tags: [render-all]

pos_start = data["position"].time_support.start[0]
fig, ax = plt.subplots(figsize=(10,3))
l1 = ax.plot(data["position"])
l2 = [ax.axvspan(iset.start[0], iset.end[0], color="red", alpha=0.1) for iset in data["forward_ep"]];
ax.set(xlim=[pos_start,pos_start+300], ylabel="Position (cm)", xlabel="Time (s)", title="Tracked position along linear maze")
ax.legend([l1[0], l2[0]], ["animal position", "forward run epochs"])
```

<div class="render-all"> 

This plot confirms that positions are only recorded while the animal is moving along the track. Additionally, it is clear that the intervals in `forward_ep` capture only perios when the animal's position is increasing, during forward runs.

We'll save out the following variables that we'll need throughout the notebook.

</div>

```{code-cell} ipython3
:tags: [render-all]

position = data["position"]
lfp = data["eeg"][:,0]
spikes = data["units"]
forward_ep = data["forward_ep"]
```

### Restricting and visualizing the data

<div class="render-all"> 

For the following exercises, we'll only focus on periods when the animal is awake and running. We can get this information from `position`.

</div>

#### 1. Save out the time support of position, giving us the epoch during which the animal is awake.

<div class="render-user">
```{code-cell} ipython3
awake_ep = 
```
</div>

```{code-cell} ipython3
awake_ep = position.time_support
awake_ep
```

<div class="render-all">

You may have noticed many `nan` values for position during the awake period; these values correspond to when the animals is at rest. We also want, then, epochs describing periods when the animal is running. Some of this information is saved already in `forward_ep`. 

</div>

#### 2. Confirm that when restricting position to `forward_ep`, there are no `nan` values in position.

<div class="render-user">
```{code-cell} ipython3
# restrict position and check for nans
```
</div>

```{code-cell} ipython3
# restrict position and check for nans
np.any(np.isnan(position.restrict(forward_ep)))
```

<div class="render-all">

What if we want *all* movement epochs, not just forward runs? We can derive this from `position` by dropping all `nan` values and recomputing the time support. 

</div>

#### 3. Extract time intervals from `position` using the [`dropna`](pynapple.Tsd.dropna) and [`find_support`](pynapple.Tsd.find_support) methods.

<div class="render-all">

- The first input argument, `min_gap`, sets the minumum separation between adjacent intervals in order to be split
- Here, use `min_gap` of 1 s

</div>

<div class="render-user">
```{code-cell} ipython3
# drop nan values
# save all run epochs in the following variable
run_ep = 
```
</div>

```{code-cell} ipython3
# drop nan values
pos_good = data["position"].dropna()
run_ep = pos_good.find_support(1)
run_ep
```

<div class="render-all">

Finally, we can use `run_ep` and `forward_ep` to extract epochs when the animal is running backwards.

</div>

#### 4. Use the [`IntervalSet`](pynapple.IntervalSet) method [`set_diff`](pynapple.IntervalSet.set_diff) to get `backward_ep` from `run_ep` and `forward_ep`

<div class="render-user">
```{code-cell} ipython3
backward_ep = 
```
</div>

```{code-cell} ipython3
backward_ep = run_ep.set_diff(forward_ep)
backward_ep
```

<div class="render-all">
    
Now, when extracting the LFP, spikes, and position, we can use `restrict()` with any of these epochs to restrict the data to our movement period of interest.

To get a sense of what the LFP looks like while the animal runs down the linear track, we can plot each variable, `lfp_run` and `position`, side-by-side. Let's do this for an example run; specifically, we'll look at forward run 9.

</div>

#### 5. Create an interval set for forward run 9, adding 2 seconds to the end of the interval. Restrict LFP and position to this epoch.

<div class="render-user">
```{code-cell} ipython3
ex_run_ep =
ex_lfp_run = 
ex_position = 
```
</div>

```{code-cell} ipython3
ex_ep = nap.IntervalSet(start=forward_ep[9].start, end=forward_ep[9].end+2)
ex_lfp = lfp.restrict(ex_ep)
ex_position = position.restrict(ex_ep)
```

<div class="render-all">

Let's plot the example LFP trace and anmimal position. Plotting `Tsd` objects will automatically put time on the x-axis.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, constrained_layout=True, figsize=(10, 4), sharex=True)

# plot LFP
axs[0].plot(ex_lfp)
axs[0].set_title("Local Field Potential on Linear Track")
axs[0].set_ylabel("LFP (a.u.)")

# plot animal's position
axs[1].plot(ex_position)
axs[1].set_title("Animal Position on Linear Track")
axs[1].set_ylabel("Position (cm)") # LOOK UP UNITS
axs[1].set_xlabel("Time (s)");
```

<div class="render-all"> 

As we would expect, there is a strong theta oscillation dominating the LFP while the animal runs down the track. This oscillation is weaker after the run is complete.

</div>

### Getting the Wavelet Decomposition

<div class="render-all">

To illustrate this further, we'll perform a wavelet decomposition on the LFP trace during this run. We can do this in pynapple using the function [`nap.compute_wavelet_transform`](pynapple.process.wavelets.compute_wavelet_transform). This function takes the following inputs (in order):
- `sig`: the input signal; a `Tsd`, a `TsdFrame`, or a `TsdTensor`
- `freqs`: a 1D array of frequency values to decompose

We will also supply the following optional arguments:
- `fs`: the sampling rate of `sig`

A [continuous wavelet transform](https://en.wikipedia.org/wiki/Continuous_wavelet_transform) decomposes a signal into a set of [wavelets](https://en.wikipedia.org/wiki/Wavelet), in this case [Morlet wavelets](https://en.wikipedia.org/wiki/Morlet_wavelet), that span both frequency and time. You can think of the wavelet transform as a cross-correlation between the signal and each wavelet, giving the similarity between the signal and various frequency components at each time point of the signal. Similar to a Fourier transform, this gives us an estimate of what frequencies are dominating a signal. Unlike the Fourier tranform, however, the wavelet transform gives us this estimate as a function of time.

We must define the frequency set that we'd like to use for our decomposition. We can do this with the numpy function [`np.geomspace`](numpy.geomspace), which returns numbers evenly spaced on a log scale. We pass the lower frequency, the upper frequency, and number of samples as positional arguments.

</div>

#### 6. Define 100 log-spaced samples between 5 and 200 Hz using [`np.geomspace`](numpy.geomspace)

<div class="render-user">
```{code-cell} ipython3
# 100 log-spaced samples between 5Hz and 200Hz
freqs = 
```
</div>

```{code-cell} ipython3
# 100 log-spaced samples between 5Hz and 200Hz
freqs = np.geomspace(5, 200, 100)
```

<div class="render-all">

We can now compute the wavelet transform on our LFP data during the example run using [`nap.compute_wavelet_transform`](pynapple.process.wavelets.compute_wavelet_transform) by passing both `ex_lfp_run` and `freqs`. We'll also pass the optional argument `fs`, which is known to be 1250Hz from the study methods.

</div>

#### 7. Compute the wavelet transform, supplying the known sampling rate of 1250 Hz.

<div class="render-user">  
```{code-cell} ipython3
sample_rate = 1250
ex_cwt =
```
</div>

```{code-cell} ipython3
sample_rate = 1250
ex_cwt = nap.compute_wavelet_transform(ex_lfp, freqs, fs=sample_rate)
```

:::{admonition} Note
:class: tip render-all
If `fs` is not provided, it can be inferred from the time series [`rate`](pynapple.Tsd.rate) attribute, e.g. `ex_lfp.rate`. However, while inferred rate is close to the true sampling rate, it can introduce a small floating-point error. Therefore, it is better to supply the true sampling rate when it is known.
:::

<div class="render-all">
    
We can visualize the results by plotting a heat map of the calculated wavelet scalogram.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, figsize=(10,4), constrained_layout=True, height_ratios=[1.0, 0.3], sharex=True)
fig.suptitle("Wavelet Decomposition")

amp = np.abs(ex_cwt.values)
cax = axs[0].pcolormesh(ex_cwt.t, freqs, amp.T)
axs[0].set(ylabel="Frequency (Hz)", yscale='log', yticks=freqs[::10], yticklabels=np.rint(freqs[::10]));
axs[0].minorticks_off()
fig.colorbar(cax,label="Amplitude")

p1 = axs[1].plot(ex_lfp)
axs[1].set(ylabel="LFP (a.u.)", xlabel="Time(s)")
axs[1].margins(0)
ax = axs[1].twinx()
p2 = ax.plot(ex_position, color="orange")
ax.set_ylabel("Position (cm)")
ax.legend([p1[0], p2[0]],["raw LFP","animal position"])
```

<div class="render-all">
    
You should see a strong presence of theta in the 6-12Hz frequency band while the animal runs down the track, which dampens during rest.

</div>

### Filtering for theta

<div class="render-all">

For the remaining exercises, we'll reduce our example epoch to the portion when the animal is running forward along the linear track.

</div>

#### 8. Restrict the LFP and position to epochs when the animal is running forward, and create a new [`IntervalSet`](pynapple.IntervalSet) for forward run 9 with no padding.

<div class="render-user">  
```{code-cell} ipython3
lfp = 
position = 
ex_run_ep =
```
</div>

```{code-cell} ipython3
lfp = lfp.restrict(forward_ep)
position = position.restrict(forward_ep)
ex_run_ep = nap.IntervalSet(start=forward_ep[9].start, end=forward_ep[9].end)
```

<div class="render-all">

We can extract the theta oscillation by applying a bandpass filter on the raw LFP. To do this, we use the pynapple function [`nap.apply_bandpass_filter`](pynapple.process.filtering.apply_bandpass_filter), which takes the the following arguments:
- `data`: the signal to be filtered; a [`Tsd`](pynapple.Tsd), [`TsdFrame`](pynapple.TsdFrame), or [`TsdTensor`](pynapple.TsdTensor)
- `cutoff`: tuple containing the frequency cutoffs, (lower frequency, upper frequency)

Conveniently, this function will recognize and handle splits in the epoched data (i.e. applying the filtering separately to discontinuous epochs), so we don't have to worry about passing signals that have been split in time.

Same as before, we'll pass the optional argument:
- `fs`: the sampling rate of `data` in Hz

</div>

#### 9. Using [`nap.apply_bandpass_filter`](pynapple.process.filtering.apply_bandpass_filter), filter the LFP for theta within a 6-12 Hz range.

<div class="render-user">   
```{code-cell} ipython3
theta_band = 
```
</div>

```{code-cell} ipython3
theta_band = nap.apply_bandpass_filter(lfp, (6.0, 12.0), fs=sample_rate)
```

<div class="render-all">

We can visualize the output by plotting the filtered signal with the original signal.

</div>

```{code-cell} ipython3
:tags: [render-all]

plt.figure(constrained_layout=True, figsize=(10, 3))
plt.plot(lfp.restrict(ex_run_ep), label="raw")
plt.plot(theta_band.restrict(ex_run_ep), label="filtered")
plt.xlabel("Time (s)")
plt.ylabel("LFP (a.u.)")
plt.title("Bandpass filter for theta oscillations (6-12 Hz)")
plt.legend();
```

### Computing theta phase

<div class="render-all">

In order to examine phase precession in place cells, we need to extract the phase of theta from the filtered signal. We can do this by taking the angle of the [Hilbert transform](https://en.wikipedia.org/wiki/Hilbert_transform).

#### 10. Use `scipy.signal.hilbert` to perform the Hilbert transform, and  the numpy function `np.angle` to extract the angle. Convert the output angle to a [0, 2pi] range, and store the result in a `Tsd` object.
- TIP: don't forget to pass the time support!
  
</div>

<div class="render-user"> 
```{code-cell} ipython3
# save the Tsd in the following variable
theta_phase = 
```
</div>

```{code-cell} ipython3
phase = np.angle(signal.hilbert(theta_band)) # compute phase with hilbert transform
phase[phase < 0] += 2 * np.pi # wrap to [0,2pi]
theta_phase = nap.Tsd(t=theta_band.t, d=phase, time_support=theta_band.time_support)
theta_phase
```

<div class="render-all">

Let's plot the phase on top of the filtered LFP signal, zooming in on a few cycles.

</div>

```{code-cell} ipython3
:tags: [render-all]


ex_run_shorter = nap.IntervalSet(ex_run_ep.start[0], ex_run_ep.start[0]+0.5)
fig,axs = plt.subplots(2,1,figsize=(10,4), constrained_layout=True, sharex=True)#, height_ratios=[2,1])
ax = axs[0]
ax.plot(lfp.restrict(ex_run_shorter))
ax.set_ylabel("LFP (a.u.)")
ax = axs[1]
p1 = ax.plot(theta_phase.restrict(ex_run_shorter), color='r')
ax.set_ylabel("Phase (rad)")
ax.set_xlabel("Time (s)")
ax = ax.twinx()
p2 = ax.plot(theta_band.restrict(ex_run_shorter))
ax.set_ylabel("Filtered LFP (a.u.)")
ax.legend([p1[0],p2[0]],["theta phase","filtered LFP"])
```

<div class="render-all">

We can see that cycle "resets" (i.e. goes from $2\pi$ to $0$) at peaks of the theta oscillation.

</div>

### Computing 1D tuning curves: place fields

<div class="render-all">

In order to identify phase precession in single units, we need to know their place selectivity. We can find place firing preferences of each unit by using the function [`nap.compute_tuning_curves`](pynapple.process.tuning_curves.compute_tuning_curves) This function has the following required inputs:
- `data`: a pynapple object containing the data for which tuning curves will be computed, either spike times (`Ts` and `TsGroup`) or continuous data (e.g. calcium transients, `Tsd` or `TsdFrame`)
- `feature`: a `Tsd` or `TsdFrame` of the feature(s) over which tuning curves are computed (e.g. position)
- `bins`: similar to the argument for `np.histogram`, this variable can either be the number of bins or the bin edges. For multiple features, you can specify `bins` by providing a list with length equal to the number of features.

First, we'll filter for units that fire at least 1 Hz and at most 10 Hz when the animal is running forward along the linear track. This will select for units that are active during our window of interest and eliminate putative interneurons (i.e. fast-firing inhibitory neurons that don't usually have place selectivity). Afterwards, we'll compute the tuning curves for these sub-selected units over position.

#### 11. Restrict `spikes` to `forward_ep` and select for units whose rate is at least 1 Hz and at most 10 Hz

</div>


<div class="render-user">
```{code-cell} ipython3
# save the filtered spikes in the following variable
good_spikes = 
```
</div>

```{code-cell} ipython3
good_spikes = spikes[(spikes.restrict(forward_ep).rate >= 1) & (spikes.restrict(forward_ep).rate <= 10)]
```

#### 12. Compute tuning curves for units in `good_spikes` with respect to forward running position, using 50 position bins.

<div class="render-user">
```{code-cell} ipython3
place_fields = 
```
</div>

```{code-cell} ipython3
place_fields = nap.compute_tuning_curves(good_spikes, position, 50, feature_names=["position"])
```

<div class="render-all">

This function returns tuning curves as an `xarray.DataArray`, with coordinates for unit (first dimension) and position (second dimension). An `xarray.DataArray` object provides convenient tools for plotting and other manipulations, and it scales well for tuning curves with more than 1 feature. 

</div>

:::{admonition} Tip
:class: tip render-all

The reason [`nap.compute_tuning_curves`](pynapple.process.tuning_curves.compute_tuning_curves) returns a `xarray.DataArray` and not a Pynapple object is because the array elements no longer correspond to *time*, which Pynapple objects require.
:::

<div class="render-all">

We can use the `xarray.DataArray` `plot` method to easily plot each unit.

</div>

```{code-cell} ipython3
:tags: [render-all]

from scipy.ndimage import gaussian_filter1d

# smooth the place fields so they look nice
place_fields.data = gaussian_filter1d(place_fields.data, 1, axis=-1)

p = place_fields.plot(x="position", col="unit", col_wrap=5, size=1.2)
p.set_ylabels("firing rate (Hz)")
```

<div class="render-all">
    
We can see spatial selectivity in each of the units; across the population, we have firing fields tiling the entire linear track. 

</div>

### Visualizing phase precession within a single unit

<div class="render-all">
    
As an initial visualization of phase precession, we'll look at a single traversal of the linear track. First, let's look at how the timing of an example unit's spikes lines up with the LFP and theta. To plot the spike times on the same axis as the LFP, we'll use the pynapple object's method [`value_from`](pynapple.TsGroup.value_from) to align the spike times with the theta amplitude. For our spiking data, this will find the amplitude closest in time to each spike. Let's start by applying [`value_from`](pynapple.TsGroup.value_from) on unit 177, who's place field is cenetered on the linear track, using `theta_band` to align the amplityde of the filtered LFP.

#### 13. Use the pynapple object method [`value_from`](pynapple.TsGroup.value_from) to find the value of `theta_band` corresponding to each spike time from unit 177.

</div>

<div class="render-user">  
```{code-cell} ipython3
unit = 177
spike_theta = 
```
</div>

```{code-cell} ipython3
unit = 177
spike_theta = spikes[unit].value_from(theta_band)
```

<div class="render-all">

Let's plot `spike_theta` on top of the LFP and filtered theta, as well as visualize the animal's position along the track.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(2, 1, figsize=(10,4), constrained_layout=True, sharex=True)
axs[0].plot(lfp.restrict(ex_run_ep), alpha=0.5, label="raw LFP")
axs[0].plot(theta_band.restrict(ex_run_ep), color="slateblue", label="filtered theta")
axs[0].plot(spike_theta.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[0].set(ylabel="LFP (a.u.)")
axs[0].legend()

axs[1].plot(ex_position, '--', color="green", label="animal position")
axs[1].plot(ex_position[(ex_position > 50).values & (ex_position < 130).values], color="green", lw=3, label="place field bounds")
axs[1].set(ylabel="Position (cm)", xlabel="Time (s)")
axs[1].legend()
```

<div class="render-all">
    
As the animal runs through unit 177's place field (thick green), the unit spikes (orange dots) at specific points along the theta cycle dependent on position: starting at the rising edge, moving towards the trough, and ending at the falling edge.

We can exemplify this pattern by plotting the spike times aligned to the phase of theta. We'll want the corresponding phase of theta at which the unit fires as the animal is running down the track, which we can again compute using the method [`value_from`](pynapple.TsGroup.value_from). 

</div>

#### 14. Compute the value of `theta_phase` corresponding to each spike time from unit 177.

<div class="render-user">  
```{code-cell} ipython3
spike_phase = 
```
</div>

```{code-cell} ipython3
spike_phase = spikes[unit].value_from(theta_phase)
```

<div class="render-all">

To visualize the results, we'll recreate the plot above, but instead with the theta phase.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(3, 1, figsize=(10,6), constrained_layout=True, sharex=True)
axs[0].plot(theta_band.restrict(ex_run_ep), color="slateblue", label="filtered theta")
axs[0].plot(spike_theta.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[0].set(ylabel="LFP (a.u.)", title="Spike times relative to filtered theta")
axs[0].legend()

axs[1].plot(theta_phase.restrict(ex_run_ep), color="slateblue", label="theta phase")
axs[1].plot(spike_phase.restrict(ex_run_ep), 'o', color="orange", label="spike times")
axs[1].set(ylabel="Phase (rad)", title="Spike times relative to theta phase")
axs[1].legend()

axs[2].plot(ex_position, '--', color="green", label="animal position")
axs[2].plot(ex_position[(ex_position > 50).values & (ex_position < 130).values], color="green", lw=3, label="place field bounds")
axs[2].set(ylabel="Position (cm)", xlabel="Time (s)", title="Animal position")
axs[2].legend()
```

<div class="render-all">
    
We now see a negative trend in the spike phase as the animal moves through unit 177's place field. This phemomena is known as phase precession: the phase at which a unit spikes *precesses* (gets earlier) as the animal runs through that unit's place field. Explicitly, that unit will spike at *late* phases of theta (higher radians) in *earlier* positions in the field, and fire at *early* phases of theta (lower radians) in *late* positions in the field.

We can observe this phenomena on average across the session by relating the spike phase to the spike position. 

</div>

#### 15. Compute the position corresponding to each spike for example unit 177.

<div class="render-user">
```{code-cell} ipython3
spike_position = 
```
</div>

```{code-cell} ipython3
spike_position = spikes[unit].value_from(position)
```

<div class="render-all">

Now we can plot the spike phase against the spike position in a scatter plot.

</div>

```{code-cell} ipython3
:tags: [render-all]

plt.subplots(figsize=(5,3))
plt.plot(spike_position, spike_phase, 'o')
plt.ylabel("Phase (rad)")
plt.xlabel("Position (cm)")
```

<div class="render-all">
    
Similar to what we saw in a single run, there is a negative relationship between theta phase and field position, characteristic of phase precession.

</div>

### Computing 2D tuning curves: position vs. phase

<div class="render-all">

The scatter plot above can be similarly be represented as a 2D tuning curve over position and phase. We can compute this using the same function, [`nap.compute_tuning_curves`](pynapple.process.tuning_curves.compute_tuning_curves), but now passing second input, `features`, as a 2-column `TsdFrame` containing the two target features.

To do this, we'll need to combine `position` and `theta_phase` into a `TsdFrame`. For this to work, both variables must have the same length. We can achieve this by upsampling `position` to the length of `theta_phase` using the pynapple object method [`interpolate`](pynapple.Tsd.interpolate). This method will linearly interpolate new position samples between existing position samples at timestamps given by another pynapple object, in our case by `theta_phase`. Once they're the same length, they can be combined into a single `TsdFrame` and used to compute 2D tuning curves.

</div>

#### 16. Interpolate `position` to the time points of `theta_phase`.

<div class="render-user"> 
```{code-cell} ipython3
upsampled_pos = 
```
</div>

```{code-cell} ipython3
upsampled_pos = position.interpolate(theta_phase)
```

<div class="render-all">

Let's visualize the results of the interpolation.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,axs = plt.subplots(2,1,constrained_layout=True,sharex=True,figsize=(10,4))
axs[0].plot(position.restrict(ex_run_ep),'.')
axs[0].set(ylabel="Position (cm)", title="Original position points")
axs[1].plot(upsampled_pos.restrict(ex_run_ep),'.')
axs[1].set(ylabel="Position (cm)", xlabel="Time (s)", title="Upsampled position points")
```

#### 17. Stack `upsampled_pos` and `theta_phase` together into a single `TsdFrame`

<div class="render-user">  
```{code-cell} ipython3
# store the resulting TsdFrame into the following variable
features = 
```
</div>

```{code-cell} ipython3
feats = np.stack((upsampled_pos.values, theta_phase.values))
features = nap.TsdFrame(
    t=theta_phase.t,
    d=np.transpose(feats),
    time_support=upsampled_pos.time_support,
    columns=["position", "phase"],
)
features
```

#### 18. Apply [`nap.compute_tuning_curves`](pynapple.process.tuning_curves.compute_tuning_curves) for `features` on our subselected group of units, `good_spikes`, using 50 bins for position and 20 bins for theta phase.

<div class="render-user">
```{code-cell} ipython3
tuning_curves =
```
</div>

```{code-cell} ipython3
tuning_curves = nap.compute_tuning_curves(good_spikes, features, bins=[50,20])
```

<div class="render-all">

We can plot 2D tuning curves for each unit and visualize how many of these units are phase precessing.

</div>

```{code-cell} ipython3
:tags: [render-all]

tuning_curves.plot(x="position", y="phase", col="unit", col_wrap=5, size=1.5, aspect=1.5)
```

<div class="render-all">

Many of the units display a negative relationship between position and phase, characteristic of phase precession.

</div>

### Decoding position from spiking activity

<div class="render-all">

Next we'll do a popular analysis in the rat hippocampus sphere: Bayesian decoding. This analysis is an elegent application of Bayes' rule in predicting the animal's location (or other behavioral variables) given neural activity at some point in time. Refer to the dropdown box below for a more in-depth explanation.

</div>

:::{admonition} Background: Bayesian decoding
:class: render-all dropdown
Recall Bayes' rule, written here in terms of our relevant variables:

$$P(position|spikes) = \frac{P(position)P(spikes|position)}{P(spikes)}$$

Our goal is to compute the unknown posterior $P(position|spikes)$ given known prior $P(position)$ and known likelihood $P(spikes|position)$. 

$P(position)$, also known as the *occupancy*, is the probability that the animal is occupying some position. This can be computed exactly by the proportion of the total time spent at each position, but in many cases it is sufficient to estimate the occupancy as a uniform distribution, i.e. it is equally likely for the animal to occupy any location.

The next term, $P(spikes|position)$, which is the probability of seeing some sequence of spikes across all neurons at some position. Computing this relys on the following assumptions:
1. Neurons fire according to a Poisson process (i.e. their spiking activity follows a Poisson distribution)
2. Neurons fire independently from one another.

While neither of these assumptions are strictly true, they are generally reasonable for pyramidal cells in hippocampus and allow us to simplify our computation of $P(spikes|position)$

The first assumption gives us an equation for $P(spikes|position)$ for a single neuron, which we'll call $P(spikes_i|position)$ to differentiate it from $P(spikes|position) = P(spikes_1,spikes_2,...,spikes_i,...,spikes_N|position) $, or the total probability across all $N$ neurons. The equation we get is that of the Poisson distribution:
$$
P(spikes_i|position) = \frac{(\tau f_i(position))^n e^{-\tau f_i(position)}}{n!}
$$
where $f_i(position)$ is the firing rate of the neuron at position $(position)$ (i.e. the tuning curve), $\tau$ is the width of the time window over which we're computing the probability, and $n$ is the total number of times the neuron spiked in the time window of interest.

The second assumptions allows us to simply combine the probabilities of individual neurons. Recall the product rule for independent events: $P(A,B) = P(A)P(B)$ if $A$ and $B$ are independent. Treating neurons as independent, then, gives us the following:
$$
P(spikes|position) = \prod_i P(spikes_i|position)
$$

The final term, $P(spikes)$, is inferred indirectly using the law of total probability:

$$P(spikes) = \sum_{position}P(position,spikes) = \sum_{position}P(position)P(spikes|position)$$

Another way of putting it is $P(spikes)$ is the normalization factor such that $\sum_{position} P(position|spikes) = 1$, which is achived by dividing the numerator by its sum.

If this method looks daunting, we have some good news: pynapple has it implemented already in the function `nap.decode_1d` for decoding a single dimension (or `nap.decode_2d` for two dimensions). All we'll need are the spikes, the tuning curves, and the width of the time window $\tau$.
:::

:::{admonition} Aside: Cross-validation
:class: tip render-all
:name: phase-precess-cv
    
Generally this method is cross-validated, which means you train the model on one set of data and test the model on a different, held-out data set. For Bayesian decoding, the "model" refers to the model *likelihood*, which is computed from the tuning curves. 

If we want to decode an example run down the track, our training set should omit this run before computing the tuning curves. We can do this by using the IntervalSet method `set_diff`, to take out the example run epoch from all run epochs. Next, we'll restrict our data to these training epochs and re-compute the place fields using `nap.compute_tuning_curves`. We'll also apply a Gaussian smoothing filter to the place fields, which will smooth our decoding results down the line.

The code cell below will do these steps for you.
:::

```{code-cell} ipython3
:tags: [render-all]

# hold out trial from place field computation
run_train = forward_ep.set_diff(ex_run_ep)
# get position of training set
position_train = position.restrict(run_train)
# compute place fields using training set
place_fields = nap.compute_tuning_curves(spikes, position_train, bins=50, feature_names=["position"])
# smooth place fields
place_fields.data = gaussian_filter1d(place_fields.data, 1, axis=-1)
```

<div class="render-all">

We can decode any number of features using the function [`nap.decode_bayes`](pynapple.process.decoding.decode_bayes). This function requires the following inputs:
- `tuning_curves`: an `xarray.DataArray`, computed by [`nap.compute_tuning_curves`](pynapple.process.tuning_curves.compute_tuning_curves), with the tuning curves relative to the feature(s) being decoded
- `data`: a `TsGroup` of spike times, or a `TsdFrame` of spike counts, for each unit in `tuning_curves`.
- `epochs`: an `IntervalSet` containing the epochs to be decoded
- `bin_size`: the time length, in seconds, of each decoded bin. If `group` is a `TsGroup` of spike times, this determines how the spikes are binned in time. If `group` is a `TsdFrame` of spike counts, this should be the bin size used for the counts.

This function will return two outputs:
- a `Tsd` containing the decoded feature at each decoded time point
- a `TsdFrame` or `TsdTensor` containing the decoded probability of each feature value at each decoded time point
    
</div>

#### 19. Use [`nap.decode_bayes`](pynapple.process.decoding.decode_bayes) to decode position during `ex_run_ep` using 40 ms time bins.

<div class="render-user">
```{code-cell} ipython3
decoded_position, decoded_prob = 
```
</div>

```{code-cell} ipython3
decoded_position, decoded_prob = nap.decode_bayes(place_fields, spikes, ex_run_ep, 0.04)
```

<div class="render-all">

Let's plot decoded position with the animal's true position. We'll overlay them on a heat map of the decoded probability to visualize the confidence of the decoder.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
c = ax.pcolormesh(decoded_position.index,place_fields.position,np.transpose(decoded_prob))
ax.plot(decoded_position, "--", color="red", label="decoded position")
ax.plot(ex_position, color="red", label="true position")
ax.legend()
fig.colorbar(c, label="decoded probability")
ax.set(xlabel="Time (s)", ylabel="Position (cm)", );
```

<div class="render-all">
    
While the decoder generally follows the animal's true position, there is still a lot of error in the decoder, especially later in the run. We can improve the decoder error by smoothing the spike counts. [`nap.decode_bayes`](pynapple.process.decoding.decode_bayes) provides the option to do this for you by specifying `sliding_window_size`, which specifies the width, in number of bins, of a uniform (all ones) kernel to convolve with the spike counts. This is equivalent to applying a moving sum to adjacent bins, where the width of the kernel is the number of adjacent bins being added together. This is equivalent to counting spikes in a *sliding window* that shifts in shorter increments than the window's width, resulting in bins that overlap. This combines the accuracy of using a wider time bin with the temporal resolution of a shorter time bin.

For example, let's say we want a sliding window of $200 ms$ that shifts by $40 ms$. This is equivalent to summing together 5 adjacent $40 ms$ bins, or convolving spike counts in $40 ms$ bins with a length-5 array of ones ($[1, 1, 1, 1, 1]$). Let's visualize this convolution.

</div>

```{code-cell} ipython3
:tags: [render-all]

ex_counts = spikes[unit].restrict(ex_run_ep).count(0.04)
workshop_utils.animate_1d_convolution(ex_counts, np.ones(5), tsd_label="original counts", kernel_label="moving sum", conv_label="convolved counts")
```

<div class="render-all">
    
The count at each time point is computed by convolving the kernel (yellow), centered at that time point, with the original spike counts (blue). For a length-5 kernel of ones, this amounts to summing the counts in the center bin with two bins before and two bins after (shaded green, top). The result is an array of counts smoothed out in time (green, bottom).

</div>

#### 20. Decode the same run as above, now using sliding window size of 5 bins.

<div class="render-user">
```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = 
```
</div>

```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = nap.decode_bayes(place_fields, spikes, ex_run_ep, bin_size=0.04, sliding_window_size=5)
```

<div class="render-all">

Let's plot the results.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig,ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
c = ax.pcolormesh(smth_decoded_position.index,place_fields.position,np.transpose(smth_decoded_prob))
ax.plot(smth_decoded_position, "--", color="red", label="decoded position")
ax.plot(ex_position, color="red", label="true position")
ax.legend()
fig.colorbar(c, label="decoded probability")
ax.set(xlabel="Time (s)", ylabel="Position (cm)", );
```

<div class="render-all">
    
This gives us a much closer approximation of the animal's true position.

Units phase precessing together creates fast, spatial sequences around the animal's true position. We can reveal this by decoding at an even shorter time scale, which will appear as smooth errors in the decoder.

</div>

#### 21. Decode again using a smaller bin size of $10 ms$.

<div class="render-user">
```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = 
```
</div>

```{code-cell} ipython3
smth_decoded_position, smth_decoded_prob = nap.decode_bayes(place_fields, spikes, ex_run_ep, bin_size=0.01, sliding_window_size=5)
```

<div class="render-all">
    
We'll make the same plot as before to visualize the results, but plot it alongside the raw and filtered LFP.

</div>

```{code-cell} ipython3
:tags: [render-all]

fig, axs = plt.subplots(2, 1, figsize=(10, 4), constrained_layout=True, height_ratios=[3,1], sharex=True)
c = axs[0].pcolormesh(smth_decoded_prob.index, smth_decoded_prob.columns, np.transpose(smth_decoded_prob))
p1 = axs[0].plot(smth_decoded_position, "--", color="r")
p2 = axs[0].plot(ex_position, color="r")
axs[0].set_ylabel("Position (cm)")
axs[0].legend([p1[0],p2[0]],["decoded position","true position"])
fig.colorbar(c, label = "predicted probability")

axs[1].plot(lfp.restrict(ex_run_ep))
axs[1].plot(theta_band.restrict(ex_run_ep))
axs[1].set_ylabel("LFP (a.u.)")

fig.supxlabel("Time (s)");
```

<div class="render-all">
    
The estimated position oscillates with cycles of theta, where each "sweep" is referred to as a "theta sequence". Fully understanding the properties of theta sequences and their role in learning, memory, and planning is an active topic of research in Neuroscience!

</div>

### Bonus Exercise

<div class="render-all">
    
Pynapple has another decoding method, [`nap.decode_template`](pynapple.process.decoding.decode_template), that is agnostic to the underlying noise model of the data. In other words, where the above implementation of Bayesian decoding is specific to spiking data (Poisson distributed data), template decoding can be applied to any data modality. As a bonus exercise, you can try decoding position using this method and compare the results to the Bayesian decoder used above!

</div>

```{code-cell} ipython3
# template decoding
```

## Part 2: Using NeMoS to disentangle position and speed encoding

<div class="render-all">
    
Up until now, we have been primarily studying how *position* influences hippocampal firing. How can we be confident that position is influencing the firing rate and not other, correlated variables? This can be disentangled by fitting a GLM.

</div>

### Preprocessing

<div class="render-all">
    
In case any variables got lost or overwritten during part 1, we'll redefine everything we need for part 2 in the cell below. 

To decrease computation time, we're going to spend the rest of the notebook focusing on three selected neurons. For GLM fitting, we're going to bin spikes at 100 Hz and up-sample the position to match that temporal resolution.

</div>

```{code-cell} ipython3
:tags: [render-all]

forward_ep = data["forward_ep"]
position = data["position"].restrict(forward_ep)
spikes = data["units"]
place_fields = nap.compute_tuning_curves(spikes, position, bins=50, epochs=position.time_support, feature_names=["distance"])

neurons = [82, 92, 220]
place_fields = place_fields.sel(unit=neurons)
spikes = spikes[neurons]
bin_size = .01
count = spikes.count(bin_size, ep=position.time_support)
position = position.interpolate(count, ep=count.time_support)
```

### Visualizing speed

<div class="render-all">

One competing variable is speed: the speed at which the animal traverse the field is not homogeneous. Does it influence the firing rate of hippocampal neurons? We can compute tuning curves for speed as well as average speed across the maze.

</div>

```{code-cell} ipython3
:tags: [render-all]

speed = []
# Analyzing each epoch separately avoids edge effects.
for s, e in position.time_support.values: 
    pos_ep = position.get(s, e)
    # Absolute difference of two consecutive points
    speed_ep = np.abs(np.diff(pos_ep)) 
    # Padding the edge so that the size is the same as the position/spike counts
    speed_ep = np.pad(speed_ep, [0, 1], mode="edge") 
    # Converting to cm/s 
    speed_ep = speed_ep * position.rate
    speed.append(speed_ep)

speed = nap.Tsd(t=position.t, d=np.hstack(speed), time_support=position.time_support)
print(speed.shape)

tc_speed = nap.compute_tuning_curves(spikes, speed, bins=20, epochs=speed.time_support, feature_names=["speed"])
fig = workshop_utils.plot_position_speed(position, speed, place_fields.sel(unit=neurons), tc_speed, neurons);
```

<div class="all">

These neurons all show both position and speed tuning, and we see that the animal's speed and position are highly correlated. GLMs can help us model responses to multiple, potentially correlated predictors. 

The goal of this project is to fit a PopulationGLM including both position and speed as predictors, and check if this model accurately captures the tuning curves of the neurons.

</div>

(basis_eval_place_cells)=
### Basis evaluation

<div class="all">
    
As we've seen before, we will use basis objects to represent the input values.  In previous tutorials, we've used the `Conv` basis objects to represent the time-dependent effects we were looking to capture. Here, we're trying to capture the non-linear relationship between our input variables and firing rate, so we want the `Eval` objects. In these circumstances, you should look at the tuning you're trying to capture and compare to the [basis kernels (visualized in NeMoS docs)](table_basis): you want your tuning to be capturable by a linear combination of them.

In this case, several of these would probably work; we will use [`MSplineEval`](nemos.basis.MSplineEval) for both, though with different numbers of basis functions.

Additionally, since we have two different inputs, we'll need two separate basis objects.

</div>

:::{note}
:class: render-all

This afternoon, we'll show how to cross-validate across basis identity, which you can use to choose the basis.

:::

#### 1. Instantiate the basis by doing the following:

<div class="render-all">

- **Create a separate basis object for each model input (speed and position).**
- **Provide a label for each basis ("position" and "speed").**
- **Visualize the basis objects.**

</div>

<div class="render-user">
```{code-cell} ipython3
position_basis = 
speed_basis = 
workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```
</div>

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position")
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed")
workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```

<div class="render-all">
    
However, now we have an issue: in all our previous examples, we had a single basis object, which took a single input to produce a single array which we then passed to the `GLM` object as the design matrix. What do we do when we have multiple basis objects?

For people new to NeMoS, but familiar with NumPy, you can call `basis.compute_features()` for each basis separately and then concatenate the outputs.

For people familiar with NeMoS basis composition, you can add the two bases together obtaining a new 2D basis, then call `compute_features` passing both position and speed to obtain the same design matrix.

</div>

#### 2. Create a design matrix by doing one of the following:

<div class="render-all">

- **Call `compute_fatures` for both position and speed bases and concatenate the result to form a single design matrix.**
- **Add the basis objects together and call `compute_fatures` on the newly created additive basis.**
  
</div>

<div class="render-user">
```{code-cell} ipython3
X_position = 
X_speed = 
X = np.concatenate(
X
```
</div>

```{code-cell} ipython3
# equivalent to calling nmo.basis.AdditiveBasis(position_basis, speed_basis)
basis = position_basis + speed_basis
basis.compute_features(position, speed)
X = basis.compute_features(position, speed)
X
```

<div class="render-uall">

Notice that, since we passed pynapple objects to the basis, we got a pynapple object back, preserving the time stamps. Additionally, `X` has the same number of time points as our input position and speed, but 25 columns. The columns come from  `n_basis_funcs` from each basis (10 for position, 15 for speed).

</div>


### Model learning

<div class="render-all">

As we've done before, we can now use the Poisson GLM from NeMoS to learn the combined model.

</div>

#### 3. Fit a GLM by doing the following:

<div class="render-all">

- **Initialize `PopulationGLM`**
- **Use the "LBFGS" solver and pass `{"tol": 1e-12}` to `solver_kwargs`.**
- **Fit the data, passing the design matrix and spike counts to the glm object.**

</div>

<div class="render-user">
```{code-cell} ipython3
# define the model
glm =
# fit
glm.fit(
```
</div>

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_kwargs={"tol": 1e-12},
    solver_name="LBFGS",
)

glm.fit(X, count)
```

### Prediction

<div class="render-all">

Let's check first if our model can accurately predict the tuning curves we displayed above. We can use the [`predict`](nemos.glm.GLM.predict) function of NeMoS and then compute new tuning curves

</div>

#### 4. Use `predict` to check whether our GLM has captured each neuron's speed and position tuning.

<div class="render-all">

- Remember to convert the predicted firing rate to spikes per second!

</div>

<div class="render-user">
```{code-cell} ipython3
# predict the model's firing rate
predicted_rate =
# compute the position and speed tuning curves using the predicted firing rate.
glm_tuning_pos = 
glm_tuning_speed = 
```
</div>

```{code-cell} ipython3
# predict the model's firing rate
predicted_rate = glm.predict(X) / bin_size

# same shape as the counts we were trying to predict
print(predicted_rate.shape, count.shape)

# compute the position and speed tuning curves using the predicted firing rate.
glm_tuning_pos = nap.compute_tuning_curves(predicted_rate, position, bins=50, epochs=position.time_support, feature_names=["position"])
glm_tuning_speed = nap.compute_tuning_curves(predicted_rate, speed, bins=30, epochs=speed.time_support, feature_names=["speed"])
```

<div class="render-all">

We can plot the results to compare the model and data tuning curves.

</div>

```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_tuning_pos, glm_tuning_speed);
```

<div class="render-all">

We can see that this model does a good job capturing both the position and the speed. 

</div>

### Bonus Exercise

<div class="render-all">

As an bonus, more open-ended exercize, we can investigate all the scientific decisions that we swept under the rug: should we regularize the model? What basis should we use? Do we need both inputs? If you're feeling ambitious, here are some suggestions to answer these questions:

- Try to fit and compare the results we just obtained with different models: 
  - A model with position as the only predictor.
  - A model with speed as the only predictor.
- Introduce L1 (Lasso) regularization and fit models with increasingly large penalty strengths ($\lambda$). Plot the regularization path showing how each coefficient changes with $\lambda$. Identify which coefficients remain non-zero longest as $\lambda$ increases - these correspond to the most informative predictors.

</div>

```{code-cell} ipython3
# bonus exercise
```

## References

<div class="render-all">

The data in this tutorial comes from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

</div>
