---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [render-all]

%matplotlib inline
```

:::{admonition} Download
:class: important render-all

This notebook can be downloaded as **{nb-download}`01_fundamentals_of_pynapple-presenters.ipynb`**. See the button at the top right to download as markdown or pdf.
:::

:::{admonition} Jupyter Lab tip
:class: important render-all

Newer versions of Jupyter Lab have addressed an issue with skipping around the notebook while scrolling. To make sure this fix is enabled, in the Jupyter Lab GUI, navigate to `Settings > Settings Editor > Notebook` and scroll down to the `Windowing mode` setting and make sure it is set to `contentVisibility`. 

Also reminder to presenter: Go to `View > Appearance`, select `Simple Interface` and turn off everything else to hide as many bars as possible. And maybe activate `Presentation Mode`.

And turn on `View > Render side-by-side` (shortcut `Shift+R`).
:::



# Learning the fundamentals of pynapple
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/live_coding/01_fundamentals_of_pynapple.md)

## Learning objectives


- Instantiate pynapple objects
- Make pynapple objects interact
- Use numpy with pynapple
- Slice pynapple objects
- Add metadata to pynapple objects
- Apply core functions of pynapple

**Resources:**
- [Pynapple documentation](https://pynapple.org)
- [API reference for objects and methods](https://pynapple.org/api.html)


Let's start by importing the pynapple package and other packages we'll need, as well as generate some fake data that we'll use throughout the notebook.



```{code-cell} ipython3
:tags: [render-all]

%matplotlib inline
import workshop_utils
import pynapple as nap
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import nemos as nmo
plt.style.use(nmo.styles.plot_style)

cos_ts = np.arange(0,100,0.1)
cos_data = np.cos(1/4*cos_ts)

rng = np.random.default_rng(1)
rand_ts = np.arange(0,100)
rand_data = rng.standard_normal((100,3))
rand_col = ['pineapple', 'banana', 'tomato']

spiral_ts = np.arange(0, 100, 0.5)
d = np.linspace(-10,10,100)
spiral_data = np.zeros((len(spiral_ts),len(d),len(d)))
for i,t in enumerate(spiral_ts):
    rv = stats.multivariate_normal([t/10*np.cos(t),t/10*np.sin(t)])
    pos = np.dstack(np.meshgrid(d,d))
    spiral_data[i] = rv.pdf(pos)

rng = np.random.default_rng(2)
t = np.arange(0,100,0.1)
p = np.cos(1/4*t)/5
p = np.where(p>0,p,0)
burst_times = t[np.where(rng.binomial(n=1, p=p))[0]]
random_times = np.sort(rng.uniform(0, 100, 100))
slow_times = np.arange(0,100,10)
```

## Instantiate pynapple objects 



Suppose we have and experiment that generated the following data.



```{code-cell} ipython3
:tags: [render-all]

plt.figure()
plt.plot(cos_ts, cos_data)
plt.title("Cosine Wave")

plt.figure()
plt.plot(rand_ts, rand_data, label=rand_col)
plt.title("Random Data")
plt.legend()

anim = workshop_utils.animate_2d_movie(spiral_data)
plt.title("Spiral Data")
anim.run()
```

### Tsd, TsdFrame, TsdTensor



**Question:** Can you instantiate the correct pynapple objects ([`Tsd`](https://pynapple.org/generated/pynapple.Tsd.html#pynapple.Tsd), [`TsdFrame`](https://pynapple.org/generated/pynapple.TsdFrame.html#pynapple.TsdFrame), and [`TsdTensor`](https://pynapple.org/generated/pynapple.TsdTensor.html#pynapple.TsdTensor)) for each of the data sets above? **NOTE**: Make sure to pass column names to the `TsdFrame` data.



```{code-cell} ipython3
cos_tsd = nap.Tsd(t=cos_ts, d=cos_data)
print(cos_tsd)
rand_tsd = nap.TsdFrame(t=rand_ts, d=rand_data, columns=rand_col)
print(rand_tsd)
spiral_tsd = nap.TsdTensor(t=spiral_ts, d=spiral_data)
print(spiral_tsd)
```

### IntervalSet



**Question:** Can you create and print an [`IntervalSet`](https://pynapple.org/generated/pynapple.IntervalSet.html#pynapple.IntervalSet) called `epochs` out of `starts` and `ends`? Be careful, times given above are in `ms`.



```{code-cell} ipython3
starts = np.array([10000, 60000, 90000]) # starts of an epoch in `ms`
ends = np.array([20000, 80000, 95000])   # ends in `ms`
epochs = nap.IntervalSet(start=starts, end=ends, time_units='ms')
print(epochs)
```

### Ts


    
Suppose we record spike times from three different neurons, plotted below.



```{code-cell} ipython3
:tags: [render-all]

plt.figure(figsize=(8,3))
plt.plot(burst_times,np.zeros_like(burst_times),'|',markersize=50)
plt.plot(random_times,np.ones_like(random_times),'|',markersize=50)
plt.plot(slow_times,1+np.ones_like(slow_times),'|',markersize=50)
plt.yticks([0,1,2],labels=['burst_times','random_times','slow_times']);
```



**Question:** Can you instantiate [`Ts`](https://pynapple.org/generated/pynapple.Ts.html#pynapple.Ts) objects for each set of spike times above?



```{code-cell} ipython3
burst_neuron = nap.Ts(t=burst_times)
random_neuron = nap.Ts(t=random_times)
slow_neuron = nap.Ts(t=slow_times)
```

### TsGroup



**Question:** Can you instantiate a [`TsGroup`](https://pynapple.org/generated/pynapple.TsGroup.html#pynapple.TsGroup) to group together the `Ts` objects defined above and print the result?



```{code-cell} ipython3
all_neurons = nap.TsGroup({0:burst_neuron, 1:random_neuron, 2:slow_neuron})
print(all_neurons)
```

## Interaction between pynapple objects 

### time_support



The [`time_support`](https://pynapple.org/generated/pynapple.TsGroup.html#pynapple.TsGroup.time_support) attribute is an `IntervalSet` associated with every pynapple object that specifys the time interval(s) over which the data is defined. This is inferred from the data or can be set directly during object initialization.

**Question:** Can you print the time support of `all_neurons`?



```{code-cell} ipython3
print(all_neurons.time_support)
```



**Question:** can you recreate the `all_neurons` object passing the true `time_support` during initialisation?



```{code-cell} ipython3
all_neurons = nap.TsGroup({0:burst_neuron, 1:random_neuron, 2:slow_neuron}, time_support = nap.IntervalSet(0, 100))
```



**Question:** Can you print the `time_support` and `rate` to see how they changed?



```{code-cell} ipython3
print(all_neurons.time_support)
print(all_neurons.rate)
```

### restrict



What if we want to limit our data to intervals of interest? We can restrict any pynapple timeseries object to a set of intervals defined by an `IntervalSet` using the object method [`restrict`](https://pynapple.org/generated/pynapple.Tsd.restrict.html#pynapple.Tsd.restrict).

**Question:** Can you create an `IntervalSet` object called `ep_signal` and use it to restrict the variable `cos_tsd`? Include two intervals: from 10s to 30s and from 50s to 100s. 



```{code-cell} ipython3
ep_signal = nap.IntervalSet(start=[10, 50], end=[30, 100])
cos_tsd_signal = cos_tsd.restrict(ep_signal)
```


    
We can print `cos_tsd_signal` to check that the timestamps are within `ep_signal`. Additionally, 
printing the `time_support` shows that it has been updated to match `ep_signal`.



```{code-cell} ipython3
print(cos_tsd_signal)
print(cos_tsd_signal.time_support)
```

### intersect


    
Pynapple `IntervalSet` objects can be intersected to create a new `IntervalSet` using the [`intersect`](https://pynapple.org/generated/pynapple.IntervalSet.intersect.html#pynapple.IntervalSet.intersect) method.



```{code-cell} ipython3
:tags: [render-all]

# random intervals
rng = np.random.default_rng(3)
ep_random = nap.IntervalSet(np.sort(rng.uniform(0, 100, 20)))
print(ep_random)
```



**Question:** Can you intersect `ep_signal` with `ep_random`?



```{code-cell} ipython3
ep_intersect = ep_signal.intersect(ep_random)
ep_intersect
```


    
We can visualize the result using the provided function `workshop_utils.visualize_intervals`



```{code-cell} ipython3
:tags: [render-all]

workshop_utils.visualize_intervals([ep_signal, ep_random, ep_intersect])
plt.yticks([0.25,0.5,0.75],["ep_signal","ep_random","ep_intersect"]);
```

### union



Pynapple `IntervalSet` objects can be joined using the [`union`](https://pynapple.org/generated/pynapple.IntervalSet.union.html#pynapple.IntervalSet.union) method.

**Question:** Can you take the union of `ep_signal` and `ep_random`?



```{code-cell} ipython3
ep_union = ep_signal.union(ep_random)
```



Let's visualize the results.



```{code-cell} ipython3
:tags: [render-all]

workshop_utils.visualize_intervals([ep_signal, ep_random, ep_union])
plt.yticks([0.25,0.5,0.75],["ep_signal","ep_random","ep_union"]);
```

### set_diff



We can also subtract one `IntervalSet` from another using the [`set_diff`](https://pynapple.org/generated/pynapple.IntervalSet.set_diff.html#pynapple.IntervalSet.set_diff) method.

**Question:** Can you take the set difference between `ep_signal` and `ep_random`? Do this twice, with each object acting as the base object. Do you expect the results to be the same?



```{code-cell} ipython3
ep_signal_diff = ep_signal.set_diff(ep_random)
ep_random_diff = ep_random.set_diff(ep_signal)
```



Visualizing the results makes it clear that order matters when using `set_diff`.



```{code-cell} ipython3
:tags: [render-all]

workshop_utils.visualize_intervals([ep_signal, ep_random, ep_signal_diff, ep_random_diff])
plt.yticks([0.2,0.4,0.6,0.8],["ep_signal","ep_random","ep_signal_diff","ep_random_diff"]);
```

### value_from



We can map a set of timepoints to their nearest value from a different timeseries data object by using the method [`value_from`](https://pynapple.org/generated/pynapple.Ts.value_from.html#pynapple.Ts.value_from).

**Question:** Using the function `value_from`, can you assign values to `burst_neuron` from the `cos_tsd` time series into a new object called `burst_cos`?



```{code-cell} ipython3
burst_cos = burst_neuron.value_from(cos_tsd)
burst_cos
```



Let's plot these objects together.



```{code-cell} ipython3
:tags: [render-all]

plt.figure()
plt.plot(cos_tsd)
plt.plot(burst_cos, 'o-')
plt.plot(burst_neuron.fillna(0), 'o')
```

### interpolate



We can resample or upsample one pynapple object to the timepoints of another pynapple object using the [`interpolate`](https://pynapple.org/generated/pynapple.TsdFrame.interpolate.html#pynapple.TsdFrame.interpolate) method. This is useful for matching the sampling rate between two pynapple objects.

**Question:** Can you upsample `rand_tsd` to the sampling rate of `cos_tsd` using the `interpolate` method?



```{code-cell} ipython3
rand_tsd_interp = rand_tsd.interpolate(cos_tsd)
```



Let's visualize the results below.



```{code-cell} ipython3
fig,axs=plt.subplots(2,1, sharex=True, sharey=True)
axs[0].plot(rand_tsd, 'o-')
axs[0].set_title("original points")
axs[1].plot(rand_tsd_interp, 'o-')
axs[1].set_title("interpolated points")
plt.xlim([0,10])
```

## Numpy & pynapple



Pynapple timeseries objects (`Tsd`, `TsdFrame`, and `TsdTensor`) behave similarly to numpy arrays. They can be sliced using similar syntax, e.g.:

  `tsd[0:10] # First 10 elements`

Arithmetic operations also behave like numpy:

  `tsd = tsd + 1`

Finally, numpy functions are compatible with pynapple objects, and in many cases will return a pynapple object when the time axis is preserved.

**Question:** Can you compute the average of `rand_tsd` across columns using `np.mean` and print the result?



```{code-cell} ipython3
print(np.mean(rand_tsd, 1))
```



**Question:** Can you compute the average frame of `spiral_tsd` along the time axis using `np.mean` and print the result?



```{code-cell} ipython3
print(np.mean(spiral_tsd, 0))
```

## Slicing pynapple objects 



**Question:** `IntervalSet` objects also behave similarly to numpy arrays. Using numpy-like indexing, can you extract the first and last epoch of `epochs`?


```{code-cell} ipython3
print(epochs[[0,2]])
```

### special case of slicing : `TsdFrame`



For `TsdFrame` objects with integer column labels, the column labels are ignored when using numpy-like indexing.



```{code-cell} ipython3
:tags: [render-all]

tsdframe = nap.TsdFrame(t=np.arange(4), d=np.random.randn(4,3), columns = [12, 0, 1])
print(tsdframe)
```



**Question:** What happens when you do `tsdframe[0]` vs `tsdframe[:,0]` vs `tsdframe[[12,1]]`?



```{code-cell} ipython3
print(tsdframe[0])
print(tsdframe[:,0])
try:
    print(tsdframe[[12,1]])
except Exception as e:
    print(e)
```



To access `TsdFrame` objects by integer column names, index using the `loc` method.

**Question:** What happen when you do `tsdframe.loc[0]` and `tsdframe.loc[[0,1]]`?



```{code-cell} ipython3
print(tsdframe.loc[0])
print(tsdframe[:,0])
```

### get



**Question:** Using the [`get`](https://pynapple.org/generated/pynapple.TsdTensor.get.html#pynapple.TsdTensor.get) method, can you get the data point from `spiral_tsd` as close as possible to the time 50.1 seconds?



```{code-cell} ipython3
print(spiral_tsd.get(50.1))
```



**Question:** Using the `get` method, can you get the data point from `spiral_tsd` that occur between 50.1 and 52.1 seconds? **NOTE:** The time support is not updated using `get`.



```{code-cell} ipython3
print(spiral_tsd.get(50.1, 52.1))
```

### get_slice



**Question:** Using the `get_slice` method, can you get the index of [`spiral_tsd`](https://pynapple.org/generated/pynapple.TsdTensor.get_slice.html#pynapple.TsdTensor.get_slice) as close as possible to the time 50.1 seconds?



```{code-cell} ipython3
print(spiral_tsd.get_slice(50.1))
print(spiral_tsd.get_slice(50.1).start)
```

## Metadata



Using metadata, we can attach additional info, such as experimental labels, to some of our pynapple objects. Specifically, the following three objects support metadata:

- `TsGroup` : to label each set of time stamps, e.g. neuron region
- `IntervalSet` : to label each interval, e.g. stimulus identity
- `TsdFrame` : to label each column, e.g. neurons in calcium imaging

Metadata can be any data type, and there are a few ways to add/access metadata to/from pynapple objects. 



### setting metadata

#### item assignment



Metadata can be added to an object using dictionary-like item assignment.

**Question:** Can you add the metadata labels `["burst","random","slow"]` using item assignment to `all_neurons["label"]` and print the result?



```{code-cell} ipython3
all_neurons["label"] = ["burst", "random", "slow"]
all_neurons
```

#### attribute assignment



Metadata can also be set directly as an attribute to the object.

**Question:** Can you add the values `[1, -1, 1]` to `epochs` as the attribute `epochs.direction`?



```{code-cell} ipython3
epochs.direction = ["left", "right", "left"]
epochs
```

#### set_info


    
Each object also has the method [`set_info`](https://pynapple.org/generated/pynapple.TsdFrame.set_info.html#pynapple.TsdFrame.set_info) which allows you to set metadata using keyword arguments to the method.

**Question:** Can you add the rgb colors `[(0,0,1), (0.5, 0.5, 1), (0.1, 0.2, 0.3)]` as metadata of `rand_tsd` using the `set_info` method?



```{code-cell} ipython3
rand_tsd.set_info(color=[(0,0,1), (0.5, 0.5, 1), (0.1, 0.2, 0.3)])
rand_tsd
```

#### at initialization



You can also add metadata at initialization as a dictionary using the keyword argument `metadata`: 



```{code-cell} ipython3
:tags: [render-all]

rand_tsd = nap.TsdFrame(
    t = rand_ts,d = rand_data,columns=rand_col,
    metadata={'color':['orange','yellow', 'red']}
)
print(rand_tsd)
```

### accessing metadata



Similar to setting metadata, we can retrieve metadata as an attribute (i.e. `all_neurons.label`) or using item access (i.e. `all_neurons['label']`). Additionally we can use [`get_info`](https://pynapple.org/generated/pynapple.TsGroup.get_info.html#pynapple.TsGroup.get_info), a complementary method to `set_info`, to access metadata.



```{code-cell} ipython3
:tags: [render-all]

all_neurons.get_info('label')
```



We can also access the metadata as a pandas DataFrame using the `metadata` attribute.



```{code-cell} ipython3
:tags: [render-all]

all_neurons.metadata
```

### slicing with metadata



Metadata can be used to slice pynapple objects by indexing or, for numeric metadata, by the method [`getby_threshold`](https://pynapple.org/generated/pynapple.TsGroup.getby_threshold.html#pynapple.TsGroup.getby_threshold).

**Question:** Can you select only the elements of `all_neurons` with rate below 1Hz?



```{code-cell} ipython3
print(all_neurons[all_neurons.rate<1.0])

print(all_neurons[all_neurons['rate']<1.0])

print(all_neurons.getby_threshold("rate", 1, "<"))
```



**Question:** Can you select the intervals in `epochs` with a direction of "left"?


```{code-cell} ipython3
print(epochs[epochs.direction=="left"])
```

#### special case of slicing : `TsdFrame`



Where metadata of `TsGroup` and `IntervalSet` objects are associated with each *row*, metadata of `TsdFrame` objects instead is associated with each *column*. This means slicing with metadata must be done on the second axis.

**Question:** Can you select the columns of `rand_tsd` where the color is orange?



```{code-cell} ipython3
print(rand_tsd[:, rand_tsd.color=="orange"])
```

## Core functions of pynapple 

### count



The [`count`](https://pynapple.org/generated/pynapple.TsGroup.count.html#pynapple.TsGroup.count) methods allows us to count or bin the number of time points that fall within each window of a given bin size. 

**Question:** Using the `count` method, can you count the number of events within 1 second bins for `all_neurons` over the `ep_signal` intervals?


```{code-cell} ipython3
count =
```


```{code-cell} ipython3
count = all_neurons.count(1, ep_signal)
print(count)
```



Let's visulize the results. **TIP**: Pynapple works directly with matplotlib. Passing a time series object to `plt.plot` will display the figure with the correct time axis.



```{code-cell} ipython3
:tags: [render-all]

plt.figure()
ax = plt.subplot(211)
plt.plot(count, 'o-')
plt.subplot(212, sharex=ax)
plt.plot(all_neurons.restrict(ep_signal).to_tsd(), '|')
```

### bin_average



Oftentimes we need to match the sampling rates between different sets of data. Pynapple provides the [`bin_average`](https://pynapple.org/generated/pynapple.TsdFrame.bin_average.html#pynapple.TsdFrame.bin_average) method to downsample data.

**Question:** Can you downsample `rand_tsd` to one time point every 5 seconds?



```{code-cell} ipython3
rand_downsamp = rand_tsd.bin_average(5.0)
```



Let's plot the column for "tomato" and it's downsampled version.



```{code-cell} ipython3
:tags: [render-all]

plt.figure()
plt.plot(rand_tsd['tomato'])
plt.plot(rand_downsamp['tomato'], 'o-')
```

### threshold



We may want to find all the intervals where our timeseries data exceeds some value. For 1-dimensional `Tsd` objects, Pynapple provides the [`threshold`](https://pynapple.org/generated/pynapple.Tsd.threshold.html#pynapple.Tsd.threshold) method to limit the `Tsd` above or below a certain value.

**Question**: Can you threshold `cos_tsd` for values above 0.0? Can you get the intervals of this thresholded data?



```{code-cell} ipython3
cos_thresh = cos_tsd.threshold(0.0)
ep_above = cos_thresh.time_support
print(ep_above)
```



Let's visualize the resulting `Tsd` and `IntervalSet`.



```{code-cell} ipython3
:tags: [render-all]

plt.figure()
plt.plot(cos_tsd)
plt.plot(cos_thresh, 'o-')
[plt.axvspan(s, e, alpha=0.2) for s,e in ep_above.values];
```

### to_trial_tensor



We can reshape a `Tsd`, `TsdFrame`, or `TsdTensor` object into a trial-based tensor using the method [`to_trial_tensor`](https://pynapple.org/generated/pynapple.Tsd.to_trial_tensor.html#pynapple.Tsd.to_trial_tensor), where trials are defined by an `IntervalSet`. The resulting tensor is returned as a numpy array, since pynapple objects do not support 2D time axes.

**Question:** Can you create a trial-based tensor of `cos_tsd` using `trials` provided below? Print the resulting shape.



```{code-cell} ipython3
trials = nap.IntervalSet(start=np.arange(0,100,20),end=np.arange(19.9,100,20))
cos_tsd.to_trial_tensor(trials).shape
```

### trial_count



A similar function to `to_trial_tensor` exists for `Ts` and `TsGroup` objects: [`trial_count`](https://pynapple.org/generated/pynapple.TsGroup.trial_count.html#pynapple.TsGroup.trial_count). Instead of reshaping the time points, however, it counts the number of time points into a trial-based array, akin to a 2D `count`. Similar to `count`, this method requires you to specify a `bin_size`.

**Question:** Can you compute a trial-based count of the spikes in `all_neurons` using `trials` set above and `bin_size=0.1`? Print the resulting shape.



```{code-cell} ipython3
all_neurons.trial_count(trials, 0.1).shape
```

## First high-level function: compute_tuning_curves



Pynapple provides functions for standard analysis in systems neuroscience. The first function we will try is [`compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves) that calculates the response of a cell to a particular feature. 

A good practice when using a function for the first time is to check the docstrings to learn how to pass the arguments.

**Question**: Can you examine the docstring of `nap.compute_tuning_curves`?



```{code-cell} ipython3
print(nap.compute_tuning_curves.__doc__)
```



**Question**: Can you compute the response (i.e. firing rate) of the units in `all_neurons` as function of the feature `cos_tsd` using the function `nap.compute_tuning_curves`? Label the feature as `"cosine"`



```{code-cell} ipython3
tuning_curves = nap.compute_tuning_curves(all_neurons, cos_tsd, bins=5, feature_names=["cosine"])
tuning_curves
```



The output is an [xarray](https://docs.xarray.dev/en/stable/) object, which acts as a wrapper to numpy arrays with extra utilities. It allows us to specify the coordinates of each dimension as well as attach additional attributes. By labeling our feature(s), we can make the output more readable.



**Question**: Can you print the underlying the units number, feature value, and bin edges from `tuning_curves`?



```{code-cell} ipython3
print(tuning_curves.unit.values)
print(tuning_curves.cosine.values)
print(tuning_curves.occupancy)
print(tuning_curves.bin_edges)
print(tuning_curves.fs)
```


    
Xarray objects also supply convenient methods for quickly plotting data.

**Question**: Can you plot the tuning curve of each unit using the `plot` method?



```{code-cell} ipython3
tuning_curves.plot(hue="unit")
plt.ylabel("firing rate");
```

## Verify Your Setup



**Question:** Does the following data download work correctly? If not, please ask a TA.



```{code-cell} ipython3
:tags: [render-all]

import workshop_utils
path = workshop_utils.fetch_data("Mouse32-140822.nwb")
print(path)
```