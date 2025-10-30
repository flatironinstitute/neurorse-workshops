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
```{code-cell} ipython3
:tags: [hide-input, render-all]

%load_ext autoreload
%autoreload 2

%matplotlib inline
import warnings

warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Ignoring cached namespace 'core'",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=(
        "invalid value encountered in div "
    ),
    category=RuntimeWarning,
)
```
:::{admonition} Download
:class: important render-all

This notebook can be downloaded as **{nb-download}`place_cells-users.ipynb`**. See the button at the top right to download as markdown or pdf.

:::
# Fit an Encoding Model
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/day2/place_cells.md)


In this short group project we will keep working on the hippocampal place field recordings. In particular, we will learn how to model neural responses to multiple predictors: position and speed. 


## >>>> Should Be Cropped When Merging
```{code-cell} ipython3
:tags: [render-all]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pynapple as nap

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
import workshop_utils

# configure plots some
plt.style.use(nmo.styles.plot_style)

import workshop_utils

from sklearn import model_selection
from sklearn import pipeline

# shut down jax to numpy conversion warning
nap.nap_config.suppress_conversion_warnings = True
```
## Pynapple

- Load the data using pynapple.

```{code-cell} ipython3
:tags: [render-all]

path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
data = nap.load_file(path)
data
```

- Extract the spike times and mouse position.

```{code-cell} ipython3
:tags: [render-all]

spikes = data["units"]
position = data["position"]
```


- Restrict data to when animal was traversing the linear track.


```{code-cell} ipython3
:tags: [render-all]

position = position.restrict(data["forward_ep"])
spikes = spikes.restrict(data["forward_ep"])
```


- Restrict neurons to only excitatory neurons, discarding neurons with a low-firing rate.


```{code-cell} ipython3
:tags: [render-all]

spikes = spikes.getby_category("cell_type")["pE"]
spikes = spikes.getby_threshold("rate", 0.3)
```
### Place fields


- Visualize the *place fields*: neuronal firing rate as a function of position.

```{code-cell} ipython3
:tags: [render-all]

place_fields = nap.compute_tuning_curves(spikes, position, bins=50, epochs=position.time_support, feature_names=["distance"])
workshop_utils.plot_place_fields(place_fields)
```


- For speed, we're only going to investigate the three neurons highlighted above.
- Bin spikes to counts at 100 Hz.
- Interpolate position to match spike resolution.


```{code-cell} ipython3
:tags: [render-all]

neurons = [82, 92, 220]
place_fields = place_fields.sel(unit=neurons)
spikes = spikes[neurons]
bin_size = .01
count = spikes.count(bin_size, ep=position.time_support)
position = position.interpolate(count, ep=count.time_support)
print(count.shape)
print(position.shape)
```
### Extract Speed per Epoch
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
```
## <<<< End of Part to Be Cropped
### Position and Speed modulation


- Compute the tuning curve with pynapple's [`compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves)


```{code-cell} ipython3
:tags: [render-all]

tc_speed = nap.compute_tuning_curves(spikes, speed, bins=20, epochs=speed.time_support, feature_names=["speed"])
```


- Visualize the position and speed tuning for these neurons.


```{code-cell} ipython3
:tags: [render-all]

fig = workshop_utils.plot_position_speed(position, speed, place_fields, tc_speed, neurons);
```


These neurons all show both position and speed tuning, and we see that the animal's speed and position are highly correlated. GLMs can help us model responses to multiple, potentially correlated predictors. 

The goal of this project is to fit a PopulationGLM including both position and speed as predictors, and check if this model  accurately captures the tuning curves of the neurons.


### Basis evaluation


- Create a separate basis object for each model input (speed and position).
- Provide a label for each basis ("position" and "speed").
- Visualize the basis objects.


```{code-cell} ipython3
position_basis = 
speed_basis = 
workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```



- For users new to NeMoS: call `compute_fatures` for both position and speed basis, and concatenate the result to form a single design matrix.
- Alternatively, for people familiar with NeMoS, add the basis together, and call `compute_fatures` on the newly created additive basis.


```{code-cell} ipython3
X_position = 
X_speed = 
X = np.concatenate(
X
```



- Notice that, since we passed the basis pynapple objects, we got one back, preserving the time stamps.
- `X` has the same number of time points as our input position and speed, but 25 columns. The columns come from  `n_basis_funcs` from each basis (10 for position, 15 for speed).


### Model learning


- Initialize `PopulationGLM`
- Use the "LBFGS" solver and pass `{"tol": 1e-12}` to `solver_kwargs`.
- Fit the data, passing the design matrix and spike counts to the glm object.



```{code-cell} ipython3
# define the model
glm =
# fit
glm.fit(
```

### Prediction


- Use `predict` to check whether our GLM has captured each neuron's speed and position tuning.
- Remember to convert the predicted firing rate to spikes per second!



```{code-cell} ipython3
# predict the model's firing rate
predicted_rate =
# compute the position and speed tuning curves using the predicted firing rate.
glm_tuning_pos = 
glm_tuning_speed = 
```



- Compare model and data tuning curves together. The model did a pretty good job!


```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_tuning_pos, glm_tuning_speed);
```


We can see that this model does a good job capturing both the position and the speed. In the rest of this notebook, we're going to investigate all the scientific decisions that we swept under the rug: should we regularize the model? what basis should we use? do we need both inputs?

## Extra Exercise


If you breezed through this exercise and you feel like working a bit more, here is some suggestions:

- Try to fit and compare the results we just obtained with different models: 
  - A model with position as the only predictor.
  - A model with speed as the only predictor.
- Introduce L1 (Lasso) regularization and fit models with increasingly large penalty strengths ($\lambda$). Plot the regularization path showing how each coefficient changes with $\lambda$. Identify which coefficients remain non-zero longest as $\lambda$ increases - these correspond to the most informative predictors.


To make your lives easier, you can use the helper function  below to visualize model predictions.


## References


The data in this tutorial comes from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

