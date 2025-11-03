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

This notebook can be downloaded as **{nb-download}`nemos_advanced.ipynb`**. See the button at the top right to download as markdown or pdf.

:::
# NeMoS Advanced: Cross-Validation and Model Selection
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/live_coding/03_nemos_advanced.md)
## Learning Objectives



In this tutorial we will keep working on the hippocampal place field recordings with the goal of learning how to combine NeMoS and scikit-learn to perform cross-validation and model selection. In particular we will:

- Learn how to use NeMoS objects with [scikit-learn](https://scikit-learn.org/) for cross-validation
- Learn how to use NeMoS objects with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- Learn how to use cross-validation to perform model and feature selection. More specifically, we will compare models including position and speed as predictors with model including only speed or only position.


## Pre-Processing


Let's first load and wrangle the data with pynapple and NeMoS. You can run the following cells for preparing the variables that we are going to use in the notebook and recapitulate the content of this dataset with a few visualizations.


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


- Compute the animal's speed.


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

# utility function to visualize predictions
tc_speed = nap.compute_tuning_curves(spikes, speed, bins=20, epochs=speed.time_support, feature_names=["speed"])

def visualize_model_predictions(glm, X):
    # predict the model's firing rate
    predicted_rate = glm.predict(X) / bin_size

    # compute the position and speed tuning curves using the predicted firing rate.
    glm_pos = nap.compute_tuning_curves(predicted_rate, position, bins=50, epochs=position.time_support, feature_names=["position"])
    glm_speed = nap.compute_tuning_curves(predicted_rate, speed, bins=30, epochs=position.time_support, feature_names=["speed"])

    workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pos, glm_speed);
```
### Define 1D NeMoS Bases 


- Define the position and speed bases, and visualize them.


```{code-cell} ipython3
:tags: [render-all]

position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position")
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed")
workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```
## Basis Composition


- Adding the position and speed bases together defines a 2D basis.
- Call `compute_features` to define a design matrix that concatenates both features.



```{code-cell} ipython3
# add the bases
basis = 
# get the design matrix
X = 
```

## Scikit-learn
### How to know when to regularize?


- How do we decide when to use regularization?
- Cross-validation allows you to fairly compare different models on the same dataset.
- NeMoS makes use of [scikit-learn](https://scikit-learn.org/), the standard machine learning library in python.
- Define [parameter grid](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) to search over.
- Anything not specified in grid will be kept constant.



```{code-cell} ipython3
# configurations of the PopulationGLM
solver_kwargs={"tol": 1e-12}
solver_name="LBFGS"
# define a Ridge regularized GLM
glm = 
# get the design matrix
X = 
```



- Initialize scikit-learn's [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) object.



```{code-cell} ipython3
cv_folds = 5
cv = 
cv
```



- We interact with this in a very similar way to the glm object.
- In particular, call `fit` with same arguments:

```{code-cell}
# enter code here
```



- Let's investigate results:

```{code-cell} ipython3
:tags: [render-all]

pd.DataFrame(cv.cv_results_)
```
### Select basis


- You can (and should) do something similar to determine how many basis functions you need for each input.
- NeMoS basis objects are not scikit-learn-compatible right out of the box.
- But we have provided a simple method to make them so:



```{code-cell} ipython3
# convert basis to transformer
position_basis = 
position_basis
```



- This gives the basis object the `transform` method, which is equivalent to `compute_features`.
- However, transformers have some limits:


```{code-cell}
# enter code here
```



- Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality.
- In order to use a basis as a transformer, you'll need to concatenate all your input in a single 2D array.


```{code-cell}
# enter code here
```



:::{dropdown} Other Caveats
:color: info
:icon: info


If the basis has more than one component (for example, if it is the addition of two 1D bases), the transformer will expect an input shape of `(n_sampels, 1)` pre component. If that's not the case, you'll provide a different input shape by calling `set_input_shape`.

**Case 1)** One input per component:

```{code-block} ipython3
# generate a composite basis
basis_2d = nmo.basis.MSplineEval(5) + nmo.basis.MSplineEval(5)
basis_2d = basis_2d.to_transformer()

# this will work: 1 input per component
x, y = np.random.randn(10, 1), np.random.randn(10, 1)
X = np.concatenate([x, y], axis=1)
result = basis_2d.transform(X)
```

**Case 2)** Multiple inputs per component.


- If one or more basis process multiple inputs (multiple columns of the 2D array), trying to call the `tranform` method directly will lead to an error. 
- This is because the basis doesn't know which component should process which column. 


```{code-block} ipython3
:tags: [raises-exception, render-all]

# Assume 2 input for the first component and 3 for the second.
x, y = np.random.randn(10, 2), np.random.randn(10, 3)
X = np.concatenate([x, y], axis=1)

res = basis_2d.transform(X)  # This will raise an exception!
```

To prevent that, use `set_input_shape` to define how many inputs each component should process.

```{code-block} ipython3
# Set the expected input shape instead, different options:

# array
res1 = basis_2d.set_input_shape(x, y).transform(X)
# int
res2 = basis_2d.set_input_shape(2, 3).transform(X)
# tuple
res3 = basis_2d.set_input_shape((2,), (3,)).transform(X)
```

:::


- Let's now create the composite basis for speed and position.



```{code-cell} ipython3
# redefine the basis with label="position"
position_basis =
# redefine the basis with label="speed"
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed")
basis = position_basis + speed_basis
# convert to transformer
basis = 
basis
```



- Stack position and speed in a single TsdFrame to hold all our inputs:


```{code-cell} ipython3
:tags: [render-all]

transformer_input = nap.TsdFrame(
    t=position.t,
    d=np.stack([position, speed]).T,
    time_support=position.time_support,
    columns=["position", "speed"],
)
```


- Pass this input to our transformed additive basis. 
- Note that we do not need to call `set_input_shape` here because each basis element processes one column of the 2D input.


```{code-cell}
# enter code here
```

### Pipelines


- If we want to cross-validate over the basis, we need more one more step: combining the basis and the GLM into a single scikit-learn estimator.
- [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to the rescue!



```{code-cell} ipython3
# set the reg strength to the optimal
glm = 
# pipe the basis and the glm
pipe = pipeline.Pipeline(
pipe
```



- Pipeline runs `basis.transform`, then passes that output to `glm`, so we can do everything in a single line:

```{code-cell}
# enter code here
```



- Visualize model predictions!


```{code-cell} ipython3
:tags: [render-all]

visualize_model_predictions(pipe, transformer_input)
```
### Cross-validating on the basis


Now that we have our pipeline estimator, we can cross-validate on any of its parameters!


```{code-cell}
# enter code here
```



Let's cross-validate on:
- The number of the basis functions of the position basis
- The functional form of the basis for speed
- Let's retrieve the those attributes from the pipeline

```{code-cell} ipython3
:tags: [render-all]

# the label of the pipeline step retrieves the basis
print(pipe["basis"])

# the position basis can by retreived by its label
print("\n", pipe["basis"]["position"])

# the n_basis_funcs is an attribute
print("\n", pipe["basis"]["position"].n_basis_funcs)

# with the same syntax we can retreive the speed basis
print("\n", pipe["basis"]["speed"])
```


- Construct `param_grid`, using `__` to stand in for `.`
- In scikit-learn pipelines, we access nested parameters using double underscores:
  - `pipe["basis"]["position"].n_basis_funcs` - normal Python syntax
  - `"basis__position__n_basis_funcs"` - scikit-learn parameter grid syntax



```{code-cell} ipython3
param_grid = 
```



- Cross-validate as before:



```{code-cell} ipython3
# define the grid search and fit
cv =
```



- Investigate results:


```{code-cell} ipython3
:tags: [render-all]

pd.DataFrame(cv.cv_results_)
```


- Can easily grab the best estimator, the pipeline that did the best:



```{code-cell} ipython3
# define the grid search and fit
best_estim =
best_estim
```



- Visualize model predictions!


```{code-cell} ipython3
:tags: [render-all]

visualize_model_predictions(best_estim, transformer_input)
```
## Feature selection


Let's move on to feature selection. Our goal is comparing alternative models, for this example we will consider: position + speed, position only or speed only.

Problem: scikit-learn's cross-validation assumes that the input to the pipeline does not change, while each model will have a different input. What can we do? 

Let's see how to circumvent this with a neat basis trick.

- Create a "null" basis that produces zero features using `CustomBasis`, which defines a basis from a list of functions.



```{code-cell} ipython3
# this function creates an empty array (n_sample, 0)
def func(x):
    return np.zeros((x.shape[0], 0))
# Create a null transformer basis using the custom basis class
null_basis = 
# this creates an empty feature
null_basis.compute_features(position).shape
```



- First we can note that the original "position + speed" additive basis is the `basis` attribute of the transformer.

```{code-cell}
# enter code here
```



- Add the null basis to the speed or position basis to generate a composite basis for the position-only and speed-only model that receives the same 2D input as the model including all predictors!


```{code-cell} ipython3
# define the 1D transformer bases with no label
position_bas = 
speed_bas = 
# combine them with each other or with the null basis to define each model.
basis_all = 
basis_position = 
basis_speed = 
# assign label (not necessary but nice)
basis_all.label = 
basis_position.label = 
basis_speed.label = 
```



- Create a parameter grid for each model of interest. 
- The attribute to cross-validate over is `"basis__basis"`, where the first "basis" is the name of the pipeline step, the second one is the attribute of the transformer.


```{code-cell} ipython3
param_grid =
```


```{code-cell} ipython3
# finally we define and fit our CV
cv =
```

```{code-cell} ipython3
:tags: [render-all]

cv_df = pd.DataFrame(cv.cv_results_)

# let's just plot a minimal subset of cols
cv_df[["param_basis__basis", "mean_test_score", "rank_test_score"]]
```
## Conclusion


Various combinations of features can lead to different results. From this quick demo it looks like the position-only model is only marginally worst compared to the full model. 

  - [Hardcastle, Kiah, et al. "A multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal cortex." Neuron 94.2 (2017): 375-387](https://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf)

  - [McClain, Kathryn, et al. "Position–theta-phase model of hippocampal place cell activity applied to quantification of running speed modulation of firing rate." Proceedings of the National Academy of Sciences 116.52 (2019): 27035-27042](https://www.pnas.org/doi/abs/10.1073/pnas.1912792116)

  - [Peyrache, Adrien, Natalie Schieferstein, and Gyorgy Buzsáki. "Transformation of the head-direction signal into a spatial code." Nature communications 8.1 (2017): 1752.](https://www.nature.com/articles/s41467-017-01908-3)

## Project Ideas

Use what you learned here and compare models including the theta phase. You can model phase precession as an interaction term between position and theta phase; You can include an interaction term by using the basis multiplication operator, for more information see,

- [Background on basis composition](https://nemos.readthedocs.io/en/latest/background/basis/plot_02_ND_basis_function.html)


## References

The data in this tutorial comes from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

