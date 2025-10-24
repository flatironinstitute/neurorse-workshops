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

This notebook can be downloaded as **{nb-download}`place_cells-presenters.ipynb`**. See the button at the top right to download as markdown or pdf.

:::

# Model selection
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/day2/place_cells.md)



## Learning objectives

- Review how to use pynapple to analyze neuronal tuning
- Learn how to combine NeMoS basis objects for modeling multiple predictors



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

# during development, set this to a lower number so everything runs faster. 
cv_folds = 5
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

### Speed modulation



- Compute animal's speed for each epoch.



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



- Compute the tuning curve with pynapple's [`compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves)



```{code-cell} ipython3
tc_speed = nap.compute_tuning_curves(spikes, speed, bins=20, epochs=speed.time_support, feature_names=["speed"])
```



- Visualize the position and speed tuning for these neurons.


```{code-cell} ipython3
:tags: [render-all]

fig = workshop_utils.plot_position_speed(position, speed, place_fields, tc_speed, neurons);
```



These neurons all show both position and speed tuning, and we see that the animal's speed and position are highly correlated. We're going to build a GLM to predict neuronal firing rate -- which variable should we use? Is the speed tuning just epiphenomenal?



## NeMoS

### Basis evaluation



- why basis?
   - without basis:
     - either the GLM says that firing rate increases exponentially as position or speed increases, which is fairly nonsensical,
     - or we have to fit the weight separately for each position or speed, which is really high-dim
   - so, basis allows us to reduce dimensionality, capture non-linear modulation of firing rate (in this case, tuning)
- why eval?
    - basis objects have two modes:
    - conv, like we've seen, for capturing time-dependent effects
    - eval, for capturing non-linear modulation / tuning
- why MSpline?
    - when deciding on eval basis, look at the tuning you want to capture, compare to the kernels: you want your tuning to be capturable by a linear combination of these
    - in cases like this, many possible basis objects we could use here and what I'll show you in a bit will allow you to determine which to use in principled manner
    - MSpline, BSpline, RaisedCosineLinear : all would let you capture this
    - weird choices:
        - cyclic bspline, except maybe for position? if end and start are the same
        - RaisedCosineLog (don't want the stretching)
        - orthogonalized exponential (specialized for...)
        - identity / history (too basic)




- Create a separate basis object for each model input.
- Visualize the basis objects.


```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10)
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15)
workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```



- Combine the two basis objects into a single "additive basis"


```{code-cell} ipython3
# equivalent to calling nmo.basis.AdditiveBasis(position_basis, speed_basis)
basis = position_basis + speed_basis
```



- Create the design matrix!
- Notice that, since we passed the basis pynapple objects, we got one back, preserving the time stamps.
- `X` has the same number of time points as our input position and speed, but 25 columns. The columns come from  `n_basis_funcs` from each basis (10 for position, 15 for speed).



```{code-cell} ipython3
X = basis.compute_features(position, speed)
X
```

### Model learning



- Initialize `PopulationGLM`
- Use the "LBFGS" solver and pass `{"tol": 1e-12}` to `solver_kwargs`.
- Fit the data, passing the design matrix and spike counts to the glm object.



```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_kwargs={"tol": 1e-12},
    solver_name="LBFGS",
)

glm.fit(X, count)
```

### Prediction



- Use `predict` to check whether our GLM has captured each neuron's speed and position tuning.
- Remember to convert the predicted firing rate to spikes per second!



```{code-cell} ipython3
# predict the model's firing rate
predicted_rate = glm.predict(X) / bin_size

# same shape as the counts we were trying to predict
print(predicted_rate.shape, count.shape)

# compute the position and speed tuning curves using the predicted firing rate.
glm_pos = nap.compute_tuning_curves(predicted_rate, position, bins=50, epochs=position.time_support, feature_names=["position"])
glm_speed = nap.compute_tuning_curves(predicted_rate, speed, bins=30, epochs=speed.time_support, feature_names=["speed"])
```



- Compare model and data tuning curves together. The model did a pretty good job!



```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pos, glm_speed);
```



We can see that this model does a good job capturing both the position and the speed. In the rest of this notebook, we're going to investigate all the scientific decisions that we swept under the rug: should we regularize the model? what basis should we use? do we need both inputs?

To make our lives easier, let's create a helper function that wraps the above
lines, because we're going to be visualizing our model predictions a lot.



```{code-cell} ipython3
def visualize_model_predictions(glm, X):
    # predict the model's firing rate
    predicted_rate = glm.predict(X) / bin_size

    # compute the position and speed tuning curves using the predicted firing rate.
    glm_pos = nap.compute_tuning_curves(predicted_rate, position, bins=50, epochs=position.time_support, feature_names=["position"])
    glm_speed = nap.compute_tuning_curves(predicted_rate, speed, bins=30, epochs=position.time_support, feature_names=["speed"])

    workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pos, glm_speed);
```



In our previous analysis of the place field hyppocampal dataset we compared multiple encoding models and tried to figure out which predictor (position, speed or phase) had more explanatory power. In this notebook we will keep going on that effort and learn more principled (and convenient) approaches to model comparison combining NeMoS and scikit-learn.

## Learning Objectives

- Learn how to use NeMoS objects with [scikit-learn](https://scikit-learn.org/) for cross-validation
- Learn how to use NeMoS objects with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- Learn how to use cross-validation to perform model and feature selection
- 


## Scikit-learn

### How to know when to regularize?



- How do we decide when to use regularization?
- Cross-validation allows you to fairly compare different models on the same dataset.
- NeMoS makes use of [scikit-learn](https://scikit-learn.org/), the standard machine learning library in python.
- Define [parameter grid](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) to search over.
- Anything not specified in grid will be kept constant.



```{code-cell} ipython3
# define a Ridge GLM
glm = nmo.glm.PopulationGLM(
    regularizer="Ridge",
    solver_kwargs={"tol": 1e-12},
    solver_name="LBFGS",
)
param_grid = {
    "regularizer_strength": [0.0001, 1.],
}
```



- Initialize scikit-learn's [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) object.


```{code-cell} ipython3
cv = model_selection.GridSearchCV(glm, param_grid, cv=cv_folds)
cv
```



- We interact with this in a very similar way to the glm object.
- In particular, call `fit` with same arguments:


```{code-cell} ipython3
cv.fit(X, count)
```



- We got a warning because we didn't specify the regularizer strength, so we just fell back on default value.
- Let's investigate results:


```{code-cell} ipython3
import pandas as pd

pd.DataFrame(cv.cv_results_)
```

### Select basis



- You can (and should) do something similar to determine how many basis functions you need for each input.
- NeMoS basis objects are not scikit-learn-compatible right out of the box.
- But we have provided a simple method to make them so:



```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position").to_transformer()
# or equivalently:
position_basis = nmo.basis.TransformerBasis(nmo.basis.MSplineEval(n_basis_funcs=10, label="position"))
position_basis
```



- This gives the basis object the `transform` method, which is equivalent to `compute_features`.
- However, transformers have some limits:



```{code-cell} ipython3
:tags: [raises-exception]

position_basis.transform(position)
```



- Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality.



```{code-cell} ipython3
position_basis.transform(position[:, np.newaxis])
```



- If the basis is composite (for example, the addition of two 1D bases), the transformer will expect a shape of `(n_sampels, 1)` each 1D component. If that's not the case, you need to call `set_input_shape`:



```{code-cell} ipython3
# generate a composite basis
basis_2d = nmo.basis.MSplineEval(5) +  nmo.basis.MSplineEval(5)
basis_2d = basis_2d.to_transformer()

# this will work: 1 input per component
x, y = np.random.randn(10, 1), np.random.randn(10, 1)
X = np.concatenate([x, y], axis=1)
result = basis_2d.transform(X)
```



- Then you can call transform on the 2d input as expected.


```{code-cell} ipython3
# Assume 2 input for the first component and 3 for the second.
x, y = np.random.randn(10, 2), np.random.randn(10, 3)
X = np.concatenate([x, y], axis=1)
try:
    basis_2d.transform(X)
except Exception as e:
    print("Exception Raised:")
    print(repr(e))

# Set the expected input shape instead.

# array
res1 = basis_2d.set_input_shape(x, y).transform(X)
# int
res2 = basis_2d.set_input_shape(2, 3).transform(X)
# tuple
res3 = basis_2d.set_input_shape((2,), (3,)).transform(X)
```



- You can, equivalently, call `compute_features` *before* turning the basis into a transformer. Then we cache the shape for future use:



```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position")
position_basis.compute_features(position)
position_basis = position_basis.to_transformer()
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed").to_transformer().set_input_shape(1)
basis = position_basis + speed_basis
basis
```



- Create a single TsdFrame to hold all our inputs:


```{code-cell} ipython3
:tags: [render-all]

transformer_input = nap.TsdFrame(
    t=position.t,
    d=np.stack([position, speed], 1),
    time_support=position.time_support,
    columns=["position", "speed"],
)
```



- Pass this input to our transformed additive basis:


```{code-cell} ipython3
basis.transform(transformer_input)
```

### Pipelines



- If we want to cross-validate over the basis, we need more one more step: combining the basis and the GLM into a single scikit-learn estimator.
- [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to the rescue!



```{code-cell} ipython3
# set the reg strength to the optimal
glm = nmo.glm.PopulationGLM(solver_name="LBFGS", solver_kwargs={"tol": 10**-12})
pipe = pipeline.Pipeline([
    ("basis", basis),
    ("glm", glm)
])
pipe
```



- Pipeline runs `basis.transform`, then passes that output to `glm`, so we can do everything in a single line:


```{code-cell} ipython3
pipe.fit(transformer_input, count)
```



- Visualize model predictions!



```{code-cell} ipython3
visualize_model_predictions(pipe, transformer_input)
```

### Cross-validating on the basis



Now that we have our pipeline estimator, we can cross-validate on any of its parameters!



```{code-cell} ipython3
pipe.steps
```



Let's cross-validate on:
- The number of the basis functions of the position basis
- The functional form of the basis for speed


```{code-cell} ipython3
print(pipe["basis"]["position"].n_basis_funcs)
print(pipe["basis"]["speed"])
```



- Construct `param_grid`, using `__` to stand in for `.`
- In sklearn pipelines, we access nested parameters using double underscores:
  - `pipe["basis"]["position"].n_basis_funcs` ← normal Python syntax
  - `"basis__position__n_basis_funcs"` ← sklearn parameter grid syntax



```{code-cell} ipython3
param_grid = {
    "basis__position__n_basis_funcs": [5, 10, 20],
    "basis__speed": [nmo.basis.MSplineEval(15).set_input_shape(1),
                      nmo.basis.BSplineEval(15).set_input_shape(1),
                      nmo.basis.RaisedCosineLinearEval(15).set_input_shape(1)],
}
```



- Cross-validate as before:


```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
cv.fit(transformer_input, count)
```



- Investigate results:


```{code-cell} ipython3
pd.DataFrame(cv.cv_results_)
```



- Can easily grab the best estimator, the pipeline that did the best:


```{code-cell} ipython3
best_estim = cv.best_estimator_
best_estim
```



- Visualize model predictions!



```{code-cell} ipython3
:tags: [render-all]

visualize_model_predictions(best_estim, transformer_input)
```

### Feature selection

```{code-cell} ipython3
# this function creates an empty array (n_sample, 0)
def func(x):
    return np.zeros((x.shape[0], 0))

# Create a null basis using the custom basis class
null_basis = nmo.basis.CustomBasis([func]).to_transformer()

# this creates an empty feature
null_basis.compute_features(position).shape
```

```{code-cell} ipython3
# first we note that the position + speed basis is in the basis attribute
print(pipe["basis"].basis)

position_bas = nmo.basis.MSplineEval(n_basis_funcs=10).to_transformer()
speed_bas = nmo.basis.MSplineEval(n_basis_funcs=15).to_transformer()

# define 2D basis per each model 
basis_all = position_bas + speed_bas
basis_position = position_bas + null_basis
basis_speed = null_basis + speed_bas

# assign label (not necessary but nice)
basis_all.label = "position + speed"
basis_position.label = "position"
basis_speed.label = "speed"


# then we create a parameter grid defining a grid of 2D basis for each model of interest
param_grid = {
    "basis__basis": 
    [
        basis_all,  
        basis_position, 
        basis_speed 
    ],
}

# finally we define and fit our CV
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
cv.fit(transformer_input, count)
```

```{code-cell} ipython3
cv_df = pd.DataFrame(cv.cv_results_)

# let's just plot a minimal subset of cols
cv_df[["param_basis__basis", "mean_test_score", "rank_test_score"]]
```

## Conclusion

## References



The data in this tutorial comes from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).



```{code-cell} ipython3

```