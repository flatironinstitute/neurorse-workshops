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


## Learning Objectives

<div class="render-all">


In this tutorial we will keep working on the hippocampal place field recordings with the goal of learning how to combine NeMoS and scikit-learn to perform cross-validation and model selection. In particular we will:

- Learn how to use NeMoS objects with [scikit-learn](https://scikit-learn.org/) for cross-validation
- Learn how to use NeMoS objects with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- Learn how to use cross-validation to perform model and feature selection. More specifically, we will compare models including position and speed as predictors with model including only speed or only position.

</div>


## Pre-Processing

<div class="render-all">

Let's first load and wrangle the data with pynapple and NeMoS. You can run the following cells for preparing the variables that we are going to use in the notebook and recapitulate the content of this dataset with a few visualizations.

</div>

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


<div class="render-user render-presenter">
- Load the data using pynapple.
</div>

```{code-cell} ipython3
:tags: [render-all]

path = workshop_utils.fetch_data("Achilles_10252013_EEG.nwb")
data = nap.load_file(path)
data
```

<div class="render-user render-presenter">
- Extract the spike times and mouse position.
</div>

```{code-cell} ipython3
:tags: [render-all]

spikes = data["units"]
position = data["position"]
```

For today, we're only going to focus on the times when the animal was traversing the linear track. 
This is a pynapple `IntervalSet`, so we can use it to restrict our other variables:

<div class="render-user render-presenter">

- Restrict data to when animal was traversing the linear track.

</div>

```{code-cell} ipython3
:tags: [render-all]

position = position.restrict(data["forward_ep"])
spikes = spikes.restrict(data["forward_ep"])
```

The recording contains both inhibitory and excitatory neurons. Here we will focus of the excitatory cells with firing above 0.3 Hz.

<div class="render-user render-presenter">

- Restrict neurons to only excitatory neurons, discarding neurons with a low-firing rate.

</div>

```{code-cell} ipython3
:tags: [render-all]

spikes = spikes.getby_category("cell_type")["pE"]
spikes = spikes.getby_threshold("rate", 0.3)
```

### Place fields

By plotting the neuronal firing rate as a function of position, we can see that these neurons are all tuned for position: they fire in a specific location on the track.

<div class="render-user render-presenter">

- Visualize the *place fields*: neuronal firing rate as a function of position.
</div>

```{code-cell} ipython3
:tags: [render-all]

place_fields = nap.compute_tuning_curves(spikes, position, bins=50, epochs=position.time_support, feature_names=["distance"])
workshop_utils.plot_place_fields(place_fields)
```

To decrease computation time, we're going to spend the rest of the notebook focusing on the neurons highlighted above. We're also going to bin spikes at 100 Hz and up-sample the position to match that temporal resolution.

<div class="render-user render-presenter">

- For speed, we're only going to investigate the three neurons highlighted above.
- Bin spikes to counts at 100 Hz.
- Interpolate position to match spike resolution.

</div>

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

In the next block, we compute the speed of the animal for each epoch (i.e. crossing of the linear track) by doing the difference of two consecutive position multiplied by the sampling rate of the position.

<div class="render-user render-presenter">

- Compute the animal's speed.

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

<div class="render-user render-presenter">

- Define the position and speed bases, and visualize them.

</div>

```{code-cell} ipython3
:tags: [render-all]

position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position")
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed")
workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```

## Basis Composition

The first new concept we will introduce will be that of basis composition. NeMoS basis can be composed using the "+" (and "*", see [NeMoS docs](https://nemos.readthedocs.io/en/latest/background/basis/plot_02_ND_basis_function.html) of more info) operator, to define more complex predictor. 

Adding two 1D basis, will result in a 2D additive basis. The `compute_features` of the additive basis requires 2 inputs, and the output will be the concatenation of the design matrices of the basis components.


<div class="render-user render-presenter">

- Adding the position and speed bases together defines a 2D basis.
- Call `compute_features` to define a design matrix that concatenates both features.

</div>

<div class="render-user">
```{code-cell} ipython3
# add the bases
basis = 
# get the design matrix
X = 
```
</div>

```{code-cell} ipython3

basis = position_basis + speed_basis

X = basis.compute_features(position, speed)
X_numpy = np.concatenate(
    [
        position_basis.compute_features(position),
        speed_basis.compute_features(speed),
    ],
    axis=1
)

print("Are the design matrices equivalent?", np.all(X.d == X_numpy.d))
```

## Scikit-learn

(sklearn-cv)=
### How to know when to regularize?

In the [head direction](./head_direction.md) notebook, we fit the all-to-all connectivity of the head-tuning dataset using the Ridge regularizer, and we learned that regularization can combat overfitting. What we didn't show is how to choose a proper regularizer. Generally, too much regularization leads to underfitting, i.e. the model is too simple and doesn't capture the neural variability well. To little regularization may overfit, especially when we have a large number of parameters, i.e. out model will capture both signal and noise. This is what we saw in the head direction notebook when we used the raw spike history as predictor. 

What we are looking for is a regularization strength that balances out the bias towards simpler models with the variance necessary to explain the data. However, how do we know how much we should regularize? One thing we can do is use cross-validation to see whether model performance on unseen data improves with regularization (behind the scenes, this is what we did!). We'll walk through how to do that now.

Instead of implementing our own cross-validation machinery, the developers of nemos decided that we should write the package to be compliant with [scikit-learn](https://scikit-learn.org), the canonical machine learning python library. Our models are all what scikit-learn calls "estimators", which means they have `.fit`, `.score.` and `.predict` methods. Thus, we can use them with scikit-learn's objects out of the box.

We're going to use scikit-learn's [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) object, which performs a cross-validated grid search, as [Edoardo explained in his presentation](https://users.flatironinstitute.org/~wbroderick/presentations/sfn-2025/model_selection.pdf).

This object requires an estimator, our `glm` object here, and `param_grid`, a dictionary defining what to check. For now, let's just compare Ridge regularization with no regularization:

<div class="render-user render-presenter">

- How do we decide when to use regularization?
- Cross-validation allows you to fairly compare different models on the same dataset.
- NeMoS makes use of [scikit-learn](https://scikit-learn.org/), the standard machine learning library in python.
- Define [parameter grid](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) to search over.
- Anything not specified in grid will be kept constant.

</div>

<div class="render-user">
```{code-cell} ipython3
# configurations of the PopulationGLM
solver_kwargs={"tol": 1e-12}
solver_name="LBFGS"
# define a Ridge regularized GLM
glm = 
# get the design matrix
X = 
```
</div>

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


<div class="render-user render-presenter">

- Initialize scikit-learn's [`GridSearchCV`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) object.

</div>

<div class="render-user">
```{code-cell} ipython3
cv_folds = 5
cv = 
cv
```
</div>

```{code-cell} ipython3
cv_folds = 5
cv = model_selection.GridSearchCV(glm, param_grid, cv=cv_folds)
cv
```


This will take a bit to run, because we're fitting the model many times!

<div class="render-user render-presenter">

- We interact with this in a very similar way to the glm object.
- In particular, call `fit` with same arguments:
</div>

```{code-cell} ipython3
cv.fit(X, count)
```

<div class="render-user render-presenter">

- Let's investigate results:
</div>

Cross-validation results are stored in a dictionary attribute called `cv_results_`, which contains a lot of info. Let's convert that to a pandas dataframe for readability,

```{code-cell} ipython3
:tags: [render-all]

pd.DataFrame(cv.cv_results_)
```

The most informative for us is the `'mean_test_score'` key, which shows the average of `glm.score` on each test-fold. Thus, higher is better, and we can see that the UnRegularized model performs better.


(sklearn-basis)=
### Select basis

We can do something similar to select the basis. In the above example, I just told you which basis function to use and how many of each. But, in general, you want to select those in a reasonable manner. Cross-validation to the rescue!

Unlike the glm objects, our basis objects are not scikit-learn compatible right out of the box. However, they can be made compatible by using the `.to_transformer()` method (or, equivalently, by using the `TransformerBasis` class)

<div class="render-user render-presenter">

- You can (and should) do something similar to determine how many basis functions you need for each input.
- NeMoS basis objects are not scikit-learn-compatible right out of the box.
- But we have provided a simple method to make them so:

</div>


<div class="render-user">
```{code-cell} ipython3
# convert basis to transformer
position_basis = 
position_basis
```
</div>

```{code-cell} ipython3

position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position").to_transformer()
# or equivalently:
position_basis = nmo.basis.TransformerBasis(nmo.basis.MSplineEval(n_basis_funcs=10, label="position"))
position_basis
```


This gives the basis object the `transform` method, which is equivalent to `compute_features`. However, transformers have some limits:

<div class="render-user render-presenter">

- This gives the basis object the `transform` method, which is equivalent to `compute_features`.
- However, transformers have some limits:

</div>

```{code-cell} ipython3
:tags: [raises-exception]

position_basis.transform(position)
```

<div class="render-user render-presenter">

- Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality.
- In order to use a basis as a transformer, you'll need to concatenate all your input in a single 2D array.

</div>

Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality.

```{code-cell} ipython3
position_basis.transform(position[:, np.newaxis])
```

<div class="render-all">

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

</div>


<div class="render-user">
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
</div>

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position")
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed")
basis = position_basis + speed_basis
basis = basis.to_transformer()
basis
```



Let's create a single TsdFrame to hold all our inputs:

<div class="render-user render-presenter">

- Stack position and speed in a single TsdFrame to hold all our inputs:

</div>

```{code-cell} ipython3
:tags: [render-all]

transformer_input = nap.TsdFrame(
    t=position.t,
    d=np.stack([position, speed]).T,
    time_support=position.time_support,
    columns=["position", "speed"],
)
```

<div class="render-user render-presenter">

- Pass this input to our transformed additive basis. 
- Note that we do not need to call `set_input_shape` here because each basis element processes one column of the 2D input.

</div>

Our new additive transformer basis can then take these behavioral inputs and turn them into the model's design matrix.

```{code-cell} ipython3

basis.transform(transformer_input)
```

### Pipelines

We need one more step: scikit-learn cross-validation operates on an estimator, like our GLMs. if we want to cross-validate over the basis or its features, we need to combine our transformer basis with the estimator into a single estimator object. Luckily, scikit-learn provides tools for this: pipelines.

Pipelines are objects that accept a series of (0 or more) transformers, culminating in a final estimator. This is defined as a list of tuples, with each tuple containing a human-readable label and the object itself:

<div class="render-user render-presenter">

- If we want to cross-validate over the basis, we need more one more step: combining the basis and the GLM into a single scikit-learn estimator.
- [Pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) to the rescue!

</div>

<div class="render-user">
```{code-cell} ipython3
# set the reg strength to the optimal
glm = 
# pipe the basis and the glm
pipe = pipeline.Pipeline(
pipe
```
</div>

```{code-cell} ipython3
# set the reg strength to the optimal
glm = nmo.glm.PopulationGLM(solver_name="LBFGS", solver_kwargs={"tol": 10**-12})
pipe = pipeline.Pipeline([
    ("basis", basis),
    ("glm", glm)
])
pipe
```



This pipeline object allows us to e.g., call fit using the *initial input*:

<div class="render-user render-presenter">

- Pipeline runs `basis.transform`, then passes that output to `glm`, so we can do everything in a single line:
</div>

```{code-cell} ipython3
pipe.fit(transformer_input, count)
```

We then visualize the predictions the same as before, using `pipe` instead of `glm`.

<div class="render-user render-presenter">

- Visualize model predictions!

</div>

```{code-cell} ipython3
:tags: [render-all]

visualize_model_predictions(pipe, transformer_input)
```

### Cross-validating on the basis

<div class="render-all">

Now that we have our pipeline estimator, we can cross-validate on any of its parameters!

</div>

```{code-cell} ipython3
pipe.steps
```

Let's cross-validate on the number of basis functions for the position basis, and the identity of the basis for the speed. That is:

<div class="render-user render-presenter">

Let's cross-validate on:
- The number of the basis functions of the position basis
- The functional form of the basis for speed
- Let's retrieve the those attributes from the pipeline
</div>

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

For scikit-learn parameter grids, we use `__` to stand in for `.`:

<div class="render-user render-presenter">

- Construct `param_grid`, using `__` to stand in for `.`
- In scikit-learn pipelines, we access nested parameters using double underscores:
  - `pipe["basis"]["position"].n_basis_funcs` - normal Python syntax
  - `"basis__position__n_basis_funcs"` - scikit-learn parameter grid syntax

</div>

<div class="render-user">
```{code-cell} ipython3
param_grid = 
```
</div>

```{code-cell} ipython3
param_grid = {
    "basis__position__n_basis_funcs": [5, 10, 20],
    "basis__speed": [nmo.basis.MSplineEval(15),
                      nmo.basis.BSplineEval(15),
                      nmo.basis.RaisedCosineLinearEval(15)],
}
```


<div class="render-user render-presenter">

- Cross-validate as before:

</div>

<div class="render-user">
```{code-cell} ipython3
# define the grid search and fit
cv =
```
</div>

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
cv.fit(transformer_input, count)
```


<div class="render-user render-presenter">

- Investigate results:

</div>

```{code-cell} ipython3
:tags: [render-all]

pd.DataFrame(cv.cv_results_)
```

scikit-learn does not cache every model that it runs (that could get prohibitively large!), but it does store the best estimator, as the appropriately-named `best_estimator_`.

<div class="render-user render-presenter">

- Can easily grab the best estimator, the pipeline that did the best:

</div>

<div class="render-user">
```{code-cell} ipython3
# define the grid search and fit
best_estim =
best_estim
```
</div>

```{code-cell} ipython3
best_estim = cv.best_estimator_
best_estim
```



We then visualize the predictions of `best_estim` the same as before.

<div class="render-user render-presenter">

- Visualize model predictions!

</div>

```{code-cell} ipython3
:tags: [render-all]

visualize_model_predictions(best_estim, transformer_input)
```

(sklearn-feature-selection)=
## Feature selection

Now, finally, we understand almost enough about how scikit-learn works to figure out whether both position and speed are necessary inputs, i.e., to do feature selection. 

What we would like to do here is comparing alternative models: position + speed, position only or speed only. However, scikit-learn's cross-validation assumes that the input to the pipeline does not change, only the hyperparameters do. So, how do we go about model selection since we require different input for different model we want to compare?

Here is a neat NeMoS trick to circumvent that. scikit-learn's GridSearchCV assumes the INPUT stays the same across all models, but for feature selection, we want to compare models with different features (position + speed, position only, speed only). The solution: create a "null" basis that produces zero features, so all models take the same 2D input (position, speed) but some features become empty. First we need to define this "null" basis taking advantage of `CustomBasis`, which defines a basis from a list of functions.

<div class="render-user render-presenter">

Let's move on to feature selection. Our goal is comparing alternative models, for this example we will consider: position + speed, position only or speed only.

Problem: scikit-learn's cross-validation assumes that the input to the pipeline does not change, while each model will have a different input. What can we do? 

Let's see how to circumvent this with a neat basis trick.

- Create a "null" basis that produces zero features using `CustomBasis`, which defines a basis from a list of functions.

</div>

<div class="render-user">
```{code-cell} ipython3
# this function creates an empty array (n_sample, 0)
def func(x):
    return np.zeros((x.shape[0], 0))
# Create a null transformer basis using the custom basis class
null_basis = 
# this creates an empty feature
null_basis.compute_features(position).shape
```
</div>

```{code-cell} ipython3
# this function creates an empty array (n_sample, 0)
def func(x):
    return np.zeros((x.shape[0], 0))

# Create a null basis using the custom basis class
null_basis = nmo.basis.CustomBasis([func]).to_transformer()

# this creates an empty feature
null_basis.compute_features(position).shape
```



Why is this useful? Because we can use this `null_basis` and basis composition to do model selection. As a first step, we can notice that the original additive basis is stored as a `basis` attribute in the `TransformerBasis`.

<div class="render-user render-presenter">

- First we can note that the original "position + speed" additive basis is the `basis` attribute of the transformer.
</div>

```{code-cell} ipython3

pipe["basis"].basis
```

<div class="render-user render-presenter">

- Add the null basis to the speed or position basis to generate a composite basis for the position-only and speed-only model that receives the same 2D input as the model including all predictors!
</div>


<div class="render-user">
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
</div>

```{code-cell} ipython3
# define the 1D transformer bases with no label
position_bas = nmo.basis.MSplineEval(n_basis_funcs=10).to_transformer()
speed_bas = nmo.basis.MSplineEval(n_basis_funcs=15).to_transformer()

# combine them with each other or with the null basis to define each model.
basis_all = position_bas + speed_bas
basis_position = position_bas + null_basis
basis_speed = null_basis + speed_bas

# assign label (not necessary but nice)
basis_all.label = "position + speed"
basis_position.label = "position"
basis_speed.label = "speed"
```


<div class="render-user render-presenter">

- Create a parameter grid for each model of interest. 
- The attribute to cross-validate over is `"basis__basis"`, where the first "basis" is the name of the pipeline step, the second one is the attribute of the transformer.
</div>

<div class="render-user">
```{code-cell} ipython3
param_grid =
```
</div>

```{code-cell} ipython3
# then we create a parameter grid defining a grid of 2D basis for each model of interest
param_grid = {
    "basis__basis": 
    [
        basis_all,  
        basis_position, 
        basis_speed 
    ],
}
```

<div class="render-user">
```{code-cell} ipython3
# finally we define and fit our CV
cv =
```
</div>

```{code-cell} ipython3
# finally we define and fit our CV
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
cv.fit(transformer_input, count)
```


Let's now take a look to the model results.

```{code-cell} ipython3
:tags: [render-all]

cv_df = pd.DataFrame(cv.cv_results_)

# let's just plot a minimal subset of cols
cv_df[["param_basis__basis", "mean_test_score", "rank_test_score"]]
```

Unsurprisingly, position comes up as the predictor with the larger explnatory power and speed adds marginal benefits.

For the next project, you can use all the tools showcased here to find a better encoding model model for these hyppocampal neurons. 

Suggestions:

- Extend the model including the theta phase as predictor. 
- Use the NeMoS [multiplicative basis](https://nemos.readthedocs.io/en/latest/generated/_basis/nemos.basis._basis.MultiplicativeBasis.html) to capture model the interaction between theta phase and position. 


## Conclusion

<div class="render-all">

Various combinations of features can lead to different results. From this quick demo it looks like the position-only model is only marginally worst compared to the full model. 

  - [Hardcastle, Kiah, et al. "A multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal cortex." Neuron 94.2 (2017): 375-387](https://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf)

  - [McClain, Kathryn, et al. "Position–theta-phase model of hippocampal place cell activity applied to quantification of running speed modulation of firing rate." Proceedings of the National Academy of Sciences 116.52 (2019): 27035-27042](https://www.pnas.org/doi/abs/10.1073/pnas.1912792116)

  - [Peyrache, Adrien, Natalie Schieferstein, and Gyorgy Buzsáki. "Transformation of the head-direction signal into a spatial code." Nature communications 8.1 (2017): 1752.](https://www.nature.com/articles/s41467-017-01908-3)

## Project Ideas

Use what you learned here and compare models including the theta phase. You can model phase precession as an interaction term between position and theta phase; You can include an interaction term by using the basis multiplication operator, for more information see,

- [Background on basis composition](https://nemos.readthedocs.io/en/latest/background/basis/plot_02_ND_basis_function.html)


## References

The data in this tutorial comes from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

</div>
