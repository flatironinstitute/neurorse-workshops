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

This notebook can be downloaded as **{nb-download}`place_cells.ipynb`**. See the button at the top right to download as markdown or pdf.

:::

# Model selection

In this session we will keep working on the hippocampal place field recordings. In particular, we will learn how to model neural responses to multiple predictors: position, speed and theta phase. We will also learn how we can leverage GLM to assess which predictor is more informative for explaining the observed activity.

<div class="render-user">
Data for this notebook comes from recordings in the mouse hippocampus while the mouse runs on a linear track, which we [explored yesterday](../day1/phase_precession-users.md).
</div>

<div class="render-all">

## Learning objectives

- Review how to use pynapple to analyze neuronal tuning
- Learn how to combine NeMoS basis objects for modeling multiple predictors

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

# during development, set this to a lower number so everything runs faster. 
cv_folds = 5
```

## Pynapple

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

### Speed modulation


The speed at which the animal traverse the field is not homogeneous. Does it influence the firing rate of hippocampal neurons? We can compute tuning curves for speed as well as average speed across the maze. In the next block, we compute the speed of the animal for each epoch (i.e. crossing of the linear track) by doing the difference of two consecutive position multiplied by the sampling rate of the position.

<div class="render-user render-presenter">

- Compute animal's speed for each epoch.

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
```

Now that we have the speed of the animal, we can compute the tuning curves for speed modulation. Here we call pynapple [`compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves):

<div class="render-user render-presenter">

- Compute the tuning curve with pynapple's [`compute_tuning_curves`](https://pynapple.org/generated/pynapple.process.tuning_curves.html#pynapple.process.tuning_curves.compute_tuning_curves)

</div>

```{code-cell} ipython3
tc_speed = nap.compute_tuning_curves(spikes, speed, bins=20, epochs=speed.time_support, feature_names=["speed"])
```

<div class="render-user render-presenter">

- Visualize the position and speed tuning for these neurons.
</div>

```{code-cell} ipython3
:tags: [render-all]

fig = workshop_utils.plot_position_speed(position, speed, place_fields, tc_speed, neurons);
```

These neurons show a strong modulation of firing rate as a function of speed but we also notice that the animal, on average, accelerates when traversing the field. Is the speed tuning we observe a true modulation or spurious correlation caused by traversing the place field at different speeds? We can use NeMoS to model the activity and give the position and the speed as input variable.

<div class="render-user render-presenter">

These neurons all show both position and speed tuning, and we see that the animal's speed and position are highly correlated. We're going to build a GLM to predict neuronal firing rate -- which variable should we use? Is the speed tuning just epiphenomenal?

</div>

## NeMoS

(basis_eval_place_cells)=
### Basis evaluation

As we've seen before, we will use basis objects to represent the input values.  In previous tutorials, we've used the `Conv` basis objects to represent the time-dependent effects we were looking to capture. Here, we're trying to capture the non-linear relationship between our input variables and firing rate, so we want the `Eval` objects. In these circumstances, you should look at the tuning you're trying to capture and compare to the [basis kernels (visualized in NeMoS docs)](table_basis): you want your tuning to be capturable by a linear combination of them.

In this case, several of these would probably work; we will use [`MSplineEval`](nemos.basis.MSplineEval) for both, though with different numbers of basis functions.

Additionally, since we have two different inputs, we'll need two separate basis objects.

:::{note}

Later in this notebook, we'll show how to cross-validate across basis identity, which you can use to choose the basis.

:::

<div class="render-presenter">

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
</div>


<div class="render-user render-presenter">

- Create a separate basis object for each model input.
- Visualize the basis objects.
</div>

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10)
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15)
workshop_utils.plot_pos_speed_bases(position_basis, speed_basis)
```

However, now we have an issue: in all our previous examples, we had a single basis object, which took a single input to produce a single array which we then passed to the `GLM` object as the design matrix. What do we do when we have multiple basis objects?

We could call `basis.compute_features()` for each basis separately and then concatenated the outputs, but then we have to remember the order we concatenated them in and that behavior gets unwieldy as we add more bases.

Instead, NeMoS allows us to combine multiple basis objects into a single "additive basis", which we can pass all of our inputs to in order to produce a single design matrix:

<div class="render-user render-presenter">

- Combine the two basis objects into a single "additive basis"
</div>

```{code-cell} ipython3
# equivalent to calling nmo.basis.AdditiveBasis(position_basis, speed_basis)
basis = position_basis + speed_basis
```

<div class="render-user render-presenter">

- Create the design matrix!
- Notice that, since we passed the basis pynapple objects, we got one back, preserving the time stamps.
- `X` has the same number of time points as our input position and speed, but 25 columns. The columns come from  `n_basis_funcs` from each basis (10 for position, 15 for speed).

</div>

```{code-cell} ipython3
X = basis.compute_features(position, speed)
X
```

### Model learning

As we've done before, we can now use the Poisson GLM from NeMoS to learn the combined model:

<div class="render-user render-presenter">

- Initialize `PopulationGLM`
- Use the "LBFGS" solver and pass `{"tol": 1e-12}` to `solver_kwargs`.
- Fit the data, passing the design matrix and spike counts to the glm object.

</div>

```{code-cell} ipython3
glm = nmo.glm.PopulationGLM(
    solver_kwargs={"tol": 1e-12},
    solver_name="LBFGS",
)

glm.fit(X, count)
```

### Prediction

Let's check first if our model can accurately predict the tuning curves we displayed above. We can use the [`predict`](nemos.glm.GLM.predict) function of NeMoS and then compute new tuning curves

<div class="render-user render-presenter">

- Use `predict` to check whether our GLM has captured each neuron's speed and position tuning.
- Remember to convert the predicted firing rate to spikes per second!

</div>

```{code-cell} ipython3
# predict the model's firing rate
predicted_rate = glm.predict(X) / bin_size

# same shape as the counts we were trying to predict
print(predicted_rate.shape, count.shape)

# compute the position and speed tuning curves using the predicted firing rate.
glm_pos = nap.compute_tuning_curves(predicted_rate, position, bins=50, epochs=position.time_support, feature_names=["position"])
glm_speed = nap.compute_tuning_curves(predicted_rate, speed, bins=30, epochs=speed.time_support, feature_names=["speed"])
```

<div class="render-user render-presenter">

- Compare model and data tuning curves together. The model did a pretty good job!

</div>

```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pos, glm_speed);
```

<div class="render-all">

We can see that this model does a good job capturing both the position and the speed. In the rest of this notebook, we're going to investigate all the scientific decisions that we swept under the rug: should we regularize the model? what basis should we use? do we need both inputs?

To make our lives easier, let's create a helper function that wraps the above
lines, because we're going to be visualizing our model predictions a lot.

</div>

```{code-cell} ipython3
def visualize_model_predictions(glm, X):
    # predict the model's firing rate
    predicted_rate = glm.predict(X) / bin_size

    # compute the position and speed tuning curves using the predicted firing rate.
    glm_pos = nap.compute_tuning_curves(predicted_rate, position, bins=50, epochs=position.time_support, feature_names=["position"])
    glm_speed = nap.compute_tuning_curves(predicted_rate, speed, bins=30, epochs=position.time_support, feature_names=["speed"])

    workshop_utils.plot_position_speed_tuning(place_fields, tc_speed, glm_pos, glm_speed);
```

<div class="render-all">

In our previous analysis of the place field hyppocampal dataset we compared multiple encoding models and tried to figure out which predictor (position, speed or phase) had more explanatory power. In this notebook we will keep going on that effort and learn more principled (and convenient) approaches to model comparison combining NeMoS and scikit-learn.

## Learning Objectives

- Learn how to use NeMoS objects with [scikit-learn](https://scikit-learn.org/) for cross-validation
- Learn how to use NeMoS objects with scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- Learn how to use cross-validation to perform model and feature selection
- 
</div>

## Scikit-learn

### How to know when to regularize?

In the [last session](./head_direction.md), Edoardo fit the all-to-all connectivity of the head-tuning dataset using the Ridge regularizer. In the model above, we're not using any regularization? Why is that?

We have far fewer parameters here then in the last example. However, how do you know if you need regularization or not? One thing you can do is use cross-validation to see whether model performance improves with regularization (behind the scenes, this is what we did!). We'll walk through how to do that now.

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

```{code-cell} ipython3
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

- We got a warning because we didn't specify the regularizer strength, so we just fell back on default value.
- Let's investigate results:
</div>

Cross-validation results are stored in a dictionary attribute called `cv_results_`, which contains a lot of info. Let's convert that to a pandas dataframe for readability,

```{code-cell} ipython3
import pandas as pd

pd.DataFrame(cv.cv_results_)
```

The most informative for us is the `'mean_test_score'` key, which shows the average of `glm.score` on each test-fold. Thus, higher is better, and we can see that the UnRegularized model performs better.


### Select basis

We can do something similar to select the basis. In the above example, I just told you which basis function to use and how many of each. But, in general, you want to select those in a reasonable manner. Cross-validation to the rescue!

Unlike the glm objects, our basis objects are not scikit-learn compatible right out of the box. However, they can be made compatible by using the `.to_transformer()` method (or, equivalently, by using the `TransformerBasis` class)

<div class="render-user render-presenter">

- You can (and should) do something similar to determine how many basis functions you need for each input.
- NeMoS basis objects are not scikit-learn-compatible right out of the box.
- But we have provided a simple method to make them so:

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

</div>

Transformers only accept 2d inputs, whereas nemos basis objects can accept inputs of any dimensionality.

```{code-cell} ipython3
position_basis.transform(position[:, np.newaxis])
```

<div class="render-user render-presenter">

- If the basis is composite (for example, the addition of two 1D bases), the transformer will expect a shape of `(n_sampels, 1)` each 1D component. If that's not the case, you need to call `set_input_shape`:

</div>

If the basis has more than one component (for example, if it is the addition of two 1D bases), the transformer will expect an input shape of `(n_sampels, 1)` pre component. If that's not the case, you'll provide a different input shape by calling `set_input_shape`.

**Option 1)** One input per component:

```{code-cell} ipython3
# generate a composite basis
basis_2d = nmo.basis.MSplineEval(5) +  nmo.basis.MSplineEval(5)
basis_2d = basis_2d.to_transformer()

# this will work: 1 input per component
x, y = np.random.randn(10, 1), np.random.randn(10, 1)
X = np.concatenate([x, y], axis=1)
result = basis_2d.transform(X)
```

**Option 2)** Multiple input per component.

<div class="render-user render-presenter">

- Then you can call transform on the 2d input as expected.
</div>

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

<div class="render-all">

- You can, equivalently, call `compute_features` *before* turning the basis into a transformer. Then we cache the shape for future use:

</div>

```{code-cell} ipython3
position_basis = nmo.basis.MSplineEval(n_basis_funcs=10, label="position")
position_basis.compute_features(position)
position_basis = position_basis.to_transformer()
speed_basis = nmo.basis.MSplineEval(n_basis_funcs=15, label="speed").to_transformer().set_input_shape(1)
basis = position_basis + speed_basis
basis
```

Let's create a single TsdFrame to hold all our inputs:

<div class="render-user render-presenter">

- Create a single TsdFrame to hold all our inputs:
</div>

```{code-cell} ipython3
:tags: [render-all]

transformer_input = nap.TsdFrame(
    t=position.t,
    d=np.stack([position, speed], 1),
    time_support=position.time_support,
    columns=["position", "speed"],
)
```

<div class="render-user render-presenter">

- Pass this input to our transformed additive basis:
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
</div>

```{code-cell} ipython3
print(pipe["basis"]["position"].n_basis_funcs)
print(pipe["basis"]["speed"])
```

For scikit-learn parameter grids, we use `__` to stand in for `.`:

<div class="render-user render-presenter">

- Construct `param_grid`, using `__` to stand in for `.`
- In sklearn pipelines, we access nested parameters using double underscores:
  - `pipe["basis"]["position"].n_basis_funcs` ← normal Python syntax
  - `"basis__position__n_basis_funcs"` ← sklearn parameter grid syntax

</div>

```{code-cell} ipython3
param_grid = {
    "basis__position__n_basis_funcs": [5, 10, 20],
    "basis__speed": [nmo.basis.MSplineEval(15).set_input_shape(1),
                      nmo.basis.BSplineEval(15).set_input_shape(1),
                      nmo.basis.RaisedCosineLinearEval(15).set_input_shape(1)],
}
```

<div class="render-user render-presenter">

- Cross-validate as before:
</div>

```{code-cell} ipython3
cv = model_selection.GridSearchCV(pipe, param_grid, cv=cv_folds)
cv.fit(transformer_input, count)
```

<div class="render-user render-presenter">

- Investigate results:
</div>

```{code-cell} ipython3
pd.DataFrame(cv.cv_results_)
```

scikit-learn does not cache every model that it runs (that could get prohibitively large!), but it does store the best estimator, as the appropriately-named `best_estimator_`.

<div class="render-user render-presenter">

- Can easily grab the best estimator, the pipeline that did the best:
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

### Feature selection

Now, finally, we understand almost enough about how scikit-learn works to figure out whether both position and speed are necessary inputs, i.e., to do feature selection. 

What we would like to do here is comparing alternative models: position + speed, position only or speed only. However, scikit-learn's cross-validation assumes that the input to the pipeline does not change, only the hyperparameters do. So, how do we go about model selection since we require different input for different model we want to compare?

Here is a neat NeMoS trick to circumvent that. scikit-learn's GridSearchCV assumes the INPUT stays the same across all models, but for feature selection, we want to compare models with different features (position + speed, position only, speed only). The solution: create a "null" basis that produces zero features, so all models take the same 2D input (position, speed) but some features become empty. First we need to define this "null" basis taking advantage of `CustomBasis`, which defines a basis from a list of functions.

```{code-cell} ipython3
# this function creates an empty array (n_sample, 0)
def func(x):
    return np.zeros((x.shape[0], 0))

# Create a null basis using the custom basis class
null_basis = nmo.basis.CustomBasis([func]).to_transformer()

# this creates an empty feature
null_basis.compute_features(position).shape
```

Why is this useful? Because we can use this `null_basis` and basis composition to do model selection.

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

Now use all the tools that you learned to find a better model for this dataset!


## Conclusion

Various combinations of features can lead to different results. Feel free to explore more. To go beyond this notebook, you can check the following references :

  - [Hardcastle, Kiah, et al. "A multiplexed, heterogeneous, and adaptive code for navigation in medial entorhinal cortex." Neuron 94.2 (2017): 375-387](https://www.cell.com/neuron/pdf/S0896-6273(17)30237-4.pdf)

  - [McClain, Kathryn, et al. "Position–theta-phase model of hippocampal place cell activity applied to quantification of running speed modulation of firing rate." Proceedings of the National Academy of Sciences 116.52 (2019): 27035-27042](https://www.pnas.org/doi/abs/10.1073/pnas.1912792116)

  - [Peyrache, Adrien, Natalie Schieferstein, and Gyorgy Buzsáki. "Transformation of the head-direction signal into a spatial code." Nature communications 8.1 (2017): 1752.](https://www.nature.com/articles/s41467-017-01908-3)

## References

<div class="render-all">

The data in this tutorial comes from [Grosmark, Andres D., and György Buzsáki. "Diversity in neural firing dynamics supports both rigid and learned hippocampal sequences." Science 351.6280 (2016): 1440-1443](https://www.science.org/doi/full/10.1126/science.aad1935).

</div>

```{code-cell} ipython3

```
