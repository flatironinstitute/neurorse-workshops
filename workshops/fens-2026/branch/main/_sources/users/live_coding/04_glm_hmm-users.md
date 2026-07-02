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

import warnings

warnings.filterwarnings(
    "ignore",
    message="coroutine .* was never awaited",
    category=RuntimeWarning,
)
```
:::{admonition} Jupyter Lab Reminders
:class: important render-all

Reminder to presenter: Go to `View > Appearance`, select `Simple Interface` and turn off everything else to hide as many bars as possible. And maybe activate `Presentation Mode`.

And turn on `View > Render side-by-side` (shortcut `Shift+R`).
:::
# Infer behavioral strategies during decision making with GLM-HMMs
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/live_coding/04_glm_hmm.md)


In this notebook, we will learn how to model **behavioral choices** by fitting a **GLM-HMM**. 


## Introduction
### Behavioral choices: dataset


Data for this notebook comes from the IBL decision-making task (IBL et al., 2021) <span id="cite1a"></span><a href="#ref1a">[1a]</a>, a variation of the two-alternative forced-choice perceptual detection task (Burgess et al., 2021 <span id="cite2a"></span><a href="#ref2a">[2a]</a>).


During this task, a sinusoidal grating with varying contrast [0\%-100\%] appeared either at the right or left side of the screen. The mice indicated this side by turning a small wheel, which moved the stimulus toward the center of the screen (Burgess et al., 2021 <span id="cite2b"></span><a href="#ref2b">[2b]</a>). If the mice chose the side correctly, they would receive a water reward; if not, they would get a noise burst and a 1-second timeout. For the first 90 trials of each session, the stimulus appeared on the left or right side with 50% probability and then this probability shifts, biasing towards one side or the other, alternating randomly every 20–100 trials. 




![Task illustration](../../_static/IBL_edited.png)

*Task illustration. Modified from IBL et al. (2021)* <span id="cite1b"></span><a href="#ref1b">[1b]</a>.


### GLM-HMM

GLM-HMMs are composed by a an HMM component governing the distribution over the latent states, and state-specific GLMs, which specify the activity of the system at each state.



![GLM HMM Graphical model](../../_static/graphical_model_glm_hmm_actions.png)

*Graphical model" of a GLM-HMM with mouse actions.*



We can fully describe a system by three elements:

1. Initial probabilities: probability distribution of the first state.
2. Transition probabilities: how the states evolve over time
3. Emission probabilities: relationship between state and observation


### How will we learn?

We will replicate the main findings of Ashwood et al. (2022) <span id="cite3a"></span><a href="#ref3a">[3a]</a>. 

![Ashwood paper figures to replicate](../../_static/ashwood_paper_figs.svg)


### Tutorial sections


1. Download and preprocess the data
2. Build a design matrix with three predictors
3. Fit the model 
4. Interpret the results


## 01. Download and preprocessing of data
:::{admonition} What do we want to do in this subsection?
:class: attention render-all

1. Download the dataset
2. See what it contains
3. Select the sessions we are interested in

:::
### Data Streaming

Let's download the data using  <a href="https://int-brain-lab.github.io/ONE/one_reference.html">Open Neurophysiology Environment (ONE)</a>

```{code-cell} ipython3
:tags: [render-all]

# Imports
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from one.api import ONE
import nemos as nmo
import workshop_utils
import jax

# Enable 64-bit floating-point and integer types
jax.config.update("jax_enable_x64", True)

# Plotting params
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)#, context="notebook")
```
```{code-cell} ipython3
:tags: [render-all]

# Configure 
ONE.setup(
    base_url='https://openalyx.internationalbrainlab.org', 
    silent=True
)
```
```{code-cell} ipython3
:tags: [render-all]

# Set up where we want our data to be downloaded
data_dir = os.environ.get("NEMOS_DATA_DIR")
print("IBL data dir:", data_dir)
```

```{code-cell} ipython3
# Instantiate ONE object
one = # Complete
```

```{code-cell} ipython3
:tags: [render-all]

# Then we need to choose our subject and run load_aggregate
subject = "CSHL_008"
trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')

# We can see the type of object we are dealing with
print(type(trials))
```
```{code-cell} ipython3
:tags: [render-all]

# We can see how it looks like
trials.head(5)
```
:::{admonition} Working without pynapple
:class: note render-all

Unlike the other notebooks in this workshop, here we work directly with `pandas` DataFrames and `numpy` arrays rather than pynapple objects. The IBL trial data is trial based, with no continuous time axis. Pynapple would not represent this well, so we keep it as a DataFrame and show how NeMoS interfaces with plain NumPy. NeMoS also accepts pynapple `Tsd`/`TsdFrame` objects directly, and we point out as we go where that would change the workflow (for example, how session boundaries are handled).
:::
```{code-cell} ipython3
:tags: [render-all]

# We can see the information we get by printing the columns
trials.columns
```


We are modeling choice as result of observables and behavioral state, so we need `choice`, `contrastLeft`, `contrastRight` and `feedbackType`. Additionally, we want to keep the information of the probability of the stimulus appearing in a given position, `probabilityLeft` since this changes within a session, and the `session` ids to know when sessions start and end.




Let's extract what we need, and inspect its contents.


```{code-cell} ipython3
:tags: [render-all]

trials = trials[
    [
        "choice", "contrastLeft", "contrastRight", 
        "feedbackType", "probabilityLeft", "session"
    ]
]

print(f"choice \nvalues: {np.sort(trials.choice.unique())}, data type: {trials.choice.dtype} \n")
print(f"contrast left \nvalues: {np.sort(trials.contrastLeft.unique())}, data type: {trials.contrastLeft.dtype} \n")

print(f"contrast right \nvalues: {np.sort(trials.contrastRight.unique())}, data type: {trials.contrastRight.dtype} \n")

print(f"reward \nvalues: {np.sort(trials.feedbackType.unique())}, data type: {trials.feedbackType.dtype} \n")

print(f"probability of stimulus on left \nvalues: {np.sort(trials.probabilityLeft.unique())}, data type: {trials.probabilityLeft.dtype} \n")

print(f"session \n(some) values: {trials.session.unique()[:5]}, data type: {trials.session.dtype}\n")
```


<table style="border-collapse: collapse; width: 100%; font-size: 0.95em;">
  <thead>
    <tr style="background-color: #2c3e50; color: #ffffff;">
      <th style="padding: 8px 12px; text-align: left; border: 1px solid #ccc; white-space: nowrap; width: 180px;">Variable</th>
      <th style="padding: 8px 12px; text-align: left; border: 1px solid #ccc;">Description</th>
    </tr>
  </thead>
  <tbody>
    <tr style="background-color: #f6f8fa;">
      <td style="padding: 8px 12px; border: 1px solid #ccc; font-family: monospace; font-weight: bold; color: #2c3e50; text-align: left; white-space: nowrap; width: 180px;">choice</td>
      <td style="padding: 8px 12px; border: 1px solid #ccc; text-align: left;">mouse choice: 1 = choice left, -1 = choice right, 0 = violation (no response within the trial period). Since we are going to use a Bernoulli GLM, we will remap the variables to 1 = choice left and 0 = choice right at the end of preprocessing.</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="padding: 8px 12px; border: 1px solid #ccc; font-family: monospace; font-weight: bold; color: #2c3e50; text-align: left; white-space: nowrap; width: 180px;">contrastLeft</td>
      <td style="padding: 8px 12px; border: 1px solid #ccc; text-align: left;">contrast of stimulus presented on the left</td>
    </tr>
    <tr style="background-color: #f6f8fa;">
      <td style="padding: 8px 12px; border: 1px solid #ccc; font-family: monospace; font-weight: bold; color: #2c3e50; text-align: left; white-space: nowrap; width: 180px;">contrastRight</td>
      <td style="padding: 8px 12px; border: 1px solid #ccc; text-align: left;">contrast of stimulus presented on the right</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="padding: 8px 12px; border: 1px solid #ccc; font-family: monospace; font-weight: bold; color: #2c3e50; text-align: left; white-space: nowrap; width: 180px;">feedbackType</td>
      <td style="padding: 8px 12px; border: 1px solid #ccc; text-align: left;">reward obtained: 1 = success, -1 = failure</td>
    </tr>
    <tr style="background-color: #f6f8fa;">
      <td style="padding: 8px 12px; border: 1px solid #ccc; font-family: monospace; font-weight: bold; color: #2c3e50; text-align: left; white-space: nowrap; width: 180px;">probabilityLeft</td>
      <td style="padding: 8px 12px; border: 1px solid #ccc; text-align: left;">probability of stimulus being presented on the left of the screen</td>
    </tr>
    <tr style="background-color: #ffffff;">
      <td style="padding: 8px 12px; border: 1px solid #ccc; font-family: monospace; font-weight: bold; color: #2c3e50; text-align: left; white-space: nowrap; width: 180px;">session</td>
      <td style="padding: 8px 12px; border: 1px solid #ccc; text-align: left;">id of session</td>
    </tr>
  </tbody>
</table>

### Preprocessing: keeping only the relevant sessions and trials


Now we will select the sessions we will fit the model to. First let's see how the probability of the stimulus appearing on one side changes as trials progress in a session.


```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_proba_left(trials)
```


We will apply three restrictions to match Ashwood et al. (2022) <a href="#ref1b">[1b]</a>:

1. Only keep sessions in which the animal went through the entire training criteria 
2. Select blocks of 50-50 within those sessions
3. Keep the blocks with less than 10 violations


```{code-cell} ipython3
:tags: [render-all]

viol_val = 0

df_trials, valid_sessions = workshop_utils.select_sessions(
    trials,
    max_violations=10,
    violation_value=viol_val
)

print(f"# of sessions before restrictions {trials['session'].nunique()}")
print(f"# of sessions after restrictions {df_trials['session'].nunique()}")
```
:::{admonition} What did we do in this subsection?
:class: attention render-all

1. Downloaded the dataset
2. Inspected its contents
3. Selected the sessions and blocks we want to fit the model to!

:::
## 02. Building the design matrix
:::{admonition} What do we want to do in this subsection?
:class: attention render-all

1. Preprocess and explain the predictors we will use to build our design matrix
2. Present different basis objects
3. Build our design matrix, which we will use an input for the GLM-HMM.
:::
### Select valid trials


First we want to take a subset of all trials. We want to keep only the valid trials. That is, trials in which the animal made a choice.



```{code-cell} ipython3
# Create boolean array for valid choices (Valid: True; Invalid: False)
valid_choices_bool = # Complete
valid_choices_bool
```



Then we will get the index of the valid trials with `np.flatnonzero`



```{code-cell} ipython3
valid_choices_idx = # Complete
valid_choices_idx
```


We are interested in building a design matrix with three predictors: previous choice, win stay lose shift and signed contrast.

- Previous choice: lagged version of current choice. We need to create an array for `choice`. 
- Win stay lose shift: interaction between past choice and outcome. We need to create arrays for `choice`, `feedbackType`.
- Signed contrast: sensory evidence in 1D. We need to create arrays for `contrastleft`, `contrastRight`.



So we can create the arrays for the variables we need, and keep only the valid trials.




```{code-cell} ipython3
# We can select all the necessary values for the design matrix: 
choices = df_trials['choice'].values[valid_choices_idx]
stim_left = df_trials['contrastLeft'].values[valid_choices_idx]
stim_right = df_trials['contrastRight'].values[valid_choices_idx]
rewards = df_trials['feedbackType'].values[valid_choices_idx]
```


### Predictor 1: previous choice



![Previous choice](../../_static/previous_choice_table.svg)





Previous choice is a lagged version of current choice. Represents serial dependence on decisions. For this, we can use a ```HistoryConv```basis.

```HistoryConv``` includes the past values of a sample as predictors (raw history). You choose how far back to go; here we only need one trial in the past (`window_size=1`). We use this to create the previous-choice predictor.



```{code-cell} ipython3
# Prev history with history of 1
prev_choice_basis = # Complete
```



We can make an example quickly to show what happens if we use ```compute_features``` with a list. 



```{code-cell} ipython3
# Example
# Complete
```




We get a lagged list. You can notice that the first element is a ```NaN```. This is because a history feature is defined using past trials. Since there is no past trial for the first trial, the feature is undefined, and NeMoS fills this with a ```NaN```.


### Predictor 2: WSLS



![WSLS](../../_static/wsls_table.svg)





WSLS is an interaction of previous choice with previous reward: $WSLS_t = c_{t-1} \cdot r_{t-1}$. If a choice was rewarded on the previous trial, the predictor signals to "stay" (repeat that choice); if it was not rewarded, it signals to "switch" to the other alternative.

To capture interaction between variables, we can use a [multiplicative basis object](https://nemos.readthedocs.io/en/latest/background/basis/plot_02_ND_basis_function.html#n-dimensional-basis), which in this case performs an element wise multiplication.



```{code-cell} ipython3
# Create lagged reward basis
prev_reward_basis = # Complete
# Multiply lagged reward basis with the lagged choice basis
wsls_basis = # Complete
# Print
print(wsls_basis)
```



We can see what this is doing by using an example.



```{code-cell} ipython3
# Example
# Complete
```



The result is an element-wise multiplication, shifted by one. The first element is $1×1=1$, the second is $2×0=0$, and so on. The shift happens for the same reason as the previous choice predictor: NeMoS applies the computation only where it is well-defined, and pads with NaN where it is not. Since a history feature depends on past trials, it is undefined for the first trial — so NeMoS fills that position with NaN.

### Predictor 3: stimulus contrast



![Stim contrast](../../_static/signed_contrast_table.svg)





Now we need our third predictor, signed contrast. This encodes sensory evidence in 1D. Within this predictor, magnitude reflects strength of evidence and sign encodes direction. 


#### Preparation: get signed contrast 1d vector and subset valid trials


We need to create our signed contrast 1d vector for our signed_contrast predictor. 

- Replace `NaN` contrast values with `0` using `np.nan_to_num`.
- Compute the signed contrast (difference between left and right)



```{code-cell} ipython3
# Replace nans with 0s
stim_left = # Complete
stim_right = # Complete
# Compute the signed contrast (left - right)
signed_contrast = # Complete
# print the signed contrast for the first valid session
select_session = df_trials["session"] == valid_sessions[0]
signed_contrast[select_session]
```



We want to keep our predictor as it is. However we also need to have it as a NeMoS basis object so we can create our design matrix. For that, we will use the```IdentityEval``` basis.

A `IdentityEval` basis is used to include the samples themselves as predictors. This may seem pointless, but it will allow us to have our predictor as a nemos basis object.



```{code-cell} ipython3
# Identity basis for stimuli
stimuli_basis = # Complete
```


```{code-cell} ipython3
# Example
# Complete
```


### Combining features and computing them


Now that we have all our bases, we can combine them into an additive basis and apply the transformation to the input data using ```compute_features```.



- Use [basis addition](https://nemos.readthedocs.io/en/latest/background/basis/plot_02_ND_basis_function.html#additive-basis-object) to define a basis which concatenates predictor.



```{code-cell} ipython3
# Create an additive basis using our three components
# Stimuli, wsls & previous choice
basis_object = # Complete
print(basis_object)
```




- Create the design matrix by calling `compute_features`.



```{code-cell} ipython3
# Call compute features to get the raw model design
X_unnormalized = basis_object.compute_features(
    # input 1 : processed with stimuli_basis
    # Complete
    # input 2 : wsls input 1: choice
    # Complete
    # input 3 : wsls input 2: reward
    # Complete
    # input 4 : processed with prev_choice
    # Complete
)
X_unnormalized[10:15]
```



As a last step, we normalize the signed-contrast predictor.

- Z-score the contrast values.



```{code-cell} ipython3
from scipy.stats import zscore
# Copy the array (we'll need the un-normalized later)
X = np.copy(X_unnormalized)
# Apply z-scoring
X[:, 0] = zscore(X[:, 0])
X[10:15]
```

:::{admonition} Why do we normalize our stimuli predictor?
:class: question render-user
:class: dropdown

When fitting a GLM-HMM, we are fitting a separate weight for each feature. However, if the features are on different numerical scales for reasons that are not related to the actual influence of each predictor, that renders the weights incomparable. Here we have three predictors:  
- (1) Previous choice and (2) WSLS are always exactly −1 or +1. Their values are discrete and bounded, and they already share the same scale.
- (3) Stimuli contrast is continuous. While it can reach −1 or +1 (full contrast), this value rarely occurs. 

Because the contrast values are typically much smaller in magnitude than ±1, the model compensates by assigning them a larger weight to match the output scale — purely because the values are numerically smaller. In practice, this is an artifact of scale that does not reflect the true influence of the predictor.

By normalizing, we rescale the predictor to have mean 0 and standard deviation 1. Previous choice and WSLS are already on a unit scale by construction — their values are symmetric around zero and their spread is naturally 1. This is why we only normalize signed contrast.
:::


We can see our design matrix now!



```{code-cell} ipython3
:tags: [render-all]

# Plot an heatmap showing the model design
workshop_utils.plot_design_matrix(X, choices);
```
:::{admonition} What did we do in this subsection?
:class: attention render-all

1. We started with raw behavioral variables (choices, rewards, and contrasts) and identified the predictors we wanted to include: signed contrast, previous choice, and WSLS.
2. Using ```IdentityEval```, ```HistoryConv``` and multiplicative basis, we defined basis functions for each predictor and combined them into a single additive basis.
3. We ended with a design matrix, generated with ```compute_features```, ready to be used as input to the GLM-HMM.
:::
## 03. Model fitting
:::{admonition} What do we want to do in this subsection?
:class: attention render-all

1. Convert choices so we can model them with a Bernoulli GLM-HMM
2. Generate a vector containing session starts to use it in fitting
3. Initialize our ```GLMHMM``` object and fit our model with ```nmo.fit()```
:::
### Converting choices

We are going to fit a Bernoulli GLM-HMM to model binary choices. For a Bernoulli GLM-HMM, observations must take values of 0 or 1. 

- In the current dataset, choices are encoded as:
  - `1` = left choice
  - `-1` = right choice
- We therefore have to remap right choices from `-1` to `0`.
  - `1` = left choice
  - `0` =  right choice




```{code-cell} ipython3
choices[choices==-1] = 0
print(choices)
```


### Creating session boundaries

Importantly, we don't fit all the trials as one continuous block. The data come as separate sessions that the mouse completed over multiple days, and we fit the model on all of them together. For our model to be accurate, we need to tell it when our session boundaries are: we don't want it to compute all sessions as if they were one. 

Let's see an example of session break.




```{code-cell} ipython3
# Create a session array
session = df_trials['session'].values

# Session break
session[88:92]
```




We can pass the session boundaries in different ways. If we are using numpy arrays, we can pass an array of booleans indicating True for the beginning of each session or an array with the indices of the session changes. If we were using Pynapple, the session boundaries would be inherited from the Pynaple objects themselves.

Now, we will build a boolean array to indicate the session changes.



```{code-cell} ipython3
# Session transitions
session_changes = # Complete
session_changes[88:92]
```



Our sessions are shifted? And what about the first session?




```{code-cell} ipython3
# See first session
session_changes[:10]
```



```{code-cell} ipython3
session_starts = # Complete
session_starts[88:92]
```


### Initialize & fit GLM-HMM


Let's initialize the ```GLMHMM``` object. The only required parameter is the number of states. 

Ashwood et al. (2022) <span id="cite3b"></span><a href="#ref3b">[3b]</a> found that most mice used 3 decision-making states when performing this task. Following that work, we will initialize our ```GLMHMM``` object with 3 states.  

We will also set `regularizer="Ridge"` to penalize large weights, and a seed for our initial parameters (`jax.random.PRNGKey(number)`).



```{code-cell} ipython3
n_states = 3
seed=jax.random.PRNGKey(12)
model = # Complete
model
```



Fit the model providing the `session_starts` markers as the `session_starts` argument of `model.fit`.



```{code-cell} ipython3
# Fit the model
# Complete
```

:::{admonition} How would this be different if we were using Pynapple objects?
:class: note render-all
:class: dropdown

In NeMoS we have two ways of indicating the beginning of a new session. You can use a Pynapple Tsd or TsdFrame to demarcate sessions, in which case session demarcations are inherited from the pynapple objects. Alternatively, when using a design matrix and a choice vector that are Numpy objects, it is necessary to pass a session indicator. This can be:
- a boolean array or integer array of 1s and 0s indicating session starts, shape ``(n_samples,)``
- an integer array of indices marking session starts, shape ``(n_sessions,)``
- a pynapple.IntervalSet marking session epochs (requires either X or y to be a pynapple Tsd or TsdFrame to get timestamps)

:::
:::{admonition} What did we do in this subsection?
:class: attention render-all

1. Started with the design matrix and behavioral choices.
2. Converted choices into a binary format suitable for a Bernoulli GLM-HMM.
3. Identified the session boundaries and created the session-start vector required for fitting.
4. Instantiated a `GLMHMM` object and fit the model using `nmo.fit()`.
5. Ended with a fitted GLM-HMM ready for inspection and analysis.
:::
## 04. Interpreting the results
:::{admonition} What do we want to do in this subsection?
:class: attention render-all

1. Inspect the output of the model
2. Interpret glm weights and transition matrix
3. Use NeMoS built in functions to visualize and interpret temporal structure of state transitions
:::
### How to visualize the fitted parameters


Latent state labels are arbitrary. Below we permute those labels to match that of the reference paper.


```{code-cell} ipython3
:tags: [hide-input, render-all]

model = workshop_utils.relabel(model)
```


The GLM coefficients and intercept, and the HMM initial and transition probabilities are stored in the following attributes:

- `model.coef_`
- `model.intercept_`
- `model.initial_prob_`
- `model.transition_prob_`

Let's print them

```{code-cell} ipython3
:tags: [render-all]

print("GLM parameters\n==============")
print(f"glm weights:\n{model.coef_}\n")
print(f"intercept:\n{model.intercept_}")

print("\n\nHMM parameters\n==============")
print(f"transition matrix \n {model.transition_prob_}\n")
print(f"initial probabilities \n {model.initial_prob_}")
```

Let's see what type of information we can gather.

### Interpreting the GLM weights

We can plot the GLM weights obtained for our 3-state model.

```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_glm_weights(model)
```


State 1 ("engaged") has a large positive weight on the stimulus and weights close to zero on all other predictors, suggesting the animal is primarily driven by sensory information in this state. States 2 ("biased left") and 3 ("biased right") show large bias weights of opposite signs (positive for state 2 and negative for state 3), which indicates a systematic tendency to choose left or right regardless of the stimulus. All three states have very low weights on previous choice and WSLS, suggesting these predictors play little role in driving behavior.
 

:::{admonition} How does this look in the original paper?
:class: question render-all
:class: dropdown

![Ashwood paper GLM-HMM weights](../../_static/glm_weights_ashwood.svg)

:::
### Interpreting the transition matrix


We can also see the fitted transition matrix for our three-state model. This describes the transition probabilities among the different states, each corresponding to a different decision-making strategy. 

The utility function below plots the heatmap.


```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_transition_matrix(model);
```


The diagonal entries are all high, which indicates that each state is highly self-persistent. That is, once the animal enters a state, it is very likely to remain in that state on the next trial. Off-diagonal transitions are rare, meaning switches between states occur infrequently. 

:::{admonition} How does this look in the original paper?
:class: question render-all
:class: dropdown

![Ashwood paper GLM-HMM weights](../../_static/transition_matrix_ashwood.svg)

:::
### Using ```smooth_proba``` to visualize and interpret posterior state probabilities


To better understand the temporal structure of decision making behavior, we can compute the probability of being in each state at each trial, conditioned on the entire observed sequence. For this, we can use ```smooth_proba```. This method uses the forward-backward algorithm to incorporate information from past and future observations. 

- Call `smooth_proba` to compute the smoothing posterior probabilities of the latent states.
- Remember to provide the `session_starts=session_starts` parameter to mark the beginning of each session.
- Filter non-nan entries and check that the posterior sums to 1 over the states.



```{code-cell} ipython3
# Compute smooth_proba
posteriors = # Complete
print(f"First five posteriors \n{posteriors[:5]} \n")
# Each (valid) posterior row sums to 1
valid = ~np.isnan(posteriors).any(axis=1)
print(
    "Does the posterior sum to one?", 
    np.allclose(posteriors[valid].sum(axis=1), 1)
)
```


Let's plot the first 90 trials, corresponding to the first session.

```{code-cell} ipython3
:tags: [render-all]

colors = ["#ff7f00", "#4daf4a", "#377eb8"]
for i, c in enumerate(colors):
    plt.plot(posteriors[4950:5039, i], color=c)
```

Let's now use the utility function to plot the three sessions shown in Fig. 3a of <span id="cite3c"></span><a href="#ref3c">[3c]</a>.

```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_posteriors(posteriors, session);
```


In these sessions, we can see that the posterior over latent states across multiple trials. We can see strong confidence in state assignments and extended periods where a single state persists across consecutive trials. 


:::{admonition} How does this look in the original paper?
:class: question render-all
:class: dropdown

![Ashwood paper GLM-HMM weights](../../_static/posteriors_ashwood.svg)

:::
### Understanding mouse behavior in different states
#### Most likely sequence of states with Viterbi


We may also want to quantfy fractional occupancies and accuracies. These give us information of how often the mouse was at each given state and how well it performed in the task at each state.

For this, we need the inferred sequence of states, this can be obtained using the [Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm) that you can run by calling the  ```decode_state``` method.

This method finds the single most likely sequence of hidden states that best explains the observed data: the state sequence that maximizes the joint probability of states and observations. 




- Get the most likely sequence of states given the observation by calling `decode_state`, which runs the Viterbi (also known as max-sum) algorithm.
- Remember to provide the `session_starts=session_starts`



```{code-cell} ipython3
# get output of viterbi in one-hot encoding
decoded_states = # Complete
decoded_states
```

#### Fractional occupancies

From this we can compute the fractional occupancy, while correctly filtering out the NaNs.

- Compute the fractional occupancy for each state.
- Remember that the decoded states may contain NaNs.


```{code-cell} ipython3
:tags: [render-all]

# calculate occupancy
print(f"Not nan? \n {~np.isnan(decoded_states)}")
```
```{code-cell} ipython3
:tags: [render-all]

valid = np.all(~np.isnan(decoded_states), axis=1)
print(f"valid? \n {valid}")
```
```{code-cell} ipython3
:tags: [render-all]

frac_occupancy_viterbi= np.nansum(decoded_states, axis=0) / valid.sum()
print(f"Fraction of occupancy \n {frac_occupancy_viterbi} \n")
```
#### Fractional accuracies

Now we can compute the mouse's overall accuracy. For this:

- Mask out the 0 contrast stimuli (because there is no correct answer in that case)
- `choice==1`: correct choice
- store in array for plotting


```{code-cell} ipython3
# Compute boolean mask for nonzero signed contrast
mask = # Complete
```


```{code-cell} ipython3
# Compute correct choices
correct_choices = # Complete
correct_choices
```


```{code-cell} ipython3
# Compute the total accuracy applying the mask
total_accuracy = # Complete
total_accuracy
```


```{code-cell} ipython3
# Store in an array of dim 4
accuracies_to_plot_viterbi = np.zeros(4)
accuracies_to_plot_viterbi[0] = total_accuracy
accuracies_to_plot_viterbi
```



And then we can use our output of ```decode_state``` to segment the trials into the estimated states and compute the accuracy within each state. We can think about this as seeing whether the animal performs better or worse depending on the state that theyre in



- Loop over the states and apply the same calculation to get the accuracy per state.



```{code-cell} ipython3
accuracy_per_state = np.zeros(n_states)
for s in range(n_states):
  in_state = # Complete
  accuracy_per_state[s] = # Complete
accuracies_to_plot_viterbi[1:] = accuracy_per_state
accuracies_to_plot_viterbi
```



And we can plot this :)


```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_accuracy_and_occupancy(
    frac_occupancy_viterbi, 
    accuracies_to_plot_viterbi
);
```


According to state occupancy derived with the Viterbi algorithm, this mouse spent the majority of the trials in the engaged state and a lesser portion of trials in the other two states. We can see that even though this mouse had an overall accuracy of 80, it achieved a higher accuracy of 87% in the "engaged" state compared to 66% and 63% in the "bias left" and "bias right", respectively.

This makes sense considering that the information needed to well perform the task was the signed contrast.


:::{admonition} How does this look in the original paper?
:class: question render-all
:class: dropdown

![Ashwood paper GLM-HMM weights](../../_static/frac_occupancy_ashwood.svg)

:::
:::{admonition} What did we do in this subsection?
:class: attention render-all

1. Started with a fitted GLM-HMM.
2. Inspected the model outputs, including the GLM weights and transition matrix.
3. Used `smooth_proba` to compute posterior state probabilities and examine the temporal structure of behavior.
4. Used `decode_state` to infer the most likely state sequence for each trial.
5. Computed summary statistics from the decoded states, including fractional occupancy and accuracy per state.
:::
## Conclusion


In this notebook, we replicated the core findings of Ashwood et al. (2022) using NeMoS, demonstrating that mice alternate between discrete behavioral strategies during perceptual decision-making. Here is what we covered:

1. **Download and preprocessing of IBL data**: we showed how to obtain a dataset from the International Brain Laboratory using ONE, and how preprocess it to fit the model to it.
2. **Design matrix construction**: we transformed raw behavioral variables into three interpretable predictors (signed contrast (sensory evidence), previous choice (serial dependence), and WSLS (reward-modulated repetition)) using NeMoS basis objects and `compute_features`.
3. **Fitting across sessions**: fitting a 3-state GLM-HMM to trials across multiple sessions required just a few lines of code:

```python
model = nmo.glm_hmm.GLMHMM(n_states=3, regularizer="Ridge", seed=seed)
model.fit(X, choices, session_starts=session_starts)
```

4. **Interpretable parameters and linking states to behavior**: the GLM weights showed three distinct strategies - an engaged state driven by stimulus contrast, and two bias states favoring left or right regardless of evidence. The transition matrix showed that each strategy was stable across multiple consecutive trials. Also, `smooth_proba` and `decode_state` allowed us to track when each strategy was used and quantify its effect on performance.


## Additional exercises


- Compute the accuracy by segmenting the trials according to the smoothing posterior, as was done in the original paper. Restrict the analysis to trials in which the model is highly confident about the state assignment — for example, keep only trials where the smoothing posterior assigns a state more than 90% probability. Compare the results with the procedure shown here.

- Try varying the regularization strength and see what happens to the learned GLM parameters. Do the results still hold as you decrease the regularization strength? What happens to the model predictions? Try cross-validating the regularization strength and see how the optimal regularizer compares with the one used here (the nemos default, which is 1).

- What happens when you add or remove states? Can you cross-validate the number of states? Which number seems optimal according to your cross-validation procedure? Can you get a better cross-validated likelihood with more states?

- What happens if you select other blocks (for example, biased trials)? Would strategies change much?


## Additional resources


- [Bishop (2006) Chapter 13 "Sequential Data"](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf): Specially section 13.2, "Hidden Markov Models", provides an overview of MLE for HMMs, the forward-backward algorithm and the viterbi algorithm.
- [Zoe Ashwood's SSM tutorial on GLM-HMMs](https://github.com/zashwood/ssm/blob/master/notebooks/2b%20Input%20Driven%20Observations%20(GLM-HMM).ipynb): this educational notebook explains GLM-HMMs and fitting with MLE and MAP.
- [GLM-HMMs blogpost by Camilla Ucheoma](https://anneurai.net/2024/01/26/a-glm-hmm-deep-dive/): this blogpost provides a summary of Ashwood et al. (2022) work and a brief explanation of GLM-HMMs


## References

<a id="ref1a"><a href="#cite1a">[1a]</a> <a id="ref1b"><a href="#cite1b">[1b]</a> [The International Brain Laboratory, Aguillon-Rodriguez, V., Angelaki, D., Bayer, H., Bonacchi, N., Carandini, M., Cazettes, F., Chapuis, G., Churchland, A. K., Dan, Y., Dewitt, E., Faulkner, M., Forrest, H., Haetzel, L., Häusser, M., Hofer, S. B., Hu, F., Khanal, A., Krasniak, C., … Zador, A. M. (2021). Standardized and reproducible measurement of decision-making in mice. eLife, 10, e63711.](https://doi.org/10.7554/eLife.63711)

<a id="ref2a"><a href="#cite2a">[2a]</a> <a id="ref2b"><a href="#cite2b">[2b]</a> [Burgess, C. P., Lak, A., Steinmetz, N. A., Zatka-Haas, P., Bai Reddy, C., Jacobs, E. A. K., Linden, J. F., Paton, J. J., Ranson, A., Schröder, S., Soares, S., Wells, M. J., Wool, L. E., Harris, K. D., & Carandini, M. (2017). High-Yield Methods for Accurate Two-Alternative Visual Psychophysics in Head-Fixed Mice. Cell Reports, 20(10), 2513–2524.](https://doi.org/10.1016/j.celrep.2017.08.047)

<a id="ref3a"><a href="#cite3a">[3a]</a> <a id="ref3b"><a href="#cite3b">[3b]</a> <a id="ref3c"><a href="#cite3c">[3c]</a> [Ashwood, Z. C., Roy, N. A., Stone, I. R., Laboratory, I. B., Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. Nature Neuroscience, 25(2), 201–212.](https://doi.org/10.1038/s41593-021-01007-z)


