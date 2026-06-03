---
jupytext:
  formats: ipynb,py:percent,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: glm_hmm_notebook (3.12.10)
  language: python
  name: python3
---

```{code-cell} ipython3
:tags: [hide-input, render-all]

import jax
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)
np.random.seed(65)
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_theme(style="ticks", palette="colorblind", font_scale=1.5, rc=custom_params)#, context="notebook")

import workshop_utils
```

:::{admonition} Working without pynapple
:class: note render-all

Unlike the other notebooks in this workshop, here we work directly with `pandas` DataFrames and `numpy` arrays rather than pynapple objects. The IBL trial data is naturally tabular (one row per trial, with no continuous time axis), so we keep it as a DataFrame and show how NeMoS interfaces with plain NumPy. NeMoS also accepts pynapple `Tsd`/`TsdFrame` objects directly, and we point out as we go where that would change the workflow (for example, how session boundaries are handled).
:::

:::{admonition} Download
:class: important render-all

This notebook can be downloaded as **{nb-download}`04_glm_hmm-presenters.ipynb`**. See the button at the top right to download as markdown or pdf.
:::

:::{admonition} Jupyter Lab tip
:class: important render-all

Newer versions of Jupyter Lab have addressed an issue with skipping around the notebook while scrolling. To make sure this fix is enabled, in the Jupyter Lab GUI, navigate to `Settings > Settings Editor > Notebook` and scroll down to the `Windowing mode` setting and make sure it is set to `contentVisibility`. 

Also reminder to presenter: Go to `View > Appearance`, select `Simple Interface` and turn off everything else to hide as many bars as possible. And maybe activate `Presentation Mode`.

And turn on `View > Render side-by-side` (shortcut `Shift+R`).
:::

# Infer behavioral strategies during decision making with GLM-HMMs
This notebook has had all its explanatory text removed and has not been run.
 It is intended to be downloaded and run locally (or on the provided binder)
 while listening to the presenter's explanation. In order to see the fully
 rendered of this notebook, go [here](../../full/live_coding/04_glm_hmm.md)



In this notebook, we will learn how to model behavioral choices by fitting a GLM-HMM, replicating the main findings of Ashwood et al. (2022) <span id="cite1a"></span><a href="#ref1a">[1a]</a>.

In particular, we will analyze the IBL decision-making task (IBL et al., 2021) <span id="cite2a"></span><a href="#ref2a">[2a]</a>, a variation of the two-alternative forced-choice perceptual detection task (Burgess et al., 2021 <span id="cite3a"></span><a href="#ref3a">[3a]</a>).

During this task, a sinusoidal grating with varying contrast [0\%-100\%] appeared either at the right or left side of the screen. The mice indicated this side by turning a small wheel, which moved the stimulus toward the center of the screen (Burgess et al., 2021 <span id="cite3b"></span><a href="#ref3b">[3b]</a>). If the mice chose the side correctly, they would receive a water reward; if not, they would get a noise burst and a 1-second timeout. For the first 90 trials of each session, the stimulus appeared randomly on either side of the screen; after that, it appeared on one side with fixed probability 0.8, alternating randomly every 20–100 trials. 



## Dataset


Data for this notebook comes from the IBL decision-making task (IBL et al., 2021) <span id="cite2c"></span><a href="#ref2c">[2c]</a>, a variation of the two-alternative forced-choice perceptual detection task (Burgess et al., 2021 <span id="cite3c"></span><a href="#ref3c">[3c]</a>).





![Task illustration](../../_static/IBL_edited.png)

*Task illustration. Modified from IBL et al. (2021)* <span id="cite2b"></span><a href="#ref2b">[2b]</a>.





Each stimulus is presented either to the left or to the right, with a probability that varies over time.







![Probability Left Stimulus](../../_static/prob_left.svg)

*Probability of left stimulus presentation over trials.*




## Data Streaming


First, let's download the data using  <a href="https://docs.internationalbrainlab.org/notebooks_external/one_quickstart.html">Open Neurophysiology Environment (ONE)</a>


```{code-cell} ipython3
:tags: [render-all]

import os
import jax
import jax.numpy as jnp
import numpy as np
import pynapple as nap
import seaborn as sns
from one.api import ONE
import matplotlib.pyplot as plt
from nemos.glm_hmm.utils import compute_rate_per_state
import nemos as nmo

# Instantiate the ONE object
data_dir = os.environ.get("NEMOS_DATA_DIR")
print("IBL data dir:", data_dir)
ONE.setup(base_url='https://openalyx.internationalbrainlab.org', silent=True)
one = ONE(password = 'international', cache_dir=data_dir)


# Then we need to choose our subject and run load_aggregate
subject = "CSHL_008"
trials = one.load_aggregate('subjects', subject, '_ibl_subjectTrials.table')

# We can see the information we get by printing the columns
print(trials.columns)
```


We only need a subset of those columns, in particular we will work with:




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


Let's extract what we need,



```{code-cell} ipython3
:tags: [render-all]

trials = trials[
    [
        "choice", "contrastLeft", "contrastRight", 
        "feedbackType", "probabilityLeft", "session"
    ]
]

```


and inspect its contents.


```{code-cell} ipython3
:tags: [render-all]

print(f"choice \nvalues: {np.sort(trials.choice.unique())}, data type: {trials.choice.dtype}, shape:  \n")
print(f"contrast left \nvalues: {np.sort(trials.contrastLeft.unique())}, data type: {trials.contrastLeft.dtype} \n")

print(f"contrast right \nvalues: {np.sort(trials.contrastRight.unique())}, data type: {trials.contrastRight.dtype} \n")

print(f"reward \nvalues: {np.sort(trials.feedbackType.unique())}, data type: {trials.feedbackType.dtype} \n")

print(f"probability of stimulus on left \nvalues: {np.sort(trials.probabilityLeft.unique())}, data type: {trials.probabilityLeft.dtype} \n")

print(f"session \n(some) values: {trials.session.unique()[:5]}, data type: {trials.session.dtype}\n")
```


Finally, let's focus our analysis on one example session.


```{code-cell} ipython3
:tags: [render-all]

# Choose example session
sess_ex = '726b6915-e7de-4b55-a38e-ff4c461211d3'
# Subset session trials
trials_sess = trials[trials.session == sess_ex].reset_index()
```



To match Ashwood et al. (2022) <a href="#ref1b">[1b]</a>, we will focus our analysis on the first 90 trials, in which the stimulus appears on the left and right with equal probability (50-50).



```{code-cell} ipython3
:tags: [render-all]


# Plot
plt.plot(trials_sess["probabilityLeft"][:300])
plt.axvspan(0, 90, color="skyblue", alpha=0.3, label="first 90 trials")
plt.axvline(90, color="skyblue", linestyle="--")

plt.ylabel("P(stimulus on the left)")
plt.xlabel("Trial number")
plt.show()

#TODO: add dropdown with original figures.
```


Let's do some pandas wrangling to keep only the sessions that have:

- Filter to sessions that went through all testing blocks (in particular all 50-50, 20-80 and 80-20 blocks). 
- Get the initial 50-50 trial block.
- Fewer than 10 invalid trials in that block. 



```{code-cell} ipython3
:tags: [render-all]

# Invalid choice marker
viol_val = 0

# Selecting the sessions as in the Ashwood at al. paper
# Boolean mask selecting sessions with 50-50, 20-80 and 80-20 blocks
has_three_blocks = (
    trials.groupby("session")["probabilityLeft"]
          .agg(lambda s: set(s.unique()) == {0.2, 0.5, 0.8})
)

# Boolean mask selecing sessions with < 10 invalid trials in the 50-50 block
violations = (
    trials.query("probabilityLeft == 0.5")
          .groupby("session")["choice"]
          .agg(lambda s: s.eq(viol_val).sum())
)

valid_sessions = has_three_blocks.index[has_three_blocks & (violations < 10)]

print(f"# of sessions before restrictions {trials['session'].nunique()}")

# Keep only the 50/50 trials from valid sessions (row order preserved by the filter)
df_trials = trials.query("session in @valid_sessions and probabilityLeft == 0.5")

print(f"# of sessions after restrictions {df_trials['session'].nunique()}")
```

## Design matrix


Now, with the valid sessions, we can compute the design matrix. In our case, we are interested in building a design matrix with three predictors: signed contrast, previous choice and win stay lose shift.





![Design matrix](../../_static/design_matrix_table.svg)






- Extract the columns we need for the design: `choice`, `contrastLeft`, `contrastRight` and `feedbackType` (reward). 



```{code-cell} ipython3

# We can select all the necessary values for the design matrix: 
# choice, contrast of stimuli and reward
choices = df_trials['choice'].values
stim_left = df_trials['contrastLeft'].values
stim_right = df_trials['contrastRight'].values
rewarded = df_trials['feedbackType'].values
session = df_trials['session'].values
```



- Replace `NaN` contrast values with `0` using `np.nan_to_num`.
- Compute the signed contrast (difference between left and right)




```{code-cell} ipython3
# Replace nans with 0s
stim_left = 
stim_right = 
# Compute the signed contrast (left - right)
signed_contrast = 
print(signed_contrast)
```


```{code-cell} ipython3
# Replace nans with 0s
stim_left = np.nan_to_num(stim_left, nan=0)
stim_right = np.nan_to_num(stim_right, nan=0)

# Compute the signed contrast
signed_contrast = stim_left - stim_right

# print the design for the first valid session
select_session = df_trials["session"] == valid_sessions[0]
signed_contrast[select_session]
```



- Get the index of the valid trials with `np.flatnonzero`




```{code-cell} ipython3
valid_choices_idx =
```


```{code-cell} ipython3
valid_choices_idx = np.flatnonzero(choices != viol_val)
```


Let's use the `basis` module from  NeMoS to define the design matrix. 

What we need is:

- A `HistoryConv` basis to capture the choice history. We need a `window_size=1` to include the previous 1 choice as predictor.




```{code-cell} ipython3
# Prev history with history of 1
prev_choice_basis = nmo.basis.HistoryConv(1)
```


```{code-cell} ipython3
# Prev history with history of 1
prev_choice_basis = nmo.basis.HistoryConv(1)
```



- A `IdentityEval` basis to include the signed contrast as is. This may seem pointless, but will allow us to add the basis and form the full design matrix in one go.




```{code-cell} ipython3
# Identity basis for stimuli
stimuli_basis = nmo.basis.IdentityEval()
```


```{code-cell} ipython3
# Identity basis for stimuli
stimuli_basis = nmo.basis.IdentityEval()
```




- Win-stay lose-shift is the product of previous choice and previous reward, $WSLS_t = c_{t-1} \cdot r_{t-1}$.
- We can use [basis multiplication](https://nemos.readthedocs.io/en/latest/background/basis/plot_02_ND_basis_function.html#n-dimensional-basis) to construct that predictor. 




```{code-cell} ipython3
# Create lagged reward basis
prev_reward_basis = 
# Multiply lagged reward basis with the lagged choice basis
wsls_basis = 
```


```{code-cell} ipython3
# Create lagged reward basis
prev_reward_basis = nmo.basis.HistoryConv(1)

# Multiply lagged reward basis with the lagged choice basis
wsls_basis = prev_choice_basis*prev_reward_basis
```




- Use [basis addition](https://nemos.readthedocs.io/en/latest/background/basis/plot_02_ND_basis_function.html#additive-basis-object) to define a basis which concatenates predictor.
- Create the design matrix by calling `compute_features`. Select the valid trials by applying the `valid_choices_idx` boolean mask.




```{code-cell} ipython3
# Create an additive basis using our three components
basis_object =
# Call compute features to get the raw model design
X_unnormalized = 
X_unnormalized[:5,:]
```


```{code-cell} ipython3
# Create an additive basis using our three components
basis_object = (
    stimuli_basis +      # will process one input
    wsls_basis +         # will process two inputs (choice & reward)
    prev_choice_basis    # will process one input
)

# Compute features
X_unnormalized = basis_object.compute_features(
    # input 1 : processed with stimuli_basis
    signed_contrast[valid_choices_idx],
    # input 2 : wsls input 1: choice
    choices[valid_choices_idx],
    # input 3 : wsls input 2: reward
    rewarded[valid_choices_idx],
    # input 4 : processed with prev_choice
    choices[valid_choices_idx],
)

X_unnormalized[:5,:]
```


- Z-score the contrast values.



```{code-cell} ipython3
from scipy.stats import zscore
# Copy the array (we'll need the un-normalized later)
X = np.copy(X_unnormalized)
# Apply z-scoring
X[:, 0] = 
```


```{code-cell} ipython3
# Copy the array (we'll need the un-normalized later)
X = np.copy(X_unnormalized)
```

```{code-cell} ipython3
from scipy.stats import zscore
# Apply z-scoring
X[:, 0] = zscore(X[:, 0])
```

:::{admonition} Why do we normalize our stimuli predictor?
:class: question render-all
:class: dropdown

When fitting a GLM-HMM, we are fitting a separate weight for each feature. However, if the features are on different numerical scales for reasons that are not related to the actual influence of each predictor, that renders the weights incomparable. Here we have three predictors:  
- (1) Previous choice and (2) WSLS are always exactly −1 or +1. Their values are discrete and bounded, and they already share the same scale.
- (3) Stimuli contrast is continuous. While it can reach −1 or +1 (full contrast), this value rarely occurs. 

Because the contrast values are typically much smaller in magnitude than ±1, the model compensates by assigning them a larger weight to match the output scale — purely because the values are numerically smaller. In practice, this is an artifact of scale that does not reflect the true influence of the predictor.

By normalizing, we rescale the predictor to have mean 0 and standard deviation 1. Previous choice and WSLS are already on a unit scale by construction — their values are symmetric around zero and their spread is naturally 1. This is why we only normalize signed contrast.
:::

```{code-cell} ipython3
:tags: [render-all]
 
# Plot an heatmap showing the model design
workshop_utils.plot_design_matrix(X, choices, valid_choices_idx);
```

## Model fitting


- For a Bernoulli GLM-HMM, observations must take values of 0 or 1.
- Convert choices to 0s and 1s. 1: Left and 0: Right. You can use `np.where`.



```{code-cell} ipython3
choices = 
```


```{code-cell} ipython3
choices = np.where(choices == -1, 0, choices)
```


- Create a vector containing the indices of each session start.



```{code-cell} ipython3
# Mark where session changes
new_sess_mouse = 
```


```{code-cell} ipython3
# Mark where session changes
new_sess_mouse = np.flatnonzero(session[1:] != session[:-1]) + 1
```

:::{admonition} How does this one-liner find the session starts?
:class: note dropdown render-all

`session` holds one session id per trial. Comparing `session[1:]` (every trial but the first) with `session[:-1]` (every trial but the last) yields a boolean array that is `True` wherever a trial's session id differs from the previous trial's — that is, exactly at the session boundaries. `np.flatnonzero` returns the indices where this is `True`, and we add `1` because the comparison is shifted by one (position `i` in the comparison corresponds to trial `i+1`). The result is the array of indices at which a new session begins.
:::


- Initialize the `GLMHMM` object with 3 states and `regularizer="Ridge"`.
- Set seed for trying different initial parameters (`jax.random.PRNGKey(number)`).
- By default, the intercept is set to match the empirical choice probability, and the coefficients are set as random gaussian centered at zero with small standard deviation.



```{code-cell} ipython3
n_states = 3
model = nmo.glm_hmm.GLMHMM(
model
```


```{code-cell} ipython3
n_states = 3

model = nmo.glm_hmm.GLMHMM(
    n_states,
    regularizer="Ridge",
    # change this to try multiple init
    seed=jax.random.PRNGKey(12), 
)

model
```



- Fit the model providing the `new_sess_mouse` markers as the `session_starts` argument of `model.fit`.



```{code-cell} ipython3
model.fit(
    X, 
    choices,
    session_starts=new_sess_mouse
)
```

## Results interpretation

### How to visualize the fitted parameters



Latent state labels are arbitrary. Below we permute those labels to match that of the reference paper.



```{code-cell} ipython3
:tags: [hide-input, render-all]

permutation = jnp.array([1, 2, 0])
model.coef_ = model.coef_[:, permutation]
model.intercept_ = model.intercept_[permutation]
model.transition_prob_ = model.transition_prob_[permutation][:, permutation]
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

workshop_utils.plot_glm_weights(model);
```



- State 1 have larger weight on the stimulus and low on the other predictors.
- The bias weight is larger in absolute value for state 2 and 3, but of opposite sign. (>0 : left; <0 : right)



### Interpreting the transition matrix



- Let's visualize the transition matrix. The utility function below plots the heatmap.



```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_transition_matrix(model);
```

### Using ```smooth_proba``` to visualize and interpret posterior state probabilities



- Call `smooth_proba` to compute the smoothing posterior probabilities of the latent states.
- Remember to provide the `session_starts=new_sess_mouse` parameter to mark the beginning of each session.
- Filter non-nan entries and check that the posterior sums to 1 over the states.




```{code-cell} ipython3
# Compute smooth_proba
posteriors = 
print(f"First five posteriors \n{posteriors[:5]} \n")
# Each (valid) posterior row sums to 1
valid = ~np.isnan(posteriors).any(axis=1)
print(
    "Does the posterior sum to one?", 
    np.allclose( , 1)
)
```


```{code-cell} ipython3
# Compute smooth_proba
posteriors = model.smooth_proba(
    X, 
    choices,
    session_starts=new_sess_mouse
)
print(f"First five posteriors \n{posteriors[:5]} \n")

# Each (valid) posterior row sums to 1
valid = ~np.isnan(posteriors).any(axis=1)
print(
    "Does the posterior sum to one?", 
    np.allclose(posteriors[valid].sum(axis=1), 1)
)
```


Let's plot the first 90 trials, corresponding to the first session.




- Use `color=["#ff7f00", "#4daf4a", "#377eb8"]` for a matching color code.



```{code-cell} ipython3
colors = ["#ff7f00", "#4daf4a", "#377eb8"]
for i, c in enumerate(colors):
    plt.plot(posteriors[:90, i], color=c)
```


Let's now use the utility function to plot the three sessions shown in Fig. 3a of <span id="cite1e"></span><a href="#ref1e">[1e]</a>.


```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_posteriors(posteriors, session);
```

### Understanding mouse behavior in different states



- Get the most likely sequence of states given the observation by calling `decode_state`, which runs the Viterbi (also known as max-sum) algorithm.
- Remember to provide the `session_starts=new_sess_mouse`




```{code-cell} ipython3
# get output of viterbi in one-hot encoding
decoded_states = 
decoded_states
```


```{code-cell} ipython3
# get output of viterbi in one-hot encoding
decoded_states = model.decode_state(
    X,
    choices,
    session_starts=new_sess_mouse,
    state_format = "one-hot"
)
decoded_states
```



- Compute the fractional occupancy for each state.
- Remember that the decoded states may contain NaNs.



```{code-cell} ipython3

# calculate occupancy
valid = np.all(~np.isnan(decoded_states), axis=1)
frac_occupancy_viterbi= np.nansum(decoded_states, axis=0) / valid.sum()
print(f"Fraction of occupancy {frac_occupancy_viterbi} \n")
```



- Compute the accuracy:
  - Mask out the 0 contrast stimuli (because there is no correct answer in that case)
  - `choice==0`: right choice, `choice==1`: left choice.
  - `signed_contrast < 0`: left stimulus presented, `signed_contrast > 0`: right stimulus presented.
  - Use convention above to get the accuracy.




```{code-cell} ipython3
# mask out the 0 contrast stimuli
mask =
# compute stimulus and chocie side applying the mask
stim_side =
# compute the accuracy
total_accuracy = 
# store in an array of dim 4
accuracies_to_plot_viterbi = np.zeros(4)
accuracies_to_plot_viterbi[0] = total_accuracy
```


```{code-cell} ipython3
# mask out the 0 contrast stimuli
mask = signed_contrast != 0

# compute stimulus and chocie side
stim_side = signed_contrast > 0

# get the correct choices boolean
correct_choices = choices == stim_side

# compute the total accuracy applying the mask
total_accuracy = np.mean(correct_choices[mask])

# store in an array of dim 4
accuracies_to_plot_viterbi = np.zeros(4)
accuracies_to_plot_viterbi[0] = total_accuracy
```



- Loop over the states and apply the same calculation to get the accuracy per state.




```{code-cell} ipython3
accuracy_per_state = np.zeros(n_states)
for s in range(n_states):
  in_state =
  accuracy_per_state[s] =
accuracies_to_plot_viterbi[1:] = accuracy_per_state
```


```{code-cell} ipython3
accuracy_per_state = np.zeros(n_states)
for s in range(n_states):
  in_state = (decoded_states[:, s] == 1) & mask
  accuracy_per_state[s] = correct_choices[in_state].mean()

accuracies_to_plot_viterbi[1:] = accuracy_per_state
```



- Plot the output.



```{code-cell} ipython3
:tags: [render-all]

workshop_utils.plot_accuracy_and_occupancy(
    frac_occupancy_viterbi, 
    accuracies_to_plot_viterbi
);
```

## Additional Exercises



- Compute the accuracy by segmenting the trials according to the smoothing posterior, as was done in the original paper. Restrict the analysis to trials in which the model is highly confident about the state assignment — for example, keep only trials where the smoothing posterior assigns a state more than 90% probability. Compare the results with the procedure shown here.

- Try varying the regularization strength and see what happens to the learned GLM parameters. Do the results still hold as you decrease the regularization strength? What happens to the model predictions? Try cross-validating the regularization strength and see how the optimal regularizer compares with the one used here (the nemos default, which is 1).

- What happens when you add or remove states? Can you cross-validate the number of states? Which number seems optimal according to your cross-validation procedure? Can you get a better cross-validated likelihood with more states?



## Additional resources



- [Bishop (2006) Chapter 13 "Sequential Data"](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf): Specially section 13.2, "Hidden Markov Models", provides an overview of MLE for HMMs, the forward-backward algorithm and the viterbi algorithm.
- [Zoe Ashwood's SSM tutorial on GLM-HMMs](https://github.com/zashwood/ssm/blob/master/notebooks/2b%20Input%20Driven%20Observations%20(GLM-HMM).ipynb): this educational notebook explains GLM-HMMs and fitting with MLE and MAP.




## References



<a id="ref1a"><a href="#cite1a">[1a]</a> <a id="ref1b"><a href="#cite1b">[1b]</a> <a id="ref1c"><a href="#cite1c">[1c]</a> <a id="ref1d"><a href="#cite1d">[1d]</a> <a id="ref1e"><a href="#cite1e">[1e]</a> [Ashwood, Z. C., Roy, N. A., Stone, I. R., Laboratory, I. B., Urai, A. E., Churchland, A. K., Pouget, A., & Pillow, J. W. (2022). Mice alternate between discrete strategies during perceptual decision-making. Nature Neuroscience, 25(2), 201–212.](https://doi.org/10.1038/s41593-021-01007-z)

<a id="ref2a"><a href="#cite2a">[2a]</a> <a id="ref2b"><a href="#cite2b">[2b]</a> <a id="ref2c"><a href="#cite2c">[2c]</a> [The International Brain Laboratory, Aguillon-Rodriguez, V., Angelaki, D., Bayer, H., Bonacchi, N., Carandini, M., Cazettes, F., Chapuis, G., Churchland, A. K., Dan, Y., Dewitt, E., Faulkner, M., Forrest, H., Haetzel, L., Häusser, M., Hofer, S. B., Hu, F., Khanal, A., Krasniak, C., … Zador, A. M. (2021). Standardized and reproducible measurement of decision-making in mice. eLife, 10, e63711.](https://doi.org/10.7554/eLife.63711)

<a id="ref3a"><a href="#cite3a">[3a]</a> <a id="ref3b"><a href="#cite3b">[3b]</a> <a id="ref3c"><a href="#cite3c">[3c]</a> [Burgess, C. P., Lak, A., Steinmetz, N. A., Zatka-Haas, P., Bai Reddy, C., Jacobs, E. A. K., Linden, J. F., Paton, J. J., Ranson, A., Schröder, S., Soares, S., Wells, M. J., Wool, L. E., Harris, K. D., & Carandini, M. (2017). High-Yield Methods for Accurate Two-Alternative Visual Psychophysics in Head-Fixed Mice. Cell Reports, 20(10), 2513–2524.](https://doi.org/10.1016/j.celrep.2017.08.047)

<a id="ref4"><a href="#cite4">[4]</a> [Escola, S., Fontanini, A., Katz, D., & Paninski, L. (2011). Hidden Markov models for the stimulus-response relationships of multistate neural systems. Neural Computation, 23(5), 1071–1132.](https://doi.org/10.1162/NECO_a_00118)

