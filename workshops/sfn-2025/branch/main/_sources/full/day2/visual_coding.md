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

%matplotlib inline
%load_ext autoreload
%autoreload 2
import warnings

warnings.filterwarnings(
    "ignore",
    message="plotting functions contained within `_documentation_utils` are intended for nemos's documentation.",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message="Converting 'd' to numpy.array",
    category=UserWarning,
)
```

:::{admonition} Download
:class: important render-all

This notebook can be downloaded as **{nb-download}`visual_coding.ipynb`**. See the button at the top right to download as markdown or pdf.
:::

# Exploring the Visual Coding Dataset


<div class="render-all">

This notebook serves as a group project: in groups of 4 or 5, you will analyze data from the [Visual Coding - Neuropixels dataset](https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels), published by the Allen Institute. This dataset uses [extracellular electrophysiology probes](https://www.nature.com/articles/nature24636) to record spikes from multiple regions in the brain during passive visual stimulation.

To start, we will focus on the activity of neurons in the visual cortex (VISp) during passive exposure to full-field flashes of color either black (coded as "-1.0") or white (coded as "1.0") in a gray background. If you have time, you can apply the same procedure to other stimuli or brain areas.

In this notebook, the pre-filled section will first select visually responsive neurons in area VISp. Then, you will fit GLMs to the selected neurons using `nemos`.

As this is the last notebook, the instructions are a bit more hands-off: you will make more of the modeling decisions yourselves. As a group, you will use your neuroscience knowledge and the skills gained over this workshop to decide:
- How to avoid overfitting.
- What features to include in your GLMs.
- Which basis functions (and parameters) to use for each feature.
- How to regularize your features.
- How to evaluate your model.

At the end of this session, we will regroup to discuss the decisions people made and evaluate each others' models.

</div>

This notebook is adapted from Camila Maura's [Intro to GLM notebook](https://alleninstitute.github.io/openscope_databook/higher-order/GLM_pynapple_nemos.html) in the Allen Institute OpenScope Databook. See that notebook for a thorough introduction to using `pynapple` and `nemos` by fitting this dataset.

```{code-cell} ipython3
:tags: [render-all]

# Import everything
import jax
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots
import workshop_utils

import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

# configure plots some
plt.style.use(nmo.styles.plot_style)
```

## Downloading and preparing data

<div class="render-all">
In this section, we will download the data and extract the relevant parts for analysis and modeling. This section is largely presented to you as is, so that you can get to the substantive sections more quickly.

First we download and load the data into pynapple.
</div>

```{code-cell} ipython3
:tags: [render-all]

data = workshop_utils.fetch_data("visual_coding_data.zip")
flashes, units = [nap.load_file(d) for d in data]
```

:::{admonition} Full dataset?
:class: dropdown

In this notebook, we are using only a subset of the full dataset. We've done a little bit of preprocessing and selection to reduce the size of the total file, which includes neurons from many different brain areas and stimulus paradigms.

To load the whole dataset for this session, run the following:

**WARNING:** this dataset is 2.6GB, so the following may take a while to run.

```{code-block} python
# Dataset information
dandiset_id = "000021"
dandi_filepath = "sub-726298249/sub-726298249_ses-754829445.nwb"

# Download the data using NeMoS
io = nmo.fetch.download_dandi_data(dandiset_id, dandi_filepath)

# load data using pynapple
data = nap.NWBFile(io.read(), lazy_loading=True)
```

`data` is then a pynapple NWB object, as we've interacted with in other notebooks. The following additional preprocessing was done to the NWB file before saving to the files we load at the beginning of this notebook:

```{code-block} python
# grab the spiking data
units = data["units"]

# map from electrodes to brain area
channel_probes = {}
for elec in data.nwb.electrodes:
    channel_id = elec.index[0]
    location = elec["location"].values[0]
    channel_probes[channel_id] = location

# Add a new column to include location in our spikes TsGroup
units.brain_area = [channel_probes[int(ch_id)] for ch_id in units.peak_channel_id]
# and only grab those located in area VISp
units = units[units.brain_area == "VISp"]

# drop unnecessary metadata
units.restrict_info(["rate", "quality", "brain_area"])

flashes = data["flashes_presentations"]
flashes.restrict_info(["color"])
```

See [Allen Institute website](https://portal.brain-map.org/circuits-behavior/visual-coding-neuropixels) for more information on the dataset.

:::
<div class="render-all">

Now that we have our spiking data, let's restrict our dataset to the relevant part.

![Visual stimuli set](../../_static/visual_stimuli_set.png)

During the flashes presentation trials, mice were exposed to white or black full-field flashes in a gray background, each lasting 250 ms, and separated by a 2 second inter-trial interval. In total, they were exposed to 150 flashes (75 black, 75 white).

</div>

```{code-cell} ipython3
:tags: [render-all]

# create a separate object for black and white flashes
flashes_white = flashes[flashes["color"] == "1.0"]
flashes_black = flashes[flashes["color"] == "-1.0"]
```

<div class="render-all">

Let's visualize our stimuli:

</div>

```{code-cell} ipython3
:tags: [render-all, hide-input]

n_flashes = 5
n_seconds = 13
offset = .5

start = flashes["start"].min() - offset
end = start + n_seconds

fig, ax = plt.subplots(figsize = (17, 4))
for flash, c in zip([flashes_white, flashes_black], ["silver", "black"]):
    for fl in flash[:n_flashes]:
        ax.axvspan(fl.start[0], fl.end[0], color=c, alpha=.4, ec=c)      

plt.xlabel("Time (s)")
plt.ylabel("Absent = 0, Present = 1")
ax.set_title("Stimuli presentation")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.xlim(start-.1,end)
    
```

## Preliminary analyses and neuron selection

<div class="render-all">

In this section, we will select a subset of the neurons that are visually responsive, which we will fit our GLM to.

First, we'll construct a {class}`~pynapple.IntervalSet` called `extended_flashes` which contains the peristimulus time. Right now, our `flashes` `IntervalSet` defines the start and end time for the flashes. In order to make sure we can model the pre-stimulus baseline and any responses to the stimulus being turned off, we would like to expand these intervals to go from 500 msecs before the start of the stimuli to 500 msecs after the end.

This `IntervalSet` will be the same shape as `flashes` and have the same metadata columns.

</div>

```{code-cell} ipython3
:tags: [render-all]

dt = .50 # 500 ms
start = flashes.start - dt # Start 500 ms before stimulus presentation
end = flashes.end + dt # End 500 ms after stimulus presentation

extended_flashes = nap.IntervalSet(start, end, metadata=flashes.metadata) 
```

<div class="render-all">

Now, we'll create two separate `IntervalSet` objects, `extended_flashes_black` and `extended_flashes_white`, which contain this info for only the black and the white flashes, respectively.

</div>

```{code-cell} ipython3
:tags: [render-all]

extended_flashes_white = extended_flashes[extended_flashes["color"] == "1.0"]
extended_flashes_black = extended_flashes[extended_flashes["color"] == "-1.0"]
```

<div class="render-all">

Now, we'll select our neurons. There are four criteria we want to use:

1. Brain area: we are interested in analyzing VISp units for this tutorial
2. Quality: we will only select “good” quality units. If you're curious, you can (optionally) [read more](https://alleninstitute.github.io/openscope_databook/visualization/visualize_unit_metrics.html) how about the Allen Institute defines quality.
3. Firing rate: overall, we want units with a firing rate larger than 2Hz around the presentation of stimuli
4. Responsiveness: we want units that actually respond to changes in the visual stimuli, i.e., their firing rate changes as a result of the stimulus.

We'll create a new `TsGroup`, `selected_units`, which includes only those units that meet the first three criteria, then check that it passes the assertion block.

</div>

```{code-cell} ipython3
:tags: [render-all]

# Filter units according criteria 1 & 2
selected_units = units[(units["brain_area"]=="VISp") & (units["quality"]=="good")] 

# Restrict around stimuli presentation
selected_units = selected_units.restrict(extended_flashes) 

# Filter according to criterion 3
selected_units = selected_units[(selected_units["rate"]>2.0)]
```

<div class="render-all">

Now, in order to determine the responsiveness of the units, it's helpful to use the {func}`~pynapple.process.perievent.compute_perievent` function: this will align units' spiking timestamps with the onset of the stimulus repetitions and take an average over them.

Let's use that function to construct two separate perievent dictionaries, one aligned to the start of the white stimuli, one aligned to the start of the black, and they should run from 250 msec before to 500 msec after the event.

</div>

```{code-cell} ipython3
:tags: [render-all]

# Set window of perievent 500 ms before and after the start of the event
window_size = (-.250, .500) 

peri_white = nap.compute_perievent(timestamps=selected_units,
                                   tref=nap.Ts(flashes_white.start),
                                   minmax=window_size)
peri_black = nap.compute_perievent(timestamps=selected_units,
                                   tref=nap.Ts(flashes_black.start),
                                   minmax=window_size)
```

<div class="render-all">

Visualizing these perievents can help us determine what features we'll want to include in our GLM. The following helper function should help.

</div>

```{code-cell} ipython3
:tags: [render-all, hide-input]

def plot_raster_psth(peri, units, color_flashes, n_units=9, start_unit=0, bin_size=.005, smoothing=0.015):
    """
    Plot perievent time histograms (PSTHs) and raster plots for multiple units.

    Parameters:
    -----------
    peri : dict
        Dictionary mapping unit names to binned spike count peri-stimulus data (e.g., binned time series).
    units : dict
        Dictionary of neural units, e.g., spike trains or trial-aligned spike events.
    color_flashes : str
        A label indicating the flash color condition ('black' or 'white'), used for visual styling.
    n_units : int
        The number of units to visualize.
    start_unit : int
        The index of the unit to start with.
    bin_size : float
        Size of the bin used for spike count computation (in seconds).
    smoothing : float
        Standard deviation for Gaussian smoothing of the PSTH traces.
    """

    # Layout setup: 9 columns (units), 2 rows (split vertically into PSTH and raster plot)
    n_cols = n_units
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols*2, 4))

    colors = plt.cm.tab10.colors

    # Extract unit names for iteration
    units_list = list(units.keys())[start_unit:start_unit+n_units]

    for i, unit in enumerate(units_list):
        u = peri[unit]
        line_color = colors[i % len(colors)]
        ax = axes[0, i]

        # Plot PSTH (smoothed firing rate)
        ax.plot(
            (np.mean(u.count(bin_size), 1) / bin_size).smooth(std=smoothing),
            linewidth=2,
            color=line_color
        )
        ax.axvline(0.0)  # Stimulus onset line

        span_color = "black" if color_flashes == "black" else "silver"
        ax.axvspan(0, 0.250, color=span_color, alpha=0.3, ec="black")  # Stim duration
        ax.set_xlim(-0.25, 0.50)
        ax.set_title(f'{unit}')

        # Plot raster
        ax = axes[1, i]
        ax.plot(u.to_tsd(), "|", markersize=1, color=line_color, mew=2)
        ax.axvline(0.0)
        ax.axvspan(0, 0.250, color=span_color, alpha=0.3, ec="black")
        ax.set_ylim(0, 75)
        ax.set_xlim(-0.25, 0.50)

    # Y-axis and title annotations
    axes[0, 0].set_ylabel("Rate (Hz)")
    axes[1, 0].set_ylabel("Trial")
    if n_rows > 2:
        axes[2, 0].set_ylabel("Rate (Hz)")
        axes[3, 0].set_ylabel("Trial")
    fig.text(0.5, 0.00, 'Time from stim(s)', ha='center')
    fig.text(0.5, 1.00, f'PSTH & Spike Raster Plot - {color_flashes} flashes', ha='center')
    plt.tight_layout()
```

```{code-cell} ipython3
:tags: [render-all]

# called like this, the function will visualize the first 9 units. play with the n_units
# and start_unit arguments to see the other units.
plot_raster_psth(peri_white, selected_units, "white", n_units=9, start_unit=0)
plot_raster_psth(peri_black, selected_units, "black", n_units=9, start_unit=0)
```

<div class="render-all">

You could manually visualize each of our units and select those that appear, from their PSTH to be responsive.

However, it would be easier to scale (and more reproducible) if you came up with some measure of responsiveness. So how do we compute something that captures "this neuron responds to visual stimuli"? Here, we will define "responsiveness" as the normalized difference in average firing rate between during stimulus presentation and before the stimulus was presented. We'll define a function that does that in the following hidden cell.

If you have other ideas and the time to explore them, you can return to this section and try other definitions of responsiveness.

</div>

```{code-cell} ipython3
:tags: [render-all, hide-input]

def get_responsiveness(perievents, bin_size=0.005):
    """Calculate responsiveness for each neuron. This is computed as:

    post_presentation_avg  : 
        Average firing rate during presentation (250 ms) of stimulus across
        all instances of stimulus. 

    pre_presentation_avg :
        Average firing rate prior (250 ms) to the presentation of stimulus
        across all instances prior of stimulus. 

    responsiveness : 
        abs((post_presentation_avg - pre_presentation_avg) / (post_presentation_avg + pre_presentation_avg))

    Larger values indicate higher responsiveness to the stimuli.
        
    Parameters
    ----------
    perievents : TsGroup
        Contains perievent information of a subset of neurons
    bin_size : float
        Bin size for calculating spike counts

    Returns
    ----------   
    resp_array : np.array
        Array of responsiveness information.

    """
    resp_dict = {}
    resp_array = np.zeros(len(perievents.keys()), dtype=float)

    for index, peri in enumerate(perievents.values()):
        # Count the number of timestamps in each bin_size bin.
        peri_counts = peri.count(bin_size)

        # Compute average spikes for each millisecond in the
        # the 250 ms before stimulus presentation
        pre_presentation = np.mean(peri_counts, 1).restrict(nap.IntervalSet([-.25,0]))

        # Compute average spikes for each millisecond in the
        # the 250 ms after stimulus presentation
        post_presentation = np.mean(peri_counts, 1).restrict(nap.IntervalSet([0,.25]))

        pre_presentation_avg = np.mean(pre_presentation)
        post_presentation_avg = np.mean(post_presentation)
        responsiveness = abs((post_presentation_avg - pre_presentation_avg) / (post_presentation_avg + pre_presentation_avg))

        resp_array[index] = responsiveness

    return resp_array
```

```{code-cell} ipython3
:tags: [render-all]

# Compute the responsiveness measure
responsiveness_white = get_responsiveness(peri_white)
responsiveness_black = get_responsiveness(peri_black)

# Get threshold for top 15% most responsive
thresh_black = np.percentile(responsiveness_black, 85)
thresh_white = np.percentile(responsiveness_white, 85)

# Only keep units that are within the 15% most responsive for either black or white
selected_units = selected_units[(responsiveness_black > thresh_black) | (responsiveness_white > thresh_white)]
```

<div class="render-all">

Let's visualize the selected units PSTHs to make sure they all look reasonable:

</div>

```{code-cell} ipython3
:tags: [render-all]

print(f"Remaining units: {len(selected_units)}")
peri_white = {k: peri_white[k] for k in selected_units.index}
peri_black = {k: peri_black[k] for k in selected_units.index}

plot_raster_psth(peri_black, selected_units, "black", n_units=len(peri_black))
plot_raster_psth(peri_white, selected_units, "white", n_units=len(peri_white))
```

## Avoiding overfitting

<div class="render-all">

As we've seen throughout this workshop, it is important to avoid overfitting your model. We've covered two strategies for doing so: either separate your dataset into train and test subsets or set up a cross-validation scheme. Pick one of these approaches and use it when fitting your GLM model in the next section.

You might find it helpful to refer back to the [advanced nemos](sklearn-cv) notebook and / or to use the following pynapple functions: {func}`~pynapple.IntervalSet.set_diff`, {func}`~pynapple.IntervalSet.union`, {func}`~pynapple.TsGroup.restrict` (see [phase precession notebook](phase-precess-cv)).

:::{admonition} Hints
:class: hint

Throughout this section and the next we'll include hints. They'll either be links back to earlier notebooks from this workshop that show an example of how to do the step in question, or a hint admonition like this, which you can expand to get a hint.

:::

</div>

Again, multiple viable ways of solving this. Here's the simplest: grab every third flash and use that as the test set.

```{code-cell} ipython3
# We take one every three flashes (33% of all flashes of test)
flashes_test_white = extended_flashes_white[::3]
flashes_test_black = extended_flashes_black[::3]
# The remaining is separated for training
flashes_train_white = extended_flashes_white.set_diff(flashes_test_white)
flashes_train_black = extended_flashes_black.set_diff(flashes_test_black)
# Merge both stimuli types in a single interval set
flashes_test = flashes_test_white.union(flashes_test_black)
flashes_train = flashes_train_white.union(flashes_train_black)
```

(visual-glm)=
## Fit a GLM

<div class="render-all">

In this section, you will use nemos to build a GLM. There are a lot of scientific decisions to be made here, so we suggest starting simple and then adding complexity. Construct a design matrix with a single predictor, using a basis of your choice, then construct, fit, and score your model to a single neuron (remembering to either use your train/test or cross-validation to avoid overfitting). Then add regularization to your GLM. Then return to the beginning and add more predictors. Then fit all the neurons. Then evaluate what basis functions and parameters are best for your predictors. Then use the tricks we covered in [the advanced nemos notebook](sklearn-feature-selection) to evaluate whether which predictors are necessary for your model, which are the most important.

You don't have to exactly follow those steps, but make sure you can go from beginning to end before getting too complex.

Good luck and we look forward to seeing what you come up with!


</div>

There are many ways to do this. We'll show fitting a single neuron `GLM` to a variety of stimulus-derived predictors. You can also see [the original notebook](https://alleninstitute.github.io/openscope_databook/higher-order/GLM_pynapple_nemos.html#adding-coupling-as-a-new-predictor) for a `PopulationGLM` that adds coupling.

### Prepare data

<div class="render-all">

- Create spike count data. (Hint: review the [current injection notebook](current-inj-basic).)

</div>

```{code-cell} ipython3
bin_size = .005
units_counts = selected_units.count(bin_size, ep=extended_flashes)
```

### Construct design matrix

<div class="render-all">

- Decide on feature(s).
- Decide on basis. (Hint: review the [current injection](current-inj-basis) or [place cell](sklearn-basis) notebooks.)
- Construct design matrix. (Hint: review the [place cell](basis_eval_place_cells) notebook.)

:::{admonition} What features should I include?
:class: hint dropdown

If you're having trouble coming up with features to include, here are some possibilities:
- Stimulus. (Review the [current injection](current-inj-prep) notebook.)
- Stimulus onset. (Hint: you can use {func}`numpy.diff` to find when the stimulus transitions from off to on.)
- Stimulus offset. (Hint: you can use {func}`numpy.diff` to find when the stimulus transitions from on to off.)
- For multiple neurons: neuron-to-neuron coupling. (Refer back to the [the head direction notebook from the first day](head_direction_fit) to see an example of fitting coupling filters.)

For the stimuli predictors, you probably want to model white and black separately.

:::

</div>

Here we'll model three separate components of the response:

- Flash onset (beginning): We will convolve the early phase of the flash presentation with a basis function. This allows for fine temporal resolution immediately after stimulus onset, where rapid neural responses are often expected.
- Flash offset (end): We will convolve the later phase of the flash (around its end) with a different basis function. This emphasizes activity changes around stimulus termination.
- Full flash duration (smoothing): We will convolve the entire flash period with a third basis function, serving as a smoother to capture more sustained or slowly varying effects across the full stimulus window.

```{code-cell} ipython3
# Create a TsdFrame filled by zeros, for the size of units_counts
stim = nap.TsdFrame(
    t=units_counts.t,
    d=np.zeros((len(units_counts), 2)), 
    columns = ['white', 'black']
)
# Check whether there is a flash within a given bin of spikes
# If there is not, put a nan in that index
idx_white = flashes_white.in_interval(units_counts)
idx_black = flashes_black.in_interval(units_counts)

# Replace everything that is not nan with 1 in the corresponding column
stim.d[~np.isnan(idx_white), 0] = 1
stim.d[~np.isnan(idx_black), 1] = 1

white_onset = nap.Tsd(
    t=stim.t, 
    d=np.hstack((0,np.diff(stim["white"])==1)),
    time_support = units_counts.time_support
)

white_offset = nap.Tsd(
    t=stim.t, 
    d=np.hstack((0,np.diff(stim["white"])==-1)),
    time_support = units_counts.time_support
)

black_onset = nap.Tsd(
    t=stim.t, 
    d=np.hstack((0,np.diff(stim["black"])==1)),
    time_support = units_counts.time_support
)
black_offset = nap.Tsd(
    t=stim.t, 
    d=np.hstack((0,np.diff(stim["black"])==-1)),
    time_support = units_counts.time_support
)
```

Now set up the basis functions

```{code-cell} ipython3
# Duration of stimuli
stimulus_history_duration = 0.250

# Window length in bin size units
window_len = int(stimulus_history_duration / bin_size)

# Initialize basis objects
basis_white_on = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=5, 
    window_size=window_len, 
    label="white_on"
)
basis_white_off = nmo.basis.RaisedCosineLinearConv(
    n_basis_funcs=5, 
    window_size=window_len, 
    label="white_off",
    conv_kwargs={"predictor_causality":"acausal"}
)
basis_white_stim= nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=5, 
    window_size=window_len, 
    label="white_stim"
)
basis_black_on = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=5, 
    window_size=window_len, 
    label="black_on"
)
basis_black_off = nmo.basis.RaisedCosineLinearConv(
    n_basis_funcs=5, 
    window_size=window_len, 
    label="black_off",
    conv_kwargs={"predictor_causality":"acausal"}
)
basis_black_stim = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=5, 
    window_size=window_len, 
    label="black_stim"
)

# Define additive basis object to construct design matrix
additive_basis = (
    basis_white_on + 
    basis_white_off + 
    basis_white_stim + 
    basis_black_on + 
    basis_black_off + 
    basis_black_stim
)
```

Construct design matrix, separately for test and train.

```{code-cell} ipython3
# Convolve basis with inputs - test set
X_test = additive_basis.compute_features(
    white_onset.restrict(flashes_test),
    white_offset.restrict(flashes_test),
    stim["white"].restrict(flashes_test),
    black_onset.restrict(flashes_test),
    black_offset.restrict(flashes_test),
    stim["black"].restrict(flashes_test)
)

# Convolve basis with inputs - train set
X_train = additive_basis.compute_features(
    white_onset.restrict(flashes_train),
    white_offset.restrict(flashes_train),
    stim["white"].restrict(flashes_train),
    black_onset.restrict(flashes_train),
    black_offset.restrict(flashes_train),
    stim["black"].restrict(flashes_train)
)
```

### Construct and fit your model

<div class="render-all">

- Decide on regularization. (Hint: review Edoardo's presentation and the [place cell](sklearn-cv) notebook.)
- Initialize GLM. (Hint: review the [current injection](current-inj-glm) or [place cell](sklearn-cv) notebooks.)
- Call fit. (Hint: review the [current injection](current-inj-glm) or [place cell](sklearn-cv) notebooks.)
- Visualize result on PSTHs. (Note that you should use {func}`~pynapple.process.perievent.compute_perievent_continuous` and the model predictions here! Otherwise, this looks very similar to our PSTH calculation above.)

</div>

Here's an example of how this could look for a single neuron. To do multiple neurons, `model` should be a `PopulationGLM` and fit to `units_counts.restrict(flashes_train)` instead.

(The following regularizer strength comes from cross-validation.)

```{code-cell} ipython3
regularizer_strength = 7.745e-06
# Initialize model object of a single unit
model = nmo.glm.GLM(
    regularizer="Ridge",
    regularizer_strength=regularizer_strength,
    solver_name="LBFGS", 
)
# Choose an example unit
unit_id = 951768318

# Get counts for train and test for said unit
u_counts = units_counts.loc[unit_id]
```

```{code-cell} ipython3
model.fit(X_train, u_counts.restrict(flashes_train))
```

```{code-cell} ipython3
# Use predict to obtain the firing rates
pred_unit = model.predict(X_test)

# Convert units from spikes/bin to spikes/sec
pred_unit = pred_unit/ bin_size
```

```{code-cell} ipython3
# Re-center timestamps around white stimuli
# +50 because we subtracted .50 at beginning of stimulus presentation
peri_white_pred_unit = nap.compute_perievent_continuous(
    timeseries=pred_unit, 
    tref=nap.Ts(flashes_test_white.start+.50),
    minmax=window_size
)  
# Re-center timestamps for black stimuli
# +50 because we subtracted .50 at beginning of stimulus presentation
peri_black_pred_unit = nap.compute_perievent_continuous(
    timeseries=pred_unit, 
    tref=nap.Ts(flashes_test_black.start+.50), 
    minmax=window_size
)  
```

<div class="render-all">

Here's a helper function for plotting the PSTH of the data and predictions (for one or multiple neurons), which you may find helpful for visualizing your model performance.

</div>

```{code-cell} ipython3
:tags: [render-all, hide-input]

def plot_pop_psth(
        peri,
        color_flashes,
        bin_size=0.005,
        smoothing=0.015,
        **peri_others
        ):
    """Plot perievent time histograms (PSTHs) and raster plots for multiple units.

    Model predictions should be passed as additional keyword arguments. The key will be
    used as the label, and the value should be a 2-tuple of `(style, peri)`, where
    `style` is a matplotlib style (e.g., "blue" or "--") and `peri` is a PSTH
    dictionary, as returned by `compute_perievent_continuous`.
    
    Parameters:
    -----------
    peri : dict or TsGroup
        Dictionary mapping unit names to binned spike count peri-stimulus data (e.g., binned time series).
    color_flashes : str
        A label indicating the flash color condition ('black' or other), used for visual styling.
    bin_size : float
        Size of the bin used for spike count computation (in seconds).
    smoothing : float
        Standard deviation for Gaussian smoothing of the PSTH traces.
    peri_others : tuple
        Model PSTHs to plot. See above for description

    """
    if not isinstance(peri, dict):
        peri = {0: peri}
        
    n_cols = len(peri)
    fig, axes = plt.subplots(1, n_cols,
                             figsize=(2.5 * n_cols, 2.5))
    if n_cols == 1:
        axes = [axes]

    for i, (unit, u) in enumerate(peri.items()):
        try:
            ax = axes[i]
        except TypeError:
            # then there's only set of axes
            ax = axes
        # Plot PSTH (smoothed firing rate)
        ax.plot(
            (np.mean(u.count(bin_size), 1) / bin_size).smooth(std=smoothing),
            linewidth=2,
            color="black",
            label="Observed"
        )
        ax.axvline(0.0)  # Stimulus onset line
        span_color = "black" if color_flashes == "black" else "silver"
        ax.axvspan(0, 0.250, color=span_color, alpha=0.3, ec="black")  # Stim duration
        ax.set_xlim(-0.25, 0.50)
        ax.set_title(f'{unit}')
        for (key, (color, peri_pred)) in peri_others.items():
            try:
                p = peri_pred[:, :, i]
            except IndexError:
                p = peri_pred
            ax.plot(
            (np.mean(p, axis=1)),
            linewidth=1.5,
            color=color,
            label=key.capitalize()
            )

    # Y-axis and title annotations
    axes[0].set_ylabel("Rate (Hz)")
    fig.legend(*ax.get_legend_handles_labels())
    fig.text(0.5, 0.00, 'Time from stim(s)', ha='center')
    fig.text(0.5, 1.00, f'PSTH - {color_flashes} flashes', ha='center')
    plt.tight_layout()
```

<div class="render-all">

The following cell shows you how to call this visualization function. Its arguments are:
- The PSTH object computed from the data. This should only contain the responses to either the black or white flashes, but can contain the PSTHs from one or more-than-one neurons.
- A string, either `"white"` or `"black"`, which determines some of the styling.
- Any number of keyword arguments (e.g., `predictions=` shown below) whose values are a tuple of `(style, peri)`, where `style` is a valid matplotlib style (e.g., `"red"`) and `peri` is additional PSTHs to plot. Expected use is, as below, to plot the predictions in a different color on top of the actual data.

</div>

```{code-cell} ipython3
:tags: [render-all]

plot_pop_psth(peri_white[unit_id], "white", predictions=("red", peri_white_pred_unit))
plot_pop_psth(peri_black[unit_id], "black", predictions=("red", peri_black_pred_unit))
```

### Score your model

<div class="render-all">

- We trained on the train set, so now we score on the test set. (Or use cross-validation.)
- Get a score for your model that you can use to compare across the modeling choices outlined above. (Hint: refer back to the [current injection](current-inj-score) or [place cell](sklearn-cv) notebook.)

</div>

```{code-cell} ipython3
# Calculate the mean score for the Stimuli + Coupling model
# using pseudo-r2-McFadden
score = model.score(
    X_test,
    u_counts.restrict(flashes_test),
    score_type="pseudo-r2-McFadden"
)

print(score)
```

### Try to improve your model?

<div class="render-all">

- Go back to the beginning of [this section](visual-glm) and try to improve your model's performance (as reflected by increased score).
- Keep track of what you've tried and their respective scores. 
    - You can do this by hand, but constructing a [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), as we've seen in the [advanced nemos notebook](sklearn-cv), is useful:

</div>

```{code-cell} ipython3
:tags: [render-all]

# Example construction of dataframe.
# In this:
# - additive_basis is the single AdditiveBasis object we used to construct the entire design matrix
# - model is the GLM we fit to a single neuron
# - unit_id is the int identifying the neuron we're fitting
# - score is the float giving the model score, summarizing model performance (on the test set)
import pandas as pd
data = [
    {
        "model_id": 0,
        "regularizer": model.regularizer.__class__.__name__,
        "regularizer_strength": model.regularizer_strength,
        "solver": model.solver_name,
        "score": score,
        "n_predictors": len(additive_basis),
        "unit": unit_id,
        "predictor_i": i,
        "predictor": basis.label.strip(),
        "basis": basis.__class__.__name__,
        # any other info you think is important ...
    }
    for i, basis in enumerate(additive_basis)
]
df = pd.DataFrame(data)

df
```
