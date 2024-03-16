# Rank-One Model Editing (ROME)

This repository provides an implementation of Rank-One Model Editing (ROME) on auto-regressive transformers (GPU-only).
We currently support OpenAI's GPT-2 XL (1.5B) and EleutherAI's GPT-J (6B). 

<p align="center">
    <img src="https://rome.baulab.info/images/eiftower-crop.svg" alt="causal tracing GIF" width="425px" />
</p>


You can find the official paper [here](https://openreview.net/forum?id=-h6WAS6eE4).

You can access the datasets [here](https://rome.baulab.info/data/).


## Table of Contents
1. [Installation](#installation)
2. [Causal Tracing](#causal-tracing)
3. [Rank-One Model Editing (ROME)](#rank-one-model-editing-rome-1)
4. [Running the Full Evaluation Suite](#running-the-full-evaluation-suite)


## Installation

We have used`conda` for managing Python, CUDA, and PyTorch-related dependencies, and `pip` for everything else. To get started, simply install `conda` and run:
```bash
./scripts/setup_conda.sh
```

## Causal Tracing

[`notebooks/causal_trace.ipynb`](notebooks/causal_trace.ipynb) demonstrates Causal Tracing, which can be modified to apply tracing to the processing of any statement.

<p align="center">
    <img src="https://thevisible.net/u/davidbau/romeweb/small-fast-ct-animation.gif" alt="causal tracing GIF" width="550px" />
</p>

## Rank-One Model Editing (ROME)

<!-- We provide a simple interactive notebook demonstrating ROME. -->

<!-- ### Second-Moment Key Statistics

**warning this is probably wrong; fixing later.**

First, key statistics must be collected. The `rome` package contains a `layer_stats` module for computing and caching key statistics. See [rome/layer_stats.py](rome/layer_stats.py) for additional flags, but the basic logic can be executed with the following commands:

GPT-2 XL:
```bash
python -m rome.layer_stats --layer_num=17 --model_name=gpt2-xl
```

GPT-J:
```bash
python -m rome.layer_stats --layer_num=10 --model_name=EleutherAI/gpt-j-6B
```

### ROME Model Rewriting -->

[`notebooks/rome.ipynb`](notebooks/rome.ipynb) demonstrates ROME. The API is simple; one simply has to specify a *requested rewrite* of the following form:

```python
request = {
    "prompt": "{} plays the sport of",
    "subject": "LeBron James",
    "target_new": {
        "str": "football"
    }
}
```

Several similar examples are included in the notebook.



### Running the Full Evaluation Suite

[`experiments/evaluate.py`](experiments/evaluate.py) can be used to evaluate any method in [`baselines/`](baselines/).
To get started (e.g. using ROME on GPT-2 XL), run:
```bash
python3 -m experiments.evaluate \
    --alg_name=ROME \
    --model_name=gpt2-xl \
    --hparams_fname=gpt2-xl.json
```

Results from each run are stored at `results/<method_name>/run_<run_id>` in a specific format:
```bash
results/
|__ ROME/
    |__ run_<run_id>/
        |__ params.json
        |__ case_0.json
        |__ case_1.json
        |__ ...
        |__ case_10000.json
```

To summarize the results, you can use [`experiments/summarize.py`](experiments/summarize.py):
```bash
python3 -m experiments.summarize --dir_name=ROME --runs=run_<run_id>
```

Running `python3 -m experiments.evaluate -h` or `python3 -m experiments.summarize -h` provides details about command-line flags.

