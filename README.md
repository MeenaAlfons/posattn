# Multi-dimensional Positional Self-Attention With Differentiable Span

This repository contains the source code accompanying the MSc. Thesis:

[Multi-dimensional Positional Self-Attention With Differentiable Span](docs/MSc_Thesis_Multi_dimensional_Positional_Self_Attention_With_Differentiable_Span.pdf) [[Slides]](docs/MSc_Slides_Multi_dimensional_Positional_Self_Attention_With_Differentiable_Span.pdf)

[Meena Alfons](https://meenaalfons.com)

## Abstract

Attentive models are inherently permutation equivariant and need positional encoding to incorporate positional information, which is crucial to structured data like time series, images and other inputs with a higher number of positional dimensions. Attentive models are suitable to represent long-distance relations, although they come with a substantial computational burden that local self-attention is designed to solve. This work presents a flexible way to produce positional features that naturally extends to multiple dimensions. We use sinusoidal representation networks (SIREN) to implicitly represent positional encoding. Our approach uses a relative positional encoding that integrates with the attentive model in a way that keeps the model translation equivariant. SIREN-based positional encoding gives comparable results to models depending on fixed sinusoidal features. We also introduce a differentiable span, a way to limit the attention span according to a locality feature inferred from the data. Using local self-attention with a differentiable span increases the model's accuracy under specific conditions. It also has the potential to reduce the computation costs of attention when the implementation makes use of the learned span to limit the computation of attention scores.

## Repository Structure

- `posattn`: contains the implementation of the concepts presented in the paper. You can use these components to include PositionalSelfAttention in your architecture.
- `cmd`
  - `main.py`: A command line tool to run experiments.
  - `models`: contains an implementation of a transformer-based classifier using PositionalSelfAttention. This model is used in our experiments.
  - `datasets`: are wrappers for the datasets we use.

## Reproduce

### Usage

Create and activate python virtual environment and install required dependencies.

```sh
python3 -m venv venv            # Create virtual environment
source venv/bin/activate        # Activate virtual environment
pip install -q -r requirements.txt # Install requirements
```

The experiments can be run through the provided command line tool `cmd/main.py`.

```sh
python cmd/main.py
```

**Configurations can be provided by:**

- A `config.yml` file which provides default values for all parameters
- Command line parameters `-D<param_name>=<value>` which override the values in `config.yml`.

```sh
python cmd/main.py -Dlearning_rate=0.05
```

Any run is saved by default to Weights & Biases. You need to set the environment variables `WANDB_ENTITY` and `WANDB_PROJECT` to identify the entity and the project you want to save the run to. You can disable saving the run to W&B that by using the option `-no_wandb=T`

```sh
python cmd/main.py -no_wandb=T
```

You can use the option `-O` in python to skip running asserts and `__debug__` blocks

```sh
python -O cmd/main.py -Dlearning_rate=0.05
```

### Experiments and config files

Using Weights & Biases, you can create a sweep and set its configurations to one of the sweep config files in `experiments/sweeps`. Then you can run a sweep agent as follows:

```sh
python cmd/main.py -mode=sweep_agent --sweep_id=$SWEEP_ID
```

Each sweep contains multiple combinations of configurations. Each sweep agent runs one combination. Therefore, you will need to run multiple sweep agents to execute all the experiments in one sweep.

The directory `experiments/sweeps` contains all the sweeps needed to reproduce the results in the paper.

## Achknowledgements

_We thank David W. Romero (Vrije Universiteit Amsterdam) for providing guidance throughout the project and for providing valuable input, especially regarding the initialisation of SIRENs. We thank Michael Cochez (Vrije Universiteit Amsterdam) for facilitating the process of getting funds for the needed computation resources for the project. We thank both David W. Romero and Michael Cochez for thoroughly reviewing the manuscript and providing valuable comments. We thank VU Amsterdam for providing the funds to use the Lisa GPU Compute Cluster managed by SURF (the IT cooperation organisation of educational and research institutions in the Netherlands). We thank [Weights \& Biases](https://www.wandb.com) for providing the platform for experiment tracking._
