# Getting Started

Deep learning is a subset of ML that uses neural nets with multiple layers.
It is not appropriate for many problems, but it is very powerful and worth learning.

## Supervised learning

Supervised learning is when you have a data set with correct answers to train on.

Supervised learning algorithsm. are really good at:

1. Things that humans can do in less than a second.
   **Classification**: _Like identifying whether a picture contains a cat or a dog._
   This is called "classification."
   **Regression** e.g., Predicting the price of a house given factors.
2. **Sequence prediction** Predicting the next item in a given sequence.
   _In France I learned to speak BLANK._ We would probably finish that with French.

So overall, the three types of supervised learning we will cover are

1. Classification
1. Regression
1. Sequence prediction

## Unsupervised learning

Two types of unsupervised learning we will cover:

1. **Generative models** Given examples of a thing, create more examples of that thing.
   e.g., generating fake images of people given real images of people.
1. **Reinforcement learning** Given an environment, complete an objective.
   AIs that learn to beat experts at games. e.g., Dota and the game Go.

## Processing

For small data sets, we can just use our CPU locally. We will do some GPU cloud computing later though with larger data sets.

## Environment setup

I created an environment with the default Continuum packages with this command, [lifted from StackOverflow](https://stackoverflow.com/a/38084286).

```bash
conda create -n summerofai anaconda
```

To start working in this directory:

```bash
conda activate summerofai
```

To install `pytorch`

```bash
conda install pytorch torchvision -c pytorch
```
