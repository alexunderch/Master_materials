# `Vi-T` in `Flax`
A very basic implementtation of [`Vi-T` paper](https://arxiv.org/abs/2010.11929) using [`Flax`](https://flax.readthedocs.io/en/latest/) neural network framework. The main goal of this one is to learn the device-agnostic framework, not get the best results. All results are collected in `wandb.sweep` using a small custom logger wrapper.

## Architecture and some implementation details
Architecture of the model is only suitable for classification tasks

![](https://production-media.paperswithcode.com/methods/Screen_Shot_2021-01-26_at_9.43.31_PM_uI4jjMq.png)

* Used `Adam` optimizer with cosine schedule of rate learning and gradient clipping;
* Used MultiHead self-attention with `n = 8` heads and hidden dimension of `768`; 
* Implemented learnable and sinusoid positional embeddings but used the former;

## Helpful links
1. https://huggingface.co/flax-community/vit-gpt2/tree/main/vit_gpt2
2. https://github.com/google/flax/blob/main/examples/imagenet/train.py
3. Official implementation
4. Good [set](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/JAX/tutorial6/Transformers_and_MHAttention.html) of jax tutorials
