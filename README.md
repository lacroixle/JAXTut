# JAX tutorial
Welcome to the JAX tutorial taught at the Rodolphe Clédassou summer school (2026), the 19th and 20th August 2026.

## What is JAX?

[JAX](https://docs.jax.dev/en/latest/index.html) is a general purpose scientific computation Python library, built around a NumPy-like API and a set of composable function transformations:
- JIT-compilable: `jit` traces a function and compiles it to optimized kernels.
- Automatic differentiation: grad differentiates native Python code to arbitrary order.
- Automatic vectorization: `vmap` maps a function over an array axis, producing a batched version without an explicit loop.
- Hardware-accelerated: the same code runs on CPU, GPU, or TPU.

This tutorial assumes confortable experience in Python and knowledge of NumPy.
Most interesting examples and exercices involve basic bachelor level maths/physics.

## What you will learn
This tutorial covers plain, pure JAX itself, not the deep learning frameworks or cosmology libraries built on top of it. 
Those, though, are easier to approach once you have a working grasp of JAX.

In particular, we consider:
- How to think in JAX (pure functionnal programming, transformations, control flow)
- Its CPU performance, compared against NumPy
- How JAX handles randomness
- Structuring data and state with pytrees
- The automatic differentiation engine

One library that we will somewhat explore fall outsides the pure-JAX scope of this tutorial:
 - [Equinox](https://github.com/patrick-kidger/equinox) ― best seen as an extension of JAX: a pytree-based module system, filtered transformations, and tools for manipulating pytrees (plus NN layers we won't use).

Once you have these foundations, you can start exploring the wider ecosystem of libraries built on JAX.
You'll find a curated set [here](notes/links.md), spanning both broadly useful libraries and more specialised, technical ones.

### What about GPUs?
The notebooks run perfectly fine on GPU (but comparison with NumPy will be obviously skewed).
One of JAX's real strengths is that the same code runs unchanged across CPU, GPU, and TPU and handles compilation and device placement for you.
Even fairly involved projects rarely need any device-specific code.

In particular, we won't cover [sharding](https://docs.jax.dev/en/latest/notebooks/shard_map.html) (the general mechanism for distributing arrays across multiple devices) nor [manual host-device memory transfers](https://docs.jax.dev/en/latest/notebooks/host-offloading.html).
Everything you learn here applies equally well to GPUs, and gives you a good basis before tackling those kinds of low-level optimizations.

## Notebook organization
The tutorial is organized around Jupyter notebooks.

Recommended order:
- [BasicJAX](notebook/BasicJAX.ipynb) ― Major differences between JAX & NumPy and just-in-time compilation transform
- [Random](notebook/Random.ipynb) ― Pseudo random number generation in JAX
- [Vmap](notebook/Vmap.ipynb) ― Vmap transform and its applications
- [ControlFlow_Loops](notebook/ControlFlow_Loops.ipynb) ― While and for loops in JAX
- [ControlFlow_Scan](notebook/ControlFlow_Scan.ipynb) ― Sequential scanning in JAX
- [Pytrees](notebook/Pytrees.ipynb) ― Pytrees and why they are important to build large scale software/computations
- [Autodiff_Basic](notebook/Autodiff_Basic.ipynb) ― Automatic differentiation engine in JAX

## Installation
Since this tutorial is simply a collection of notebooks, there is not much to install.
However, it is always good practice to have dedicated environnements.

To retrieve the content of this GIT repos (in your current directory):
```bash
git clone https://github.com/lacroixle/JAXTut
```

### Through venv/pip

``` bash
python3 -m venv path/to/environnement
source path/to/environnement/bin/activate
pip install jax matplotlib jupyterlab equinox
```

Then to activate:
```bash
source path/to/environnement/bin/activate
```

### Through conda
With `JAXTut` as environnement name:
``` bash
conda create -n JAXTut
conda activate JAXTut
conda install -c conda-forge jax matplotlib jupyterlab equinox
```

Then to activate:
```
conda activate JAXTut
```

### JAX with GPUs and TPUs
[Read this.](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)

If you're using [CC-IN2P3 Jupyter notebooks](https://notebook.cc.in2p3.fr/hub/spawn), there is not much to install (apart from Equinox), and simply need to select a GPU interactive session with a "scientific kernel".

Note that fast double-precision (FP64) is limited to NVIDIA's compute-focused GPUs (V100, A100, H100, ...).
On all their other cards (including your laptop GPU), FP64 runs at only 1/64 of the single-precision (FP32) rate.

---
Written by [Leander Lacroix](https://github.com/lacroixle).
