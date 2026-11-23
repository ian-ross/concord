# Concord toolkit

**TODO**: Improve this.

This package installs the main Concord Python libraries.

## Installation

Install the toolkit from its package index or from a wheel:

```sh
python -m pip install concord-toolkit
```

`concord-base` depends on PyTorch. The Concord development workspace currently
selects the CUDA 13.0 PyTorch wheels, but CUDA 13.0 is not a requirement of
the toolkit. A CPU build or a build for an earlier CUDA version can be used if
it supports the required PyTorch and torchvision versions.

A wheel records the PyTorch version requirements, but it does not record the
PyTorch package index. The repository's `tool.uv.index` and `tool.uv.sources`
settings therefore do not affect someone installing a published Concord wheel.

For a specific PyTorch build, install PyTorch first and then install the
toolkit. For example, a CPU-only installation can use:

```sh
python -m pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install concord-toolkit
```

For CUDA installations, choose the command that matches the machine's driver
and supported CUDA version from the [PyTorch installation
selector](https://pytorch.org/get-started/locally/). Installing PyTorch first
lets pip keep that build when it installs `concord-toolkit`, provided the
version satisfies `concord-base`'s dependency constraints.

uv can select a compatible PyTorch backend automatically:

```sh
uv pip install concord-toolkit --torch-backend=auto
```

For deployments that need the same PyTorch build every time, pin the PyTorch
and torchvision versions in the deployment requirements or lockfile. Do not
rely on the Concord wheel alone to select a CUDA build.

