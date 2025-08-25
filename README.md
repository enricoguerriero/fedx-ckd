# FedX with Cross‑Knowledge Distillation (CKD)

This repository contains a complete, runnable implementation of **FedX** with *cross‑knowledge distillation* (CKD) as described in the ECCV 2022 paper [“FedX: Unsupervised Federated Learning with Cross Knowledge Distillation”](https://arxiv.org/abs/2207.09158).  The goal of FedX is to learn strong, unbiased representations from decentralized and heterogeneous data without requiring clients to share raw samples.  Our implementation supports CIFAR‑10, Fashion‑MNIST and SVHN, provides deterministic non‑IID data partitioning, logs training metrics via [Weights & Biases](https://wandb.ai/) and includes evaluation scripts for linear probing and a centralized baseline.

## Highlights

- **Exact CKD loss** — The two‑sided distillation objective uses a contrastive NT‑Xent term and a relational Jensen–Shannon term as originally defined.  See `src/fedx/loss.py` for a faithful implementation.
- **Modular architecture** — The codebase separates data loading, partitioning, model definitions, federated client/server logic, evaluation scripts and utility functions.
- **Deterministic non‑IID splits** — Clients receive samples according to a Dirichlet distribution with a configurable `--beta` parameter.  Setting `--seed` yields repeatable partitions.
- **Backbone** — A CIFAR‑style ResNet‑18 with a 3×3 convolution and no max‑pool layer is used throughout.  A projection MLP and a prediction MLP are attached when training FedX; a linear classifier is used for the baseline and linear probe.
- **Reproducibility** — All random number generators (NumPy, PyTorch and Python) are seeded.  CuDNN is run in deterministic mode.
- **W&B logging** — Metric tracking is optional via `--wandb_project`.  When unspecified the code falls back to local logging.
- **Colab notebook** — A ready‑to‑run notebook demonstrates a small federated training run, linear probing and the centralized baseline.

## Quickstart

### Installation

Create a fresh Python environment and install the required packages:

```bash
git clone https://github.com/enricoguerriero/fedx-ckd
cd fedx-ckd
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Federated Training (FedX + CKD)

The main training loop is provided by `src/fedx/trainer.py`.  It builds the dataset, partitions it non‑IID across clients and performs cross‑knowledge distillation.

```bash
python -m src.fedx.trainer \
  --dataset cifar10 \
  --n_clients 10 \
  --epochs_per_round 1 \
  --n_rounds 5 \
  --beta 0.5 \
  --batch_size 128 \
  --seed 42 \
  --device auto \
  --wandb_project FedX_CKD \
  --outdir runs/cifar10_fedx_s42
```

### Linear Probe

After federated training finishes, evaluate the learned representation by freezing the backbone and training a linear classifier on the union of all client training data:

```bash
python -m src.eval.linear_probe \
  --dataset cifar10 \
  --checkpoint runs/cifar10_fedx_s42/global_round_005.pt \
  --epochs 100 \
  --batch_size 256 \
  --seed 42 \
  --device auto \
  --wandb_project FedX_CKD
```

### Centralized Baseline

To compare against a non‑federated model, train the same CIFAR‑style ResNet‑18 centrally:

```bash
python -m src.eval.baseline_central \
  --dataset cifar10 \
  --epochs 100 \
  --batch_size 128 \
  --seed 42 \
  --device auto \
  --wandb_project FedX_CKD \
  --outdir runs/cifar10_central_s42
```

## Repository Layout

```
fedx-ckd/
  README.md
  requirements.txt
  src/
    data/
      loaders.py
      partition.py
    models/
      resnet_cifar.py
      projector.py
    fedx/
      loss.py
      client.py
      server.py
      trainer.py
      utils.py
    eval/
      linear_probe.py
      baseline_central.py
  scripts/
    run_fedx.sh
    run_probe.sh
    run_baseline.sh
  notebooks/
    colab_fedx_demo.ipynb
```

### Design Choices

- **CKD loss implementation** — We followed the pseudocode from the official FedX repository.  The contrastive term is the NT‑Xent loss and operates both locally and between local and global representations.  The relational term computes a symmetric Jensen–Shannon divergence between similarity distributions of the student and teacher representations.  Our implementation preserves the two temperature hyper‑parameters used by the authors.
- **Data loaders** — Each dataset loader returns two random augmentations of the same image alongside its label and index.  This is essential for contrastive learning.  Fashion‑MNIST images are converted to three channels so that a single backbone suffices for all datasets.
- **Dirichlet partitioning** — We use a per‑class Dirichlet allocation with parameter `α=beta` to distribute samples across clients.  If any client receives no samples, the partitioning is resampled until every client has at least one instance.
- **Linear probing** — The linear probe script constructs a classifier on top of the frozen backbone and trains with stochastic gradient descent for exactly `--epochs` iterations.
- **Reproducibility** — Seeding is applied to Python’s `random` module, NumPy and PyTorch.  We set `torch.backends.cudnn.deterministic=True` and `torch.backends.cudnn.benchmark=False` to minimise nondeterminism.
- **Logging** — Weights & Biases logging is optional.  If `--wandb_project` is omitted, metrics are printed to the console instead.  When enabled, runs use the project name provided on the CLI and group by dataset and experiment.

If any element of this implementation differs from the original paper, the divergence is explained in this document.  Feel free to open issues or pull requests if you notice a discrepancy.
