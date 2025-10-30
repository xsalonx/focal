# FoCal Generative Surrogate Models

This repository accompanies the master thesis on fast simulation of the ALICE Forward Calorimeter (FoCal) using modern generative models. It contains research code for training neural surrogates that emulate the hadronic section of the FoCal calorimeter and drastically reduce the cost of detailed detector simulations. The project targets negative pion showers in the forward rapidity region of the Large Hadron Collider (LHC), where conventional GEANT4-based simulations become extremely expensive in time and computing resources.

Machine-learning surrogates built here (GANs, diffusion models, normalizing flows, variational approximations, and latent-enhanced variants) learn to reproduce calorimeter response images conditioned on primary particle parameters. Physics-inspired objectives—such as matching the expected hit position derived from pseudorapidity and azimuthal angle—help ensure that generated showers remain faithful to detector geometry.

## Repository layout

```
.
├── focal/                  # Python package with training and evaluation utilities
│   ├── models/             # Implementations of GAN, diffusion, flow, approximation, VQ and latent diffusion models
│   └── utils/              # Data loading, preprocessing, losses, metrics, and training helpers
├── utilss/                 # Dataset analysis notebooks, plotting helpers, and physics-aware utilities
└── README.md
```

Key modules include:

- `focal/models/diffusion/`: DDIM-based conditional diffusion models with optional physics losses and SLURM-ready scripts.
- `focal/models/gan/`: Conditional GAN trainers, multiple generator/discriminator variants, and positional loss integration.
- `focal/models/approx/`: Deterministic and variational approximators that regress shower characteristics before projecting them back to pixel space.
- `focal/models/normflow/` and `focal/models/vq-diffusion/`: Flow-based and vector-quantized diffusion experiments for alternative generative families.
- `focal/utils/`: Shared preprocessing, dataset loaders, evaluation metrics (including Wasserstein distances, LPIPS, and OT-based scores), logging, and checkpoint helpers.
- `utilss/`: Standalone utilities for inspecting datasets, plotting response distributions, computing physics losses, and I/O helpers for `.npz` FoCal data dumps.

## Installation

1. **Create an environment** (Python 3.10+ recommended for JAX, PyTorch, and Diffusers compatibility):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. **Install core dependencies.** The project relies on both PyTorch and JAX for different model families, as well as `diffusers`, `wandb`, `scikit-learn`, `optax`, `matplotlib`, `tqdm`, `lpips-j`, and `POT` (for optimal transport metrics). Install matching CUDA builds when running on GPUs:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   pip install jax jaxlib diffusers==0.27.* wandb scikit-learn optax matplotlib tqdm pillow lpips-j pot
   ```
   Adjust package versions to match your hardware and thesis experiments.

3. **Editable install (optional):**
   ```bash
   pip install -e .
   ```

The code expects CUDA for most training runs but will fall back to CPU if no GPU is available.

## FoCal datasets

Training scripts assume calorimeter data stored as NumPy archives within a dataset directory. By default `focal.utils.data.load` looks for files such as:

- `particles.npz`: `(N, 3)` array with primary particle energy, pseudorapidity η, and azimuth φ.
- `responses.npz`: `(N, 105, 105)` hadronic shower images (single channel by default).
- Optional `centroids.npz` and `characteristics.npz` for approximation pipelines, plus `acceptance-mask.npz` or cached samples for quick previews.

A typical directory therefore looks like:

```
data/
  single-shower/
    particles.npz
    responses.npz
    centroids.npz
    characteristics.npz
    sample_particles.npz
    sample_responses.npz
```

Use the helpers in `utilss/std/data_io.py` to save or inspect FoCal datasets and `utilss/dataset_analyzis.py` or `utilss/plot.py` to generate quick quality reports before training.

When `load_data_scope` is invoked, data are split into train/validation/test partitions, scaled with logarithmic or standard transformations, and converted into PyTorch dataloaders shared by all model families. Physics-aware preprocessing (e.g., log1p scalers with added stochastic noise) is available for flow models via `CaloINNPreprocessorNP` and related classes.

## Training workflows

All trainers are exposed as Python modules with CLI arguments. Common flags include `--data-dir`, `--trainings-dir`, `--train-batch-size`, `--responses-scaler`, and `--no-wandb` to disable Weights & Biases logging. Checkpoint files are stored under `<trainings_dir>/<run-name>/checkpoints/` and support automatic resume and evaluation-only runs via `--ftest`.

### Diffusion models

```bash
python -m focal.models.diffusion.diffusion \
  --name focal-ddim \
  --model-id 1 \
  --train-batch-size 64 \
  --data-dir data/single-shower \
  --trainings-dir outputs \
  --epochs 100 \
  --responses-scaler 3log+1
```

- Supports DDIM sampling with configurable inference steps and optional positional losses derived from FoCal geometry (`--positional-loss-type` and `--positional-loss-lambda`).
- Provides `--ftest` mode to reload a specific epoch and regenerate metrics without retraining.

### Conditional GANs

```bash
python -m focal.models.gan.gan \
  --name focal-gan \
  --g-model-id 2 \
  --d-model-id 1 \
  --train-batch-size 128 \
  --data-dir data/single-shower \
  --trainings-dir outputs \
  --epochs 500 \
  --responses-scaler 3log+1
```

- Multiple generator backbones (upsampling, transpose-convolution, residual) and a shared discriminator are selectable via model IDs.
- Optional positional losses align the shower center of mass with expected FoCal hit coordinates.

### Approximation and hybrid models

`focal/models/approx` hosts deterministic regressors, VAEs, and enhancer stacks that first predict compressed shower descriptors and then reconstruct calorimeter images via physics-based decoders. These are useful for rapid ablation studies or combining with diffusion enhancers located in `focal/models/approx/diffusion` and `focal/models/approx/enhancer`.

### Flow-based experiments

- `focal/models/normflow/normflow.py` implements normalizing flows trained on log-transformed responses and supports temperature-based sampling during evaluation.

### SLURM integration

Batch launchers (`sbatch.sh`) illustrate how to submit training jobs on CERN clusters (e.g., `athena`) with predefined wall times and dataset paths. Use them as templates when adapting the experiments to HPC infrastructure.

## Evaluation and metrics

`focal/utils/eval.py` implements the shared evaluation loop. It logs MSE on predicted diffusion noise, computes Wasserstein distances, sliced W2, LPIPS-based perceptual similarity, and energy ratio plots. Generated samples and statistics are cached as `.npz` files for reproducibility, and the metrics module streams results to Weights & Biases while also capturing SLURM environment variables.

To run evaluation without training, rerun any trainer with `--ftest <epoch>` or load checkpoints manually using helpers in `focal.utils.train` (`load_ckpt`, `save_ckpt`, `default_generate_fn`). Visualizations of generated showers, energy distributions, and positional statistics are logged through `Metrics.plot_responses` and `plot_energy_ratio_distributions`.

## Analysis notebooks and utilities

The `utilss/image_analyzis` directory provides notebooks for exploring calorimeter energy characteristics and reconstruction heuristics, while `utilss/dataset_analyzis.py` offers quick scripts to remove invalid entries (NaNs, negative values) before training. Use these tools to validate new GEANT4 samples or to produce diagnostic plots included in the thesis.

## Reproducing the thesis results

1. Prepare datasets that mirror the thesis scenarios: single-shower and many-shower regimes, cleaned to remove NaNs and clipped to realistic energy ranges.
2. Select the corresponding model family and hyperparameters documented above. For diffusion experiments, start with model ID 1 and `3log+1` response scaling as used in the thesis.
3. Enable Weights & Biases logging (default) to collect comparable metrics and plots. Metrics automatically capture SLURM job metadata when run on a cluster, simplifying audit trails.
4. For physics-aware evaluation, activate positional losses and inspect the cached evaluation files generated by `focal/utils/eval.py`.

## Acknowledgements

This codebase was developed in collaboration with the ALICE FoCal team. Please cite the associated master thesis when publishing results derived from this repository. Contributions and issues are welcome via pull requests.
