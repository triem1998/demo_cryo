# demo_cryo

Self-supervised cryo-ET denoising using **Equivariant Imaging (EI)** and cube-symmetry rotations.

Two training modes:
- **patch** — crop-based, fast
- **full** — whole-tomogram tiled, slow

Three methods, selected by `general.preset`:
- **`missingwedge_ei`** — EI with an FFT missing-wedge operator, applied to precomputed FBP volumes (icecream-style).
- **`unrolled`** — PGD reconstruction unrolled through the *real* tomography operator, trained on the measured tilt series.
- **`tomo_ei`** — EI with the real tomography operator: a plain denoiser on each half's FBP volume, with the equivariance term re-simulated through `fbp(A(·))`.

---



## Quick start

### Cluster setup (Jean Zay / SLURM)

```bash
# 1. Load the PyTorch module (provides torch, numpy, scipy, etc.)
module load pytorch-gpu/py3/2.7.0

# 2. Install deepinv from the `ddp-feature` fork branch (not the PyPI release):
#   https://github.com/bmalezieux/deepinv/tree/ddp-feature
# It adds the 2-D process topology (`inner_world_size`) that lets DDP data
# parallelism sit on top of deepinv's distributed tiling.
git clone https://github.com/bmalezieux/deepinv.git
cd deepinv && git checkout 9d3afce72f1cee2f68820e24a998a2ee60ec6ec5
pip install --user -e .
cd ..

# 3. Install this repo (also installs mrcfile, pydantic, and other missing deps)
pip install --user -e /path/to/demo_cryo

# 4. Submit a job (set execution_mode: submitit in the config)
python main.py --config configs/conf_equivariant_patch.yml   # patch, missingwedge_ei
python main.py --config configs/conf_full_missing_wedge.yml  # full, missingwedge_ei
python main.py --config configs/conf_full_unrolled.yml       # full, unrolled
python main.py --config configs/conf_full_eq_tomo.yml        # full, tomo_ei
```




---

## Project structure

```
main.py                      # CLI launcher (local + SLURM via submitit)
pyproject.toml               # dependencies + build config (install with uv sync)
configs/
  conf_equivariant_patch.yml # patch training config
  conf_full_*.yml            # full-volume training configs (one per preset)
  conf_*_inference*.yml      # standalone inference configs
src/
  toolcryo/                  # installable package
    base_config.py           # RunEIBaseConfig (shared fields)
    run.py                   # RunEIFullConfig, RunEIPatchConfig, run_full, run_patch
    trainer.py               # BaseTrainer, EIFullTrainer, EIPatchTrainer
    registry.py              # preset -> (physics, model, losses, forward, recon)
    models.py                # build_ei_model (denoiser), build_unrolled_model (PGD)
    forward.py               # per-preset forward passes (how x_net/y_net are computed)
    transform.py             # Rotate3D (cube-symmetry group)
    physics/
      missingwedge.py        # MissingWedge (FFT wedge operator)
      tomography.py          # TomographyEM (astra backend)
      tomography_torch.py    # TomographyEMTorch (pure-torch, CPU/ROCm capable)
      tomography_build.py    # backend choice, EVN/ODD pairing, angle sharding
      __init__.py            # one physics builder per preset
    losses/
      losses_equivariant_wedge.py  # ObsLoss, EqLoss (missingwedge_ei, icecream-based)
      losses_equivariant_tomo.py   # ObsLoss, EqLoss (unrolled / tomo_ei, true physics)
    dataset/
      dataset_full.py        # full-volume dataset + dataloaders
      dataset_patch.py       # patch dataset + dataloaders
    inference/
      infer_full.py          # standalone inference for full-volume checkpoints
      infer_patch.py         # standalone + post-training inference for patch checkpoints
    utils/
      utils.py               # GpuFSC, MRC I/O, metrics helpers
      plot.py                # slice figures, FSC plots, metrics plots
    icecream_orig/           # vendored IceCream UNet3D — do not modify
```

---

## Key methods

### Patch training

Similar to IceCream's patch-based training with a few differences:

- **Multi-GPU (DDP)**: wraps the model in `DistributedDataParallel` when `world_size > 1`. Each GPU gets its own shard of the dataset via a `DistributedSampler`, so effective batch size scales linearly with GPU count.
- **Memory-mapped volumes**: volumes are loaded with `mrcfile` in memory-map mode (`mode="r"`), so only the patches actually sampled are paged into RAM — allows training on datasets larger than available memory.
- **Mixed-volume batches**: each batch draws `n_crops_per_vol` patches from every volume in the dataset per epoch, so a single batch contains patches from multiple tomograms. This differs from IceCream which iterates one volume at a time.
- **Not yet implemented**: IceCream has an option to bias patch sampling toward regions with more information content rather than uniform random sampling. Our sampler is purely random. This can be added in the future if needed.

### Full-volume training

Uses deepinv's [distributed tiling framework](https://github.com/bmalezieux/deepinv/tree/ddp-feature) (`deepinv.distributed.distribute`) to run the UNet/drunet on a whole tomogram by splitting it into overlapping 3D tiles, processing each tile on a GPU, and stitching results back — no spatial downsampling.

**Current dataset handling**: the raw EMPIAR-11830 volumes are `1024×1024×512` (D×H×W) and are fed to the trainer at native resolution — no cropping by default (`crop_size: null`). `target_shape` trilinearly downsamples volumes *and* the matching tilt series for local smoke tests. The `unrolled`/`tomo_ei` presets read the measured tilt series instead of FBP volumes (`data_source: measurement`), with the FBP volumes used only as the PGD initialisation.

---

## Config reference

### Key parameters

**`general`**
| Key | Description |
|---|---|
| `input_dir` | Path to tomogram directory (expects `vol_*/` subdirs with `.mrc` + `.tlt`) |
| `output_root` | Root for run outputs (default `./runs`) |
| `run_name` | Sub-directory name prefix |
| `execution_mode` | `local` or `submitit` |
| `max_train_vols` | Cap on training volumes (`null` = all) |
| `max_val_vols` | Cap on validation volumes |
| `seed` | Global random seed |
| `preset` | `missingwedge_ei` \| `unrolled` \| `tomo_ei` — selects the (physics, model, losses) triple |
| `target_shape` | Trilinearly resample volumes/sinograms to `[Y, X, Z]` (`null` = native; local testing only) |

**`equivariant`**
| Key | Description |
|---|---|
| `tilt_max` / `tilt_min` | Missing-wedge tilt range in degrees |
| `use_spherical_support` | Spherical rather than cylindrical wedge |
| `wedge_double_size` | Pad FFT to 2× before applying wedge (patch mode) |
| `eq_weight` | Weight of the equivariant loss term |
| `pixel_size_angstrom` | Pixel size for FSC resolution reporting |

**`training`** (patch)
| Key | Description |
|---|---|
| `num_epochs` | Training epochs |
| `learning_rate` | Adam learning rate |
| `grad_clip` | Gradient norm clip (`null` = disabled) |
| `ckp_interval` | Save checkpoint every N epochs |
| `eval_interval` | Run validation every N epochs |
| `log_every_n_epochs` | Print loss summary and save figures every N epochs |
| `infer_stride` | Sliding-window stride for post-training inference |
| `mixed_precision` | `"off"` \| `"fp16"` \| `"bf16"` — one switch for training, validation and inference. `"off"` is pure fp32; `"fp16"` adds a GradScaler; `"bf16"` needs none and does not overflow at native resolution |
| `model_type` | `"unet"` or `"drunet"` |

**`patch`**
| Key | Description |
|---|---|
| `crop_size` | Cubic patch side length (default 72) |
| `n_crops_per_vol` | Virtual crops per volume per epoch |
| `batch_size` | Crops per batch |
| `normalize` | Per-volume z-score normalisation |

**`distributed`** (full-volume only)
| Key | Description |
|---|---|
| `patch_size` | UNet tile size `[D, H, W]` |
| `overlap` | Tile overlap `[D, H, W]` |
| `max_batch_size` | Max tiles on GPU simultaneously |
| `checkpoint_batches` | Gradient checkpointing in tiled forward (`"auto"` or int) |
| `num_operators` | Angle-sharded tomography physics: `null` = one full operator per rank, `"auto"` = one per rank, int = that many (capped at the tilt count) |
| `tomography_backend` | `auto` \| `astra` \| `torch` — `auto` picks astra where its CUDA kernels run, else the pure-torch operator (CPU / AMD-ROCm) |

**`unrolled`** (unrolled / tomo_ei presets)
| Key | Description |
|---|---|
| `n_iter` | PGD steps unrolled into the network |
| `init_stepsize` | Initial PGD stepsize (operators are unit-norm, so used as-is) |
| `train_algo_params` | Learn `stepsize` jointly with the denoiser |

**`slurm`**
| Key | Description |
|---|---|
| `nodes` | Number of SLURM nodes |
| `gpus_per_node` | GPUs per node |
| `ntasks_per_node` | Tasks per node (= GPUs) |
| `cpus_per_task` | CPU threads per task |
| `time` | Wall time limit (`"HH:MM:SS"`) |
| `account` / `constraint` / `qos` | SLURM allocation |
| `setup` | List of bash commands run before the job |

---





