# demo_cryo

Self-supervised cryo-ET denoising using **Equivariant Imaging (EI)** with missing-wedge physics and cube-symmetry rotations. 

Two training modes:
- **patch** — crop-based, fast
- **full** — whole-tomogram tiled, slow

---



## Quick start

### Cluster setup (Jean Zay / SLURM)

```bash
# 1. Load the PyTorch module (provides torch, numpy, scipy, etc.)
module load pytorch-gpu/py3/2.7.0

# 2. Install deepinv from PR #1088 (not the PyPI release):
#   https://github.com/deepinv/deepinv/pull/1088
git clone https://github.com/deepinv/deepinv.git
cd deepinv && git fetch origin pull/1088/head:pr-1088 && git checkout pr-1088
pip install --user -e .
cd ..

# 3. Install this repo (also installs mrcfile, pydantic, and other missing deps)
pip install --user -e /path/to/demo_cryo

# 4. Submit a job (set execution_mode: submitit in the config)
python main.py --config configs/conf_equivariant_patch.yml
python main.py --config configs/conf_equivariant_full.yml
```




---

## Project structure

```
main.py                      # CLI launcher (local + SLURM via submitit)
pyproject.toml               # dependencies + build config (install with uv sync)
configs/
  conf_equivariant_patch.yml # patch training config
  conf_equivariant_full.yml  # full-volume training config
src/
  toolcryo/                  # installable package
    base_config.py           # RunEIBaseConfig (shared fields)
    run.py                   # RunEIFullConfig, RunEIPatchConfig, run_full, run_patch
    trainer.py               # BaseTrainer, EIFullTrainer, EIPatchTrainer
    physics.py               # MissingWedge (missing-wedge forward operator)
    transform.py             # Rotate3D (cube-symmetry group)
    losses/
      losses.py              # ObsLoss, EqLoss (icecream-based)
      losses_custom.py       # ObsLoss, EqLoss (direct torch.fft, no icecream dependency)
    dataset/
      dataset_full.py        # full-volume dataset + dataloaders
      dataset_patch.py       # patch dataset + dataloaders
    inference/
      infer_full.py          # standalone inference for full-volume checkpoints
      infer_patch.py         # standalone + post-training inference for patch checkpoints
    utils/
      utils.py               # GpuFSC, build_ei_model, MRC I/O, metrics helpers
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

Uses deepinv's [distributed tiling framework](https://github.com/deepinv/deepinv/pull/1088) (`deepinv.distributed.distribute`) to run the UNet/drunet on a whole tomogram by splitting it into overlapping 3D tiles, processing each tile on a GPU, and stitching results back — no spatial downsampling.

**Current dataset handling**: the raw EMPIAR-11830 volumes are `1024×1024×512` (D×H×W). To keep things simple the dataset centre-crops each volume to a cube of side `min(D, H, W) = 512`, giving `512³` tensors fed to the trainer. A decision is still needed on how to utilize the full volume, e.g., center cropping, random 512³ cropping, or alternative patch-sampling strategies.

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

**`equivariant`**
| Key | Description |
|---|---|
| `tilt_max` / `tilt_min` | Missing-wedge tilt range in degrees |
| `use_spherical_support` | Spherical rather than cylindrical wedge |
| `wedge_double_size` | Pad FFT to 2× before applying wedge (patch mode) |
| `eq_weight` | Weight of the equivariant loss term |
| `loss_type` | `"icecream"` (default) or `"custom"` (direct torch.fft, no icecream dependency) |
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
| `use_mixed_precision` | fp16 forward pass + scaled backward |
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





