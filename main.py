#!/usr/bin/env python3
"""Submitit/local launcher for cryo-ET demos.

Usage
-----
# Local run (uses GPU if available):
    python main.py --config configs/conf_equivariant_full_local.yml

# SLURM via submitit:
    python main.py --config configs/conf_equivariant_full.yml

"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import submitit

from toolcryo.run import RunEIFullConfig, RunEIPatchConfig
from toolcryo.run import run_full as run_training_ei_full, run_patch as run_training_ei_patch
from toolcryo.inference.infer_full import RunEIFullInferenceConfig
from toolcryo.inference.infer_full import run_inference as run_inference_ei
from toolcryo.inference.infer_patch import RunEIPatchInferenceConfig
from toolcryo.inference.infer_patch import run_inference as run_inference_patch

ROOT = Path(__file__).resolve().parent


def _print_config(cfg, header: str = "RunConfig") -> None:
    lines = [f"[config] {header}"]
    for key, val in cfg.model_dump().items():
        lines.append(f"  {key}: {val}")
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

_METHODS: dict[str, tuple] = {
    "equivariant_full":   (RunEIFullConfig,          run_training_ei_full),
    "equivariant_patch":  (RunEIPatchConfig,          run_training_ei_patch),
    "ei_inference":       (RunEIFullInferenceConfig,  run_inference_ei),
    "ei_patch_inference": (RunEIPatchInferenceConfig, run_inference_patch),
}

# Required YAML sections per method
_REQUIRED_SECTIONS: dict[str, list[str]] = {
    "equivariant_full":   ["general", "training", "distributed", "slurm"],
    "equivariant_patch":  ["general", "training", "patch", "equivariant", "slurm"],
    "ei_inference":       ["general", "distributed", "slurm"],
    "ei_patch_inference": ["general", "slurm"],
}


# ---------------------------------------------------------------------------
# SLURM job callable (picklable)
# ---------------------------------------------------------------------------

class CryoTrainingJob:
    def __init__(self, method: str, cfg_dict: dict):
        self.method = method
        self.cfg_dict = cfg_dict

    def __call__(self):
        submitit.helpers.TorchDistributedEnvironment().export(
            set_cuda_visible_devices=False
        )
        env = submitit.JobEnvironment()

        if self.method == "equivariant_full":
            from toolcryo.run import RunEIFullConfig, run_full as run_fn
            cfg = RunEIFullConfig(**self.cfg_dict)
        elif self.method == "equivariant_patch":
            from toolcryo.run import RunEIPatchConfig, run_patch as run_fn
            cfg = RunEIPatchConfig(**self.cfg_dict)
        elif self.method == "ei_inference":
            from toolcryo.inference.infer_full import RunEIFullInferenceConfig, run_inference as run_fn
            cfg = RunEIFullInferenceConfig(**self.cfg_dict)
        elif self.method == "ei_patch_inference":
            from toolcryo.inference.infer_patch import RunEIPatchInferenceConfig, run_inference as run_fn
            cfg = RunEIPatchInferenceConfig(**self.cfg_dict)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        cfg.output_dir = str(Path(cfg.output_dir) / f"slurm-{env.job_id}")

        is_inference = self.method in ("ei_inference", "ei_patch_inference")
        if not is_inference:
            print(
                f"[submitit] job_id={env.job_id} rank={env.global_rank} "
                f"local_rank={env.local_rank} world_size={env.num_tasks}",
                flush=True,
            )
            if env.global_rank == 0:
                _print_config(cfg)

        return run_fn(cfg)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submitit/local launcher for equivariant cryo-ET demo."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs/conf_equivariant_full_local.yml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local mode regardless of general.execution_mode in config.",
    )
    return parser.parse_args()


def _require_section(conf: dict, name: str) -> dict:
    section = conf.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"Missing or invalid '{name}' section in config.")
    return section


def load_config(path: str | Path) -> dict:
    conf_path = Path(path)
    if not conf_path.exists():
        raise FileNotFoundError(f"Config file not found: {conf_path}")

    with conf_path.open("r", encoding="utf-8") as f:
        conf = yaml.safe_load(f) or {}

    if not isinstance(conf, dict):
        raise ValueError("Top-level YAML config must be a dictionary.")

    method = str(conf.get("method", "equivariant_full")).lower()
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Supported: {', '.join(_METHODS)}.")

    for section_name in _REQUIRED_SECTIONS[method]:
        _require_section(conf, section_name)

    return conf


def build_run_config(conf: dict):
    method = str(conf.get("method", "equivariant_full")).lower()
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Supported: {', '.join(_METHODS)}.")
    cfg_class, _ = _METHODS[method]
    return method, cfg_class.from_yaml(conf)


# ---------------------------------------------------------------------------
# SLURM submission
# ---------------------------------------------------------------------------

def submit_job(method: str, cfg, slurm: dict) -> None:
    submitit_folder = Path(cfg.output_dir) / "submitit_logs"
    submitit_folder.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(folder=str(submitit_folder), slurm_python="python")
    gpus_per_node = int(slurm.get("gpus_per_node", 1))
    additional_params = dict(slurm.get("additional_parameters", {}))
    for src_key, dst_key in [
        ("ntasks_per_node", "ntasks-per-node"),
        ("cpus_per_task",   "cpus-per-task"),
        ("account",         "account"),
        ("constraint",      "constraint"),
        ("qos",             "qos"),
    ]:
        if src_key in slurm:
            additional_params[dst_key] = slurm[src_key]

    executor.update_parameters(
        name=str(slurm.get("job_name", "demo-cryo-ei")),
        nodes=int(slurm.get("nodes", 1)),
        slurm_gres=str(slurm.get("gres", f"gpu:{gpus_per_node}")),
        slurm_time=str(slurm.get("time", "02:00:00")),
        slurm_stderr_to_stdout=bool(slurm.get("stderr_to_stdout", True)),
        slurm_additional_parameters=additional_params,
        slurm_setup=list(slurm.get("setup", [])),
    )

    job = executor.submit(CryoTrainingJob(method, cfg.model_dump()))
    print(f"Submitted job: {job.job_id}")
    print(f"Submitit logs: {submitit_folder.resolve()}")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    conf = load_config(args.config)
    general = _require_section(conf, "general")
    slurm   = _require_section(conf, "slurm")

    execution_mode = str(general.get("execution_mode", "local")).lower()
    if args.local:
        execution_mode = "local"

    method, cfg = build_run_config(conf)

    if execution_mode == "local":
        _print_config(cfg)
        _, run_fn = _METHODS[method]
        run_fn(cfg)
        return

    submit_job(method, cfg, slurm)


if __name__ == "__main__":
    main()
