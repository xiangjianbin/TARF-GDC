from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
import shutil
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from .data import SingleSampleDataset
from .contracts import VERSION, validate_config, verify_implementation_freeze
from .losses import compute_loss
from .metrics import aggregate_metrics, metrics_per_sample, objective_score, v032_objective_score
from .model import build_model, parameter_count
from .staged_refinement import initialize_stage1, verify_frozen
from .physics import Normalization, build_operator
from .utils import (
    environment_payload,
    read_json,
    resolve_path,
    seed_everything,
    sha256_file,
    state_dict_sha256,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
REVISION_VERSION = VERSION
CODE_VERSIONS = {REVISION_VERSION}


def selection_policy(config: dict[str, Any], checkpoint: str = "best") -> dict[str, Any]:
    if checkpoint == "last":
        return {"metric": "epoch", "mode": "last", "split": "validation_iid", "noise_realization": 0}
    if checkpoint == "best_geometry":
        metric, mode = "density_support_dice", "max"
    elif checkpoint == "best":
        training = config["training"]
        metric = str(training.get("checkpoint_metric", "objective_score"))
        mode = str(training.get("checkpoint_mode", "min" if metric == "density_sae" else "max"))
    else:
        raise ValueError(f"unsupported checkpoint {checkpoint!r}")
    if mode not in {"min", "max"}:
        raise ValueError("checkpoint_mode must be min or max")
    return {"metric": metric, "mode": mode, "aggregation": "mean_per_model",
            "split": "validation_iid", "noise_realization": 0}


def selection_value(metrics: dict[str, float], policy: dict[str, Any]) -> float:
    metric = policy["metric"]
    if metric == "objective_score":
        value = objective_score(metrics)
    elif metric == "v032_objective":
        value = v032_objective_score(metrics)
    else:
        value = float(metrics[metric])
    if not math.isfinite(value):
        raise FloatingPointError(f"non-finite checkpoint metric {metric}: {value}")
    return value


def improves(value: float, best: float, mode: str) -> bool:
    return value < best if mode == "min" else value > best


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def categorical_diagnostics(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    names = sorted({name for row in rows for name in row if name.endswith(".alpha_top_channel")})
    return {name: dict(sorted(Counter(str(int(row[name])) for row in rows if name in row).items()))
            for name in names}


def validate_resume_config(config: dict[str, Any], saved: dict[str, Any]) -> None:
    if config != saved:
        changed = sorted(key for key in set(config) | set(saved) if config.get(key) != saved.get(key))
        raise ValueError(f"resume config differs from the frozen run config: {changed}")


def _save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _resume_history(path: Path, checkpoint: dict[str, Any]) -> None:
    """Reconcile the append-only log with the last committed epoch checkpoint."""
    epoch = int(checkpoint["epoch"])
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break  # A process may have stopped during the final log write.
        if int(row["epoch"]) < epoch:
            rows.append(row)
    if "history_record" in checkpoint:
        rows.append(checkpoint["history_record"])
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                     for row in rows), encoding="utf-8")
        temporary.replace(path)


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _amp_dtype(name: str) -> torch.dtype:
    values = {
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if name not in values:
        raise ValueError(f"unsupported AMP dtype {name!r}")
    return values[name]


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for name, value in batch.items()
    }


def build_dataset(
    config: dict[str, Any],
    split: str,
    noise_realization: int = 0,
) -> SingleSampleDataset:
    data_config = config["data"]
    if split == "train":
        mode = str(data_config.get("train_observation_mode", "online"))
        limit = data_config.get("train_limit")
    else:
        mode = str(data_config.get("evaluation_observation_mode", "frozen"))
        limit = (data_config.get("validation_limit") if split.startswith("validation")
                 else data_config.get("evaluation_limits", {}).get(split))
    return SingleSampleDataset(
        root=resolve_path(data_config["directory"], PROJECT_ROOT),
        split=split,
        observation_mode=mode,
        noise_realization=noise_realization,
        cache_shards=int(data_config.get("cache_shards", 2)),
        limit=int(limit) if limit else None,
        cache_directory=data_config.get('cache_directory'),
    )


def build_loader(
    dataset: SingleSampleDataset,
    config: dict[str, Any],
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    training = config["training"]
    num_workers = int(training.get("num_workers", 2))
    sampler = None
    loader_generator = generator
    if config.get("code_version") == REVISION_VERSION:
        # Worker/iterator seed draws must not advance the sampler or model RNG.
        # Dataset noise is a pure function of sample_id and the shared epoch.
        loader_generator = torch.Generator().manual_seed(int(config["seed"]) + (179 if shuffle else 283))
        if shuffle:
            sampler_generator = generator or torch.Generator().manual_seed(int(config["seed"]))
            sampler = RandomSampler(dataset, generator=sampler_generator)
    return DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        generator=loader_generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        **({"prefetch_factor": int(training.get("prefetch_factor", 2))} if num_workers else {}),
        drop_last=False,
    )


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    operator: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    input_mode: str,
    physics_mode: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    metric_batches: list[dict[str, torch.Tensor]] = []
    rows: list[dict[str, Any]] = []
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32

    for batch in loader:
        batch = _move_batch(batch, device)
        prepared = operator.prepare_inputs(
            batch["raw6"], batch["receiver_mask"], input_mode
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            output = model(prepared, operator)
        values = metrics_per_sample(output, batch, operator, physics_mode)
        categories: dict[str, torch.Tensor] = {}
        for layer_name in ("sab1", "sab2", "sab3"):
            layer = getattr(model, layer_name, None)
            for name, tensor in getattr(layer, "last_diagnostics", {}).items():
                if not torch.is_tensor(tensor) or tensor.shape != (len(batch["sample_id"]),):
                    raise ValueError(f"{layer_name}.{name} diagnostics must have one value per sample")
                column = f"{layer_name}.{name}"
                if name == "alpha_top_channel":
                    categories[column] = tensor
                else:
                    values[column] = tensor.float()
        metric_batches.append(values)
        cpu_values = {name: tensor.detach().float().cpu().tolist() for name, tensor in values.items()}
        cpu_categories = {name: tensor.detach().cpu().tolist() for name, tensor in categories.items()}
        for index, sample_id in enumerate(batch["sample_id"]):
            row: dict[str, Any] = {
                "sample_id": str(sample_id),
                "mechanism": str(batch["mechanism"][index]),
                "noise_mode": str(batch["noise_mode"][index]),
                "subtype": str(batch["subtype"][index]),
                "support_stratum": str(batch["support_stratum"][index]),
            }
            for name, value in cpu_values.items():
                row[name] = float(value[index])
            for name, value in cpu_categories.items():
                row[name] = int(value[index])
            rows.append(row)
    return aggregate_metrics(metric_batches), rows


def _scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if step < warmup_steps:
            return max(1.0e-6, float(step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    normalization: Normalization,
    epoch: int,
    global_step: int,
    score: float,
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "code_version": str(config.get("code_version", "v024_single_sample_v1")),
        "config": config,
        "epoch": epoch,
        "global_step": global_step,
        "metrics": metrics,
        "model": model.state_dict(),
        "normalization": normalization.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "scheduler": scheduler.state_dict(),
        "score": score,
    }


def _snapshot_sources(output_dir: Path, config_path: Path) -> None:
    snapshot = output_dir / "source_snapshot"
    for source in sorted(CODE_ROOT.rglob("*.py")):
        destination = snapshot / "code" / source.relative_to(CODE_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(config_path, snapshot / config_path.name)


def run_training(
    config: dict[str, Any],
    config_path: Path,
    run_id: str | None = None,
    resume: bool = False,
) -> Path:
    """A per-run kernel lock covers checks, writes, training and failure records."""
    validate_config(config)
    if config.get('phase') == 'reuse':
        raise ValueError('historical controls are evaluation-only')
    if config['phase'] == 'formal' and not config.get('staged_refinement'):
        raise ValueError('Stage-1 controls are reused, not retrained in this experiment')
    resolved_run_id = run_id or str(config["experiment_id"])
    if resolved_run_id in {"", ".", ".."} or Path(resolved_run_id).name != resolved_run_id:
        raise ValueError("run_id must be a single directory name")
    if config["phase"] == "formal" and resolved_run_id != config["experiment_id"]:
        raise ValueError("formal run identifiers cannot be overridden")
    output_root = resolve_path(config["output_root"], PROJECT_ROOT)
    lock_dir = output_root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / (resolved_run_id + ".lock")).open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("run already has a live lock owner: " + resolved_run_id) from error
        output_dir = output_root / resolved_run_id
        if output_dir.exists() and not resume:
            raise FileExistsError("run already exists: " + str(output_dir))
        if config["phase"] == "formal":
            verify_implementation_freeze(PROJECT_ROOT, config_path, config)
        torch.set_num_threads(int(config["training"].get("cpu_threads", 2)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # An embedding process may already have initialized its pool.
        identity = {
            "pid": os.getpid(), "run_id": resolved_run_id,
            "proc_start_ticks": Path("/proc/self/stat").read_text().rsplit(") ", 1)[1].split()[19],
            "config_sha256": sha256_file(config_path), "started_at": time.time(),
        }
        write_json(lock_dir / (resolved_run_id + ".json"), identity)
        try:
            result = _run_training(config, config_path, run_id, resume)
        except Exception as error:
            write_json(output_dir / "RUN_FAILED.json", {
                **identity, "error": repr(error), "status": "FAIL", "failed_at": time.time(),
            })
            raise
        if (output_dir / "RUN_FAILED.json").exists():
            (output_dir / "RUN_FAILED.json").rename(
                output_dir / ("RECOVERED_FAILURE_" + str(time.time_ns()) + ".json"))
        return result


def _run_training(
    config: dict[str, Any],
    config_path: Path,
    run_id: str | None = None,
    resume: bool = False,
) -> Path:
    config = copy.deepcopy(config)
    if config.get("code_version") not in CODE_VERSIONS:
        raise ValueError(f"config code_version must be one of {sorted(CODE_VERSIONS)}")
    if not torch.cuda.is_available() and config.get("device", "cuda") == "cuda":
        raise RuntimeError("CUDA was requested but is unavailable")

    seed = int(config["seed"])
    seed_everything(seed)
    device = torch.device(config.get("device", "cuda"))
    amp_dtype = _amp_dtype(str(config["training"].get("amp_dtype", "bf16")))
    amp_enabled = device.type == "cuda" and amp_dtype != torch.float32
    input_mode = str(config["task"]["input_mode"])
    physics_mode = str(config["task"]["physics_mode"])
    variant = str(config["model"]["variant"])
    if (variant == "t0") != (input_mode == "tzz"):
        raise ValueError("only T0 uses Tzz-only inputs")

    output_root = resolve_path(config.get("output_root", "runs"), PROJECT_ROOT)
    resolved_run_id = run_id or str(config["experiment_id"])
    output_dir = output_root / resolved_run_id
    if output_dir.exists() and not resume:
        raise FileExistsError(f"run already exists: {output_dir}")
    if resume:
        validate_resume_config(config, read_json(output_dir / "config_snapshot.json"))
        snapshot = output_dir / "source_snapshot" / "code"
        for source in sorted(CODE_ROOT.rglob("*.py")):
            saved_source = snapshot / source.relative_to(CODE_ROOT)
            if not saved_source.is_file() or sha256_file(source) != sha256_file(saved_source):
                raise ValueError("resume source differs from snapshot: " + str(source))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    normalization_path = resolve_path(config["data"]["normalization_path"], PROJECT_ROOT)
    normalization = Normalization.from_json(normalization_path)
    operator = build_operator(config, normalization, PROJECT_ROOT).to(device)
    model = build_model(config, normalization, operator).to(device)
    stage_report = initialize_stage1(model, config) if config.get('staged_refinement') else None
    initial_full_hash = state_dict_sha256(model.state_dict())
    initial_common_hash = state_dict_sha256(model.common_state_dict())

    train_dataset = build_dataset(config, "train")
    validation_dataset = build_dataset(config, "validation_iid", noise_realization=0)
    train_generator = torch.Generator()
    train_loader = build_loader(train_dataset, config, True, train_generator)
    validation_loader = build_loader(validation_dataset, config, False)
    print(f'[{resolved_run_id}] START dataset={train_dataset.root} '
          f'train={len(train_dataset)} validation={len(validation_dataset)} '
          f'epochs={config["training"]["epochs"]} lr={config["training"]["learning_rate"]} '
          f'loss={config["loss"]} architecture={config["model"].get("architecture", "legacy-" + variant)}', flush=True)

    training = config["training"]
    epochs = int(training["epochs"])
    max_steps = int(training.get("max_steps", 0))
    expected_steps = epochs * len(train_loader)
    total_steps = max_steps if max_steps > 0 else expected_steps
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 1.0e-4)),
    )
    scheduler = _scheduler(
        optimizer, total_steps, float(training.get("warmup_fraction", 0.05))
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled and amp_dtype == torch.float16
    )

    start_epoch = 1
    global_step = 0
    policy = selection_policy(config)
    geometry_policy = selection_policy(config, "best_geometry")
    best_score = math.inf if policy["mode"] == "min" else -math.inf
    best_epoch = 0
    best_geometry_score = -math.inf
    best_geometry_epoch = 0
    epochs_completed = 0
    last_epoch = 0
    last_epoch_complete = False
    elapsed_before_resume = 0.0
    last_path = checkpoint_dir / "last.pt"
    if resume:
        if not last_path.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {last_path}")
        checkpoint = _torch_load(last_path)
        validate_resume_config(config, checkpoint["config"])
        if not checkpoint.get("epoch_complete", True):
            raise ValueError("resume requires a checkpoint at a complete epoch boundary")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_score = float(checkpoint.get("best_score", checkpoint.get("score", best_score)))
        best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
        best_geometry_score = float(checkpoint.get("best_geometry_score", -math.inf))
        best_geometry_epoch = int(checkpoint.get("best_geometry_epoch", 0))
        epochs_completed = int(checkpoint["epoch"])
        last_epoch = epochs_completed
        last_epoch_complete = True
        elapsed_before_resume = float(checkpoint.get("elapsed_seconds", 0.0))
        _resume_history(output_dir / "history.jsonl", checkpoint)
        if "rng_state" in checkpoint:
            restore_rng_state(checkpoint["rng_state"])
        elif config.get("code_version") == REVISION_VERSION:
            raise ValueError("V027 resume requires saved Python/NumPy/Torch RNG states")
        del checkpoint
    else:
        write_json(output_dir / "config_snapshot.json", config)
        write_json(output_dir / "environment.json", environment_payload(device))
        write_json(
            output_dir / "initialization_report.json",
            {
                "common_state_sha256": initial_common_hash,
                "full_state_sha256": initial_full_hash,
                "module_hashes": model.initialization_hashes()
                if hasattr(model, "initialization_hashes") else {},
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": parameter_count(model),
                "extra_capacity": model.extra_capacity_report(),
                "seed": seed,
            },
        )
        write_json(
            output_dir / "data_manifest.json",
            {
                "dataset": str(resolve_path(config["data"]["directory"], PROJECT_ROOT)),
                "dataset_manifest_sha256": sha256_file(
                    resolve_path(config["data"]["directory"], PROJECT_ROOT) / "manifest.json"
                ),
                "normalization": str(normalization_path),
                "normalization_sha256": sha256_file(normalization_path),
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "cache_directory": str(train_dataset.cache_directory),
            },
        )
        _snapshot_sources(output_dir, config_path)

    history_path = output_dir / "history.jsonl"
    started = time.perf_counter()
    if stage_report:
        if stage_report['freeze_backbone']:
            verify_frozen(model, stage_report['frozen_state_sha256'])
        if not resume:
            write_json(output_dir/'stage1_loading_report.json', stage_report)
            zero_metrics, zero_rows = evaluate_model(
                model, operator, validation_loader, device, amp_dtype, input_mode, physics_mode)
            best_score = selection_value(zero_metrics, policy)
            best_geometry_score = selection_value(zero_metrics, geometry_policy)
            if config['phase'] == 'formal':
                reference_metrics = read_json(Path(config['staged_refinement']['source_config']).parent/
                                              'training_summary.json')['final_validation_metrics']
                for key in ('density_sae','density_support_dice','physics_nrms'):
                    if not math.isclose(zero_metrics[key], reference_metrics[key], rel_tol=2e-5, abs_tol=1e-5):
                        raise AssertionError('Epoch-zero prediction does not reproduce stage 1: '+key)
            write_json(output_dir/'epoch_zero.json', dict(
                epoch=0, stage2_steps=0, validation=zero_metrics,
                selection_value=best_score, eligible_for_selection=True,
                note='Pretrained checkpoint before extra training; eligible for both continuation and refinement.'))
            zero_payload = _checkpoint_payload(
                model, optimizer, scheduler, scaler, config, normalization,
                0, 0, best_score, zero_metrics)
            zero_payload.update(best_score=best_score, best_epoch=0,
                best_geometry_score=best_geometry_score, best_geometry_epoch=0,
                selection_policy=policy, geometry_selection_policy=geometry_policy,
                epoch_complete=True, rng_state=capture_rng_state(device), elapsed_seconds=0.,
                checkpoint_kind='best', checkpoint_selection_policy=policy,
                checkpoint_selection_value=best_score)
            _save_checkpoint(zero_payload, checkpoint_dir/'best.pt')
            _save_checkpoint(zero_payload, last_path)
            write_csv(output_dir/'metrics_per_sample.csv', zero_rows)
            print(f'[{resolved_run_id}] stage1_loaded frozen={stage_report["freeze_backbone"]} epoch0_sae={best_score:.6f} '
                  f'trainable={parameter_count(model)}', flush=True)
        else:
            loaded_report = read_json(output_dir/'stage1_loading_report.json')
            if loaded_report != stage_report:
                raise ValueError('Stage-1 identity changed on resume')
    log_interval = int(training.get("log_interval", 100))
    gradient_clip = float(training.get("gradient_clip", 1.0))
    stopped_by_max_steps = False

    for epoch in range(start_epoch, epochs + 1):
        if max_steps > 0 and global_step >= max_steps:
            stopped_by_max_steps = True
            break
        model.train()
        train_dataset.set_epoch(epoch - 1)
        train_generator.manual_seed(seed + 991 + epoch)
        epoch_losses: list[dict[str, float]] = []
        epoch_started = time.perf_counter()

        for batch_index, batch in enumerate(train_loader, start=1):
            if max_steps > 0 and global_step >= max_steps:
                stopped_by_max_steps = True
                break
            batch = _move_batch(batch, device)
            prepared = operator.prepare_inputs(
                batch["raw6"], batch["receiver_mask"], input_mode
            )
            optimizer.zero_grad(set_to_none=True)
            record_context = bool(getattr(model, "uses_depth_context", False)) and (
                global_step < 3 or (log_interval > 0 and batch_index % log_interval == 0)
            )
            model.collect_diagnostics = record_context
            record_relation = bool(getattr(model, 'uses_relation', False)) and (
                global_step < 3 or (log_interval > 0 and batch_index % log_interval == 0))
            model.collect_relation_diagnostics = record_relation
            record_refinement = bool(getattr(model, 'uses_residual_refiner', False)) and (
                global_step < 3 or (log_interval > 0 and batch_index % log_interval == 0))
            if getattr(model, 'uses_residual_refiner', False):
                model.residual_refiner.collect_diagnostics = record_refinement
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                output = model(prepared, operator)
            with torch.autocast(device_type=device.type, enabled=False):
                total_loss, loss_values = compute_loss(
                    output, batch, operator, config["loss"], physics_mode
                )
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {batch_index}")
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            if record_refinement:
                diagnostic = dict(epoch=epoch, global_step=global_step+1,
                                  **model.residual_refiner.last_diagnostics, **loss_values)
                for name, param in model.residual_refiner.named_parameters():
                    if name in ('output.weight', 'local.0.weight', 'blocks.0.qkv.weight'):
                        diagnostic[name+'_grad_norm'] = float(param.grad.norm()) if param.grad is not None else 0.
                with (output_dir/'refinement_diagnostics.jsonl').open('a',encoding='utf-8') as handle:
                    handle.write(json.dumps(diagnostic,ensure_ascii=False)+'\n')
            if record_relation:
                diagnostic = dict(epoch=epoch, global_step=global_step + 1,
                    gate_mean=float(model.relation.last_gate_mean),
                    relation_to_backbone_rms=float(model.last_relation_ratio),
                    projected_update_rms=float(model.relation.last_update_rms), **loss_values)
                for name, param in model.relation.named_parameters():
                    if name in ('project.weight', 'block.q.weight', 'block.k.weight', 'block.v.weight',
                                'block.ffn.0.weight', 'tokens.pair_project.weight'):
                        diagnostic[name + '_grad_norm'] = float(param.grad.norm()) if param.grad is not None else 0.
                with (output_dir/'module_diagnostics.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(diagnostic, ensure_ascii=False) + '\n')
            if record_context:
                gradient = model.depth_branch.context_proj.weight.grad
                diagnostic = {"epoch": epoch, "global_step": global_step + 1,
                              **model.last_diagnostics,
                              "context_projection_gradient_rms": float(
                                  gradient.detach().float().square().mean().sqrt().cpu())}
                with (output_dir / "context_diagnostics.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_values["gradient_norm"] = float(gradient_norm.detach().cpu())
            epoch_losses.append(loss_values)
            global_step += 1
            if log_interval > 0 and batch_index % log_interval == 0:
                print(
                    f"[{resolved_run_id}] epoch={epoch}/{epochs} batch={batch_index}/{len(train_loader)} "
                    f"step={global_step} loss={loss_values['total']:.6f}",
                    flush=True,
                )

        model.collect_diagnostics = False
        validation_metrics, validation_rows = evaluate_model(
            model,
            operator,
            validation_loader,
            device,
            amp_dtype,
            input_mode,
            physics_mode,
        )
        if stage_report and stage_report['freeze_backbone']:
            verify_frozen(model, stage_report['frozen_state_sha256'])
            baseline = read_json(output_dir/'epoch_zero.json')['validation']['density_sae']
            if not math.isclose(validation_metrics['initial_density_sae'], baseline, rel_tol=2e-5, abs_tol=1e-4):
                raise AssertionError('Frozen initial density drifted on validation')
        score = selection_value(validation_metrics, policy)
        geometry_score = selection_value(validation_metrics, geometry_policy)
        primary_improved = improves(score, best_score, policy["mode"])
        geometry_improved = improves(geometry_score, best_geometry_score, "max")
        if primary_improved:
            best_score, best_epoch = score, epoch
        if geometry_improved:
            best_geometry_score, best_geometry_epoch = geometry_score, epoch
        average_losses = {
            name: float(np.mean([row[name] for row in epoch_losses]))
            for name in epoch_losses[0]
        } if epoch_losses else {}
        history = {
            "epoch": epoch,
            "epoch_complete": len(epoch_losses) == len(train_loader),
            "epoch_seconds": round(time.perf_counter() - epoch_started, 2),
            "epoch_started_unix": time.time() - (time.perf_counter() - epoch_started),
            "epoch_finished_unix": time.time(),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train": average_losses,
            "validation": validation_metrics,
            "validation_score": objective_score(validation_metrics),
            "selection_policy": policy,
            "selection_value": score,
            "best_epoch": best_epoch,
            "best_geometry_epoch": best_geometry_epoch,
            "categorical_diagnostics": categorical_diagnostics(validation_rows),
        }

        payload = _checkpoint_payload(
            model, optimizer, scheduler, scaler, config, normalization,
            epoch, global_step, score, validation_metrics,
        )
        complete_epoch = len(epoch_losses) == len(train_loader)
        last_epoch, last_epoch_complete = epoch, complete_epoch
        if complete_epoch:
            epochs_completed = epoch
        payload.update({
            "best_score": best_score,
            "best_epoch": best_epoch,
            "best_geometry_score": best_geometry_score,
            "best_geometry_epoch": best_geometry_epoch,
            "selection_policy": policy,
            "geometry_selection_policy": geometry_policy,
            "epoch_complete": complete_epoch,
            "rng_state": capture_rng_state(device),
            "history_record": history,
            "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
        })
        if primary_improved:
            _save_checkpoint({**payload, "checkpoint_kind": "best",
                              "checkpoint_selection_policy": policy,
                              "checkpoint_selection_value": score}, checkpoint_dir / "best.pt")
            write_csv(output_dir / "metrics_per_sample.csv", validation_rows)
        if geometry_improved and training.get("save_geometry_checkpoint", False):
            _save_checkpoint({**payload, "checkpoint_kind": "best_geometry",
                              "checkpoint_selection_policy": geometry_policy,
                              "checkpoint_selection_value": geometry_score}, checkpoint_dir / "best_geometry.pt")
        _save_checkpoint({**payload, "checkpoint_kind": "last",
                          "checkpoint_selection_policy": selection_policy(config, "last"),
                          "checkpoint_selection_value": epoch}, last_path)
        if config["phase"] == "preflight" and epoch == 1:
            _save_checkpoint(payload, checkpoint_dir / "epoch_1_resume_check.pt")
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(history, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            f"[{resolved_run_id}] epoch={epoch} {policy['metric']}={score:.6f} "
            f"density_dice={validation_metrics['density_support_dice']:.6f} "
            f"fg_nrmse={validation_metrics['foreground_nrmse']:.6f}",
            flush=True,
        )
        if stopped_by_max_steps or (max_steps > 0 and global_step >= max_steps):
            break

    best_checkpoint = _torch_load(checkpoint_dir / "best.pt")
    model.load_state_dict(best_checkpoint["model"])
    final_metrics, final_rows = evaluate_model(
        model,
        operator,
        validation_loader,
        device,
        amp_dtype,
        input_mode,
        physics_mode,
    )
    write_csv(output_dir / "metrics_per_sample.csv", final_rows)
    summary = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_geometry_epoch": best_geometry_epoch,
        "best_geometry_score": best_geometry_score,
        "selection_policy": policy,
        "geometry_selection_policy": geometry_policy,
        "checkpoint_selection": {
            "best": {"epoch": best_epoch, "value": best_score, "policy": policy},
            "best_geometry": {"epoch": best_geometry_epoch, "value": best_geometry_score,
                              "policy": geometry_policy},
            "last": {"epoch": last_epoch, "epoch_complete": last_epoch_complete,
                     "policy": selection_policy(config, "last")},
        },
        "elapsed_seconds": round(elapsed_before_resume + time.perf_counter() - started, 2),
        "epochs_completed": epochs_completed,
        "experiment_id": config["experiment_id"],
        "final_validation_metrics": final_metrics,
        "final_validation_categorical_diagnostics": categorical_diagnostics(final_rows),
        "global_steps": global_step,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": parameter_count(model),
        "stage1": stage_report,
        "stage1_frozen_verified": bool(stage_report and stage_report['freeze_backbone']),
        "total_stage_training_epochs": (50 if stage_report else 0) + epochs_completed,
        "peak_gpu_mb": (
            round(torch.cuda.max_memory_allocated(device) / 1024**2, 1)
            if device.type == "cuda"
            else 0.0
        ),
        "peak_gpu_reserved_mb": (
            round(torch.cuda.max_memory_reserved(device) / 1024**2, 1)
            if device.type == "cuda" else 0.0
        ),
        "run_id": resolved_run_id,
        "seed": seed,
        "variant": variant,
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "RUN_COMPLETE.json", {
        "status": "PASS" if epochs_completed == epochs else "STOPPED_BY_MAX_STEPS",
        "epochs_completed": epochs_completed, "epochs_requested": epochs,
    })
    return output_dir


def load_run(run_dir: str | Path, device: torch.device, checkpoint: str = "best") -> tuple[
    dict[str, Any], Normalization, torch.nn.Module, torch.nn.Module
]:
    directory = Path(run_dir).resolve()
    checkpoint_name = checkpoint
    if checkpoint_name not in {"best", "last", "best_geometry"}:
        raise ValueError(f"unsupported checkpoint {checkpoint_name!r}")
    checkpoint = _torch_load(directory / "checkpoints" / f"{checkpoint_name}.pt")
    config = checkpoint["config"]
    evaluation_config = read_json(directory / 'config_snapshot.json')
    if evaluation_config.get('phase') == 'reuse':
        raise ValueError('Historical reuse adapters are not portable; use release_runtime.load_model')
    normalization = Normalization(
        scalar_rms=float(checkpoint["normalization"]["scalar_rms"]),
        spin1_rms=float(checkpoint["normalization"]["spin1_rms"]),
        spin2_rms=float(checkpoint["normalization"]["spin2_rms"]),
        raw6_rms=tuple(checkpoint["normalization"]["raw6_rms"]),
        epsilon=float(checkpoint["normalization"].get("epsilon", 1.0e-6)),
    )
    operator = build_operator(config, normalization, PROJECT_ROOT).to(device)
    model = build_model(config, normalization, operator).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.loaded_checkpoint_metadata = {
        "checkpoint": checkpoint_name,
        "epoch": int(checkpoint["epoch"]),
        "global_step": int(checkpoint["global_step"]),
        "selection_policy": checkpoint.get("checkpoint_selection_policy", selection_policy(config, checkpoint_name)),
        "selection_value": checkpoint.get("checkpoint_selection_value",
                                          int(checkpoint["epoch"]) if checkpoint_name == "last"
                                          else selection_value(checkpoint["metrics"], selection_policy(config, checkpoint_name))),
        "training_selection_policy": checkpoint.get("selection_policy", selection_policy(config)),
    }
    return config, normalization, model, operator
