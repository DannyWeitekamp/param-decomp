"""Resumable training checkpoints.

Layout under `resume_dir`:

    step_00012000/
        shared.pt      # rank 0: component model state, optimizer state, step, world size
        rank{r}.pt     # every rank: persistent-PGD state, RNG states, data-loader positions
        COMPLETE       # written last; a directory without it is never loaded
    step_00012000.tmp/ # in-progress write, renamed to step_XXXXXXXX/ once every file is fsynced
    DONE               # training reached config.steps

The checkpoint for step `s` holds the state at the top of loop iteration `s`, i.e. after the
optimizer update of step `s - 1`. Resuming restarts the loop at `s`.

Persistent-PGD sources with per_batch_per_position scope are per rank (sharded by batch), so each
rank writes its own shard and nothing is gathered to rank 0.

Data position: each rank saves `IterableDataset.state_dict()` for its streaming train and eval
datasets (HF shard index + offset within the shard). The shuffle buffer contents are not part of
that state, so the stream after a resume is a slightly different permutation of the same
not-yet-seen examples.
"""

import os
import random
import shutil
import signal
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

import numpy as np
import torch
from datasets import IterableDataset
from torch.utils.data import DataLoader

from spd.log import logger
from spd.persistent_pgd import PersistentPGDState
from spd.utils.distributed_utils import (
    all_reduce,
    get_distributed_state,
    is_main_process,
    sync_across_processes,
)

N_CHECKPOINTS_TO_KEEP = 2
COMPLETE_MARKER = "COMPLETE"
DONE_MARKER = "DONE"

_stop_requested = False


def _request_stop(signum: int, _frame: FrameType | None) -> None:
    global _stop_requested
    _stop_requested = True
    logger.warning(f"Received signal {signal.Signals(signum).name}; will checkpoint and stop.")


def install_stop_signal_handlers() -> None:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGUSR1, _request_stop)


def stop_requested_on_any_rank(device: str) -> bool:
    flag = torch.tensor([1.0 if _stop_requested else 0.0], device=device)
    return all_reduce(flag).item() > 0


def _rank_and_world_size() -> tuple[int, int]:
    dist_state = get_distributed_state()
    if dist_state is None:
        return 0, 1
    return dist_state.rank, dist_state.world_size


def _step_dir_name(step: int) -> str:
    return f"step_{step:08d}"


def _complete_checkpoints(resume_dir: Path) -> list[Path]:
    """Complete checkpoint directories, oldest first."""
    dirs = [
        d
        for d in resume_dir.glob("step_*")
        if d.is_dir() and not d.name.endswith(".tmp") and (d / COMPLETE_MARKER).exists()
    ]
    return sorted(dirs, key=lambda d: int(d.name.removeprefix("step_")))


def latest_checkpoint(resume_dir: Path) -> Path | None:
    checkpoints = _complete_checkpoints(resume_dir)
    return checkpoints[-1] if checkpoints else None


def is_done(resume_dir: Path) -> bool:
    return (resume_dir / DONE_MARKER).exists()


def mark_done(resume_dir: Path) -> None:
    if is_main_process():
        _atomic_write_bytes(resume_dir / DONE_MARKER, b"")


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _save_fsynced(obj: Any, path: Path) -> None:
    with open(path, "wb") as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _dataset_of(loader: DataLoader[Any]) -> IterableDataset:
    dataset = loader.dataset
    assert isinstance(dataset, IterableDataset), (
        f"Resumable data position needs a streaming HF IterableDataset, got {type(dataset)}"
    )
    return dataset


def _rng_state(device: str) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if device.startswith("cuda"):
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _set_rng_state(state: dict[str, Any], device: str) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if device.startswith("cuda"):
        torch.cuda.set_rng_state(state["cuda"], device)


def _discard_checkpoint(checkpoint_dir: Path) -> None:
    """Rename out of the step_* namespace first so a partial delete can never look loadable.

    The delete itself may leave NFS placeholders for files another process still has open;
    those directories are retried on later saves instead of failing the run.
    """
    trash = checkpoint_dir.with_name("deleting_" + checkpoint_dir.name)
    os.rename(checkpoint_dir, trash)
    shutil.rmtree(trash, ignore_errors=True)
    if trash.exists():
        logger.warning(f"Could not fully delete {trash} (open NFS files?); will retry later")


def save_checkpoint(
    resume_dir: Path,
    step: int,
    component_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ppgd_states: list[PersistentPGDState],
    train_loader: DataLoader[Any],
    eval_loader: DataLoader[Any],
    batch_size: int,
    device: str,
) -> None:
    """Write a checkpoint for the top of step `step`. Must be called on all ranks."""
    rank, world_size = _rank_and_world_size()
    final_dir = resume_dir / _step_dir_name(step)
    tmp_dir = resume_dir / (_step_dir_name(step) + ".tmp")

    sync_across_processes()
    if is_main_process():
        resume_dir.mkdir(parents=True, exist_ok=True)
        for stale in resume_dir.glob("step_*.tmp"):
            shutil.rmtree(stale)
        assert not final_dir.exists(), f"checkpoint {final_dir} already exists"
        tmp_dir.mkdir()
    sync_across_processes()

    rank_state = {
        "rank": rank,
        "world_size": world_size,
        "ppgd_states": [s.state_dict() for s in ppgd_states],
        "rng": _rng_state(device),
        "train_data": _dataset_of(train_loader).state_dict(),
        "eval_data": _dataset_of(eval_loader).state_dict(),
    }
    _save_fsynced(rank_state, tmp_dir / f"rank{rank}.pt")

    if is_main_process():
        shared_state = {
            "step": step,
            "world_size": world_size,
            "batch_size": batch_size,
            "faithfulness_warmup_done": True,
            "component_model": component_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        _save_fsynced(shared_state, tmp_dir / "shared.pt")

    sync_across_processes()
    if is_main_process():
        expected = {"shared.pt"} | {f"rank{r}.pt" for r in range(world_size)}
        present = {p.name for p in tmp_dir.iterdir()}
        assert present == expected, f"checkpoint files {sorted(present)} != {sorted(expected)}"
        _atomic_write_bytes(tmp_dir / COMPLETE_MARKER, f"{step}\n".encode())
        _fsync_dir(tmp_dir)
        os.rename(tmp_dir, final_dir)
        _fsync_dir(resume_dir)
        for old in _complete_checkpoints(resume_dir)[:-N_CHECKPOINTS_TO_KEEP]:
            _discard_checkpoint(old)
        for leftover in resume_dir.glob("deleting_*"):
            shutil.rmtree(leftover, ignore_errors=True)
        logger.info(f"Saved resumable checkpoint {final_dir}")
    sync_across_processes()


@dataclass
class LoadedCheckpoint:
    step: int
    shared: dict[str, Any]
    rank_state: dict[str, Any]


def load_checkpoint_files(checkpoint_dir: Path, batch_size: int) -> LoadedCheckpoint:
    rank, world_size = _rank_and_world_size()
    # Read fully instead of mmap: an mmapped file stays open, and on NFS deleting an open file
    # leaves a .nfsXXXX placeholder that makes pruning this directory fail later.
    shared = torch.load(checkpoint_dir / "shared.pt", map_location="cpu", weights_only=False)
    assert shared["world_size"] == world_size, (
        f"checkpoint was written with world_size={shared['world_size']}, now {world_size}. "
        "Resuming into a different world size changes the run."
    )
    assert shared["batch_size"] == batch_size, (
        f"checkpoint batch_size={shared['batch_size']}, config batch_size={batch_size}"
    )
    assert shared["faithfulness_warmup_done"]
    rank_state = torch.load(
        checkpoint_dir / f"rank{rank}.pt", map_location="cpu", weights_only=False
    )
    assert rank_state["rank"] == rank and rank_state["world_size"] == world_size
    return LoadedCheckpoint(step=shared["step"], shared=shared, rank_state=rank_state)


def restore_data_positions(
    ckpt: LoadedCheckpoint, train_loader: DataLoader[Any], eval_loader: DataLoader[Any]
) -> None:
    """Must run before the first iterator over either loader is created."""
    _dataset_of(train_loader).load_state_dict(ckpt.rank_state["train_data"])
    _dataset_of(eval_loader).load_state_dict(ckpt.rank_state["eval_data"])


def restore_training_state(
    ckpt: LoadedCheckpoint,
    component_model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    ppgd_states: list[PersistentPGDState],
    device: str,
) -> None:
    """Restore model, optimizer, persistent-PGD state and RNG. Call last, right before the loop."""
    component_model.load_state_dict(ckpt.shared["component_model"])
    optimizer.load_state_dict(ckpt.shared["optimizer"])
    saved_ppgd = ckpt.rank_state["ppgd_states"]
    assert len(saved_ppgd) == len(ppgd_states), (
        f"{len(saved_ppgd)} saved persistent-PGD states, {len(ppgd_states)} configured"
    )
    for state, saved in zip(ppgd_states, saved_ppgd, strict=True):
        state.load_state_dict(saved)
    _set_rng_state(ckpt.rank_state["rng"], device)
