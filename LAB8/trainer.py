"""
ex05_mnist_cnn_train.py
-----------------------
Companion to ex05_mnist_cnn.cu.

Runs:
  Part A : full training of MnistCNN for 10 epochs.
  Part B : 4-config ablation study, 5 epochs each.
  Part C : data-augmentation comparison.
  Part D : CUDA streams (async H2D), AMP, torch.profiler.

Run modes (selectable via --mode):
    full    -- everything (slow)
    a       -- Part A only
    ablate  -- Part B only
    aug     -- Part C only
    bonus   -- Part D only

Usage:
    python ex05_mnist_cnn_train.py --mode full --epochs 10 --batch 256

Outputs results as `results.json`, plots into `./plots/`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Model
# =============================================================================
class MnistCNN(nn.Module):
    """
    conv -> bn -> relu -> pool -> conv -> bn -> relu -> pool -> fc.
    Toggleable BatchNorm and Dropout for the ablation study.
    """

    def __init__(self, use_bn: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        # TODO B1 - layer definitions.
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(32) if use_bn else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)
        # TODO B2 - classifier head.
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO B3 - forward graph.
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


# =============================================================================
# Data
# =============================================================================
def get_loaders(batch: int, augment: bool, num_workers: int = 2):
    train_tfms = [transforms.ToTensor(),
                  transforms.Normalize((0.1307,), (0.3081,))]
    if augment:
        # TODO Part C : augmentations.
        train_tfms = [
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, shear=10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(p=0.1),
        ]
    test_tfms = [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,))]

    train = torchvision.datasets.MNIST(root="./data", train=True, download=True,
                                       transform=transforms.Compose(train_tfms))
    test = torchvision.datasets.MNIST(root="./data", train=False, download=True,
                                      transform=transforms.Compose(test_tfms))
    train_loader = DataLoader(train, batch_size=batch, shuffle=True,
                              num_workers=num_workers,
                              pin_memory=(DEVICE.type == "cuda"))
    test_loader = DataLoader(test, batch_size=512, shuffle=False,
                             num_workers=num_workers,
                             pin_memory=(DEVICE.type == "cuda"))
    return train_loader, test_loader


# =============================================================================
# Optim / scheduler builders.
# =============================================================================
def build_optimizer(name: str, params, lr: float):
    # TODO D1 - return correct optimizer for the requested name.
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    raise ValueError(f"unknown optimizer '{name}'")


def build_scheduler(name: str, optimizer, epochs: int):
    # TODO D2 - cosine, step, none.
    name = (name or "none").lower()
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    raise ValueError(f"unknown scheduler '{name}'")


# =============================================================================
# Train / evaluate.
# =============================================================================
@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    test_acc: float
    time_s: float
    mem_mb: float


def evaluate(model, loader) -> float:
    # TODO C3 - evaluation loop.
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_one_run(
    model: nn.Module,
    train_loader,
    test_loader,
    epochs: int,
    optim_name: str,
    sched_name: str,
    lr: float,
    use_amp: bool = False,
    log_prefix: str = "",
) -> dict:
    model = model.to(DEVICE)
    optimizer = build_optimizer(optim_name, model.parameters(), lr)
    scheduler = build_scheduler(sched_name, optimizer, epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history: list[EpochStats] = []
    epochs_to_95 = None
    t_total0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running, correct, total = 0.0, 0, 0
        t0 = time.perf_counter()
        for x, y in train_loader:
            # TODO C1 - data to GPU.
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            ctx = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else nullcontext()
            # TODO C2 - forward / backward / step.
            with ctx:
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward() if use_amp else loss.backward()
            (scaler.step(optimizer), scaler.update()) if use_amp else optimizer.step()

            running += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        if scheduler is not None:
            scheduler.step()

        train_loss = running / total
        train_acc = correct / total
        test_acc = evaluate(model, test_loader)
        epoch_t = time.perf_counter() - t0
        mem_mb = (torch.cuda.max_memory_allocated() / 1024 / 1024
                  if DEVICE.type == "cuda" else 0.0)

        if epochs_to_95 is None and test_acc >= 0.95:
            epochs_to_95 = epoch

        history.append(EpochStats(epoch, train_loss, train_acc, test_acc,
                                  epoch_t, mem_mb))
        print(f"{log_prefix}epoch {epoch:2d}: "
              f"train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} "
              f"test_acc={test_acc:.4f} "
              f"({epoch_t:.1f}s, {mem_mb:.0f} MiB)")

    return {
        "epochs": [asdict(h) for h in history],
        "final_test_acc": history[-1].test_acc,
        "epochs_to_95": epochs_to_95,
        "total_time_s": time.perf_counter() - t_total0,
    }


# =============================================================================
# Experiments
# =============================================================================
def part_a_full_training(args) -> dict:
    print("\n=== Part A : Full training (10 epochs) ===")
    train_loader, test_loader = get_loaders(args.batch, augment=False)
    model = MnistCNN(use_bn=True, dropout=0.5)
    return train_one_run(model, train_loader, test_loader,
                         epochs=args.epochs,
                         optim_name="adam",
                         sched_name="cosine",
                         lr=1e-3,
                         log_prefix="[A] ")


def part_b_ablation(args) -> dict:
    print("\n=== Part B : 4-config ablation (5 epochs each) ===")
    train_loader, test_loader = get_loaders(args.batch, augment=False)
    configs = [
        ("baseline",   dict(use_bn=False, dropout=0.0), "adam", "none", 1e-3),
        ("+BN",        dict(use_bn=True,  dropout=0.0), "adam", "none", 1e-3),
        ("+Dropout",   dict(use_bn=True,  dropout=0.5), "adam", "none", 1e-3),
        ("SGD+cosine", dict(use_bn=True,  dropout=0.5), "sgd",  "cosine", 1e-2),
    ]
    out = {}
    for name, mkw, opt, sch, lr in configs:
        print(f"\n-- config: {name} --")
        torch.manual_seed(0)
        m = MnistCNN(**mkw)
        out[name] = train_one_run(m, train_loader, test_loader,
                                  epochs=5,
                                  optim_name=opt,
                                  sched_name=sch,
                                  lr=lr,
                                  log_prefix=f"[{name}] ")
    return out


def part_c_augment(args) -> dict:
    print("\n=== Part C : Augmentation comparison (5 epochs each) ===")
    out = {}
    for tag, augment in [("no_aug", False), ("with_aug", True)]:
        print(f"\n-- {tag} --")
        torch.manual_seed(0)
        train_loader, test_loader = get_loaders(args.batch, augment=augment)
        m = MnistCNN(use_bn=True, dropout=0.5)
        out[tag] = train_one_run(m, train_loader, test_loader,
                                 epochs=5,
                                 optim_name="adam",
                                 sched_name="cosine",
                                 lr=1e-3,
                                 log_prefix=f"[{tag}] ")
    return out


def part_d_bonus(args) -> dict:
    """CUDA streams + AMP + profiler."""
    print("\n=== Part D : Bonus (streams, AMP, profiler) ===")
    out = {}

    # ----- AMP timing -----
    train_loader, test_loader = get_loaders(args.batch, augment=False)

    print("\n-- AMP off --")
    torch.manual_seed(0)
    m = MnistCNN(use_bn=True, dropout=0.5)
    out["fp32"] = train_one_run(m, train_loader, test_loader,
                                epochs=3, optim_name="adam",
                                sched_name="none", lr=1e-3,
                                use_amp=False, log_prefix="[fp32] ")

    print("\n-- AMP on --")
    torch.manual_seed(0)
    m = MnistCNN(use_bn=True, dropout=0.5)
    out["amp"] = train_one_run(m, train_loader, test_loader,
                               epochs=3, optim_name="adam",
                               sched_name="none", lr=1e-3,
                               use_amp=True, log_prefix="[amp] ")

    # ----- Async H2D using stream -----
    if DEVICE.type == "cuda":
        print("\n-- Async H2D vs sync (100 batches) --")
        x_cpu = torch.randn(args.batch, 1, 28, 28, pin_memory=True)
        # sync
        t0 = time.perf_counter()
        for _ in range(100):
            _ = x_cpu.to(DEVICE)
            torch.cuda.synchronize()
        t_sync = time.perf_counter() - t0

        # async, separate stream
        stream = torch.cuda.Stream()
        t0 = time.perf_counter()
        for _ in range(100):
            with torch.cuda.stream(stream):
                _ = x_cpu.to(DEVICE, non_blocking=True)
        torch.cuda.synchronize()
        t_async = time.perf_counter() - t0
        out["h2d_sync_s"] = t_sync
        out["h2d_async_s"] = t_async
        print(f"sync H2D : {t_sync*1000:.2f} ms total")
        print(f"async H2D: {t_async*1000:.2f} ms total")
        print(f"speedup  : {t_sync/t_async:.2f}x")

    # ----- Profiler -----
    print("\n-- Profiler (3 iters) --")
    torch.manual_seed(0)
    m = MnistCNN(use_bn=True, dropout=0.5).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    it = iter(train_loader)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if DEVICE.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(3):
            x, y = next(it)
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = m(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
    sort_key = "cuda_time_total" if DEVICE.type == "cuda" else "cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=10))
    return out


# =============================================================================
# Entry
# =============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full",
                   choices=["full", "a", "ablate", "aug", "bonus"])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch",  type=int, default=256)
    args = p.parse_args()

    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU   : {torch.cuda.get_device_name(0)}")

    results: dict[str, Any] = {}
    if args.mode in ("a", "full"):       results["part_a"] = part_a_full_training(args)
    if args.mode in ("ablate", "full"):  results["part_b"] = part_b_ablation(args)
    if args.mode in ("aug", "full"):     results["part_c"] = part_c_augment(args)
    if args.mode in ("bonus", "full"):   results["part_d"] = part_d_bonus(args)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\n[done] results saved to results.json")


if __name__ == "__main__":
    main()
