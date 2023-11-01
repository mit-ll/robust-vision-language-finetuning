'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Utility functions used in training or testing of models

from argparse import Namespace
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Callable
import torch.distributed as dist
from torch import optim

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "redirect_output",
    "pprint_args",
    "save_log",
    "save_log_json",
    "vis_log",
    "vis_log_json",
    "distributed_lr_warmup"
]


def distributed_lr_warmup(configure_optimizer: Callable, n_warmup_epochs: int):
    def adjust_lr_schedule(name: str, model, lr: float, weight_decay: float):
        if dist.is_initialized():
            old_lr = lr
            lr = lr * dist.get_world_size()
            print(f"Distributed Training: old_lr={old_lr}, new_lr={lr}")
        optimizer, scheduler = configure_optimizer(name, model, lr, weight_decay)
        if dist.is_initialized():
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer, 
                start_factor=1.0 / dist.get_world_size(), 
                end_factor=1.0, 
                total_iters=n_warmup_epochs
            )
            scheduler = optim.lr_scheduler.ChainedScheduler([scheduler, warmup_scheduler])
        return optimizer, scheduler
    return adjust_lr_schedule


def pprint_args(args: Namespace):
    maxlen = 0
    for arg in vars(args):
        if len(arg) > maxlen:
            maxlen = len(arg)

    for arg in vars(args):
        print(f"    {arg:{maxlen}} : {getattr(args, arg)}")

    print("\n")


@contextmanager
def redirect_output(log_path: str):
    """Redirect standard output and error to a text file at the given path within the context.

    Args:
        log_path (str): Path to the log file. `.out` and `.err` will be appended to the end for stdout and stderr respectively.
    """
    new_stdout = open(Path(log_path).with_suffix(".out"), 'w')
    new_stderr = open(Path(log_path).with_suffix(".err"), 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdout.flush()
    old_stderr.flush()
    sys.stdout = new_stdout
    sys.stderr = new_stderr
    yield
    new_stdout.flush()
    new_stderr.flush()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    new_stdout.close()
    new_stderr.close()
    return


def save_log(val_acc_log, directory):
    val_acc_log = np.array(val_acc_log)  # Nepochs x 2 (epoch, val_acc)
    np.save(directory + "val_acc_log.npy", val_acc_log)


def save_log_json(data, directory):
    fname = directory + "val_acc_log.json"
    with open(fname, "w") as f:
        s = json.dumps(data, indent=4)
        f.write(s)


def vis_log(filename, fsize=(10, 10)):
    la = np.load(filename)
    epochs, val_acc = la[:, 0], la[:, 1]
    plt.figure(figsize=fsize)
    plt.plot(epochs, val_acc)
    plt.xlabel("Epoch")
    plt.ylabel("Val Acc")
    plt.grid()
    plt.show()


def vis_log_json(filename, fsize=(10, 10)):
    with open(filename, "r") as f:
        data = json.load(f)

    epoch = data["epoch"]
    k = data["k"]
    acc = np.array(data["accuracy"])

    f, ax = plt.subplots(figsize=fsize)
    ax.plot(epoch, acc)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Acc")
    ax.grid(True)
    plt.show()