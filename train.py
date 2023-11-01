'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Training script for multiple independent runs, datasets, architectures

import argparse
from contextlib import nullcontext
import os
from pathlib import Path

import torch.optim as optim

from fmdg import (
    loader_funcs,
    convert_arch_to_filename,
    get_data,
    make_model,
    pprint_args,
    train_wilds,
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    redirect_output,
    distributed_lr_warmup,
    parallelize,
)

arch_choices = loader_funcs.keys()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Dataset name",
        type=str,
        choices=["camelyon17", "camelyon17_0", "fmow", "fmow_0", "iwildcam", "iwildcam_0"],
        required=True,
    )
    parser.add_argument(
        "--arch",
        help="Model architecture; default = 'resnet50'",
        type=str,
        choices=arch_choices,
        required=True,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size; default = 128",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--train-alg",
        help="Training algorithm; default = 'ERM'",
        type=str,
        choices=["ERM"],
        default="ERM",
    )
    parser.add_argument(
        "--nepochs",
        help="Number of epochs; default = 30",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--swa-epochs",
        help="The number of epochs before the end of training when SWA will begin averaging.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--lr",
        help="Learning rate; default = 1e-5",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--swa-lr",
        help="High learning rate for SWA training",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--perc-train",
        help="Percentage of training data to use; default = 1.0",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--save-freq-ep",
        help="Epoch frequency for saving model; default = 1",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--nruns",
        help="Number of training runs; default = 3",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--num-workers",
        help="Number of workers to assign for data loading; default = 64",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--model-dir-path",
        dest="model_path",
        help="Path to directory where models will be saved; default = './models'",
        type=str,
        default="./models",
    )
    parser.add_argument(
        "--log-dir-path",
        dest="log_path",
        help="Path to directory where validation accuracy logs are saved; default = './logs'",
        type=str,
        default="./logs",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root directory containing the dataset; default = './data'",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["Adam", "SGD"],
        default="Adam",
        help="Optimizer to use during training; default = 'Adam'",
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="Enable sync-bn to activate synchronized batch norm for distributed training"
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Raise this flag to display a progress bar during training and evaluation",
    )

    # print all arguments to the screen
    args = parser.parse_args()

    return args

def configure_optimizer(name: str, model, lr: float, weight_decay: float):
    if name == "Adam":
        print("Using Adam optimizer")
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif name == "SGD":
        print("Using SGD optimizer")
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True,
        )
    else:
        raise ValueError(f"Optimizer {name} not supported")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    return optimizer, scheduler


def main():
    args = parse_args()

    if "MASTER_PORT" in os.environ and "MASTER_ADDR" in os.environ:
        setup_distributed()
    
    rank_out = Path(args.log_path, f"rank_{get_rank()}")
    rank_out.parent.mkdir(parents=True, exist_ok=True)
    with redirect_output(rank_out) if get_rank() != 0 else nullcontext():
        print("Training parameters")
        pprint_args(args)

        # dataset-specific parameters
        weight_decay = 1e-2 if "camelyon17" in args.dataset else 1e-4

        # create output directories if they don't already exist
        if not os.path.exists(args.model_path):
            print(f"Making {args.model_path}")
            Path(args.model_path).mkdir(parents=True, exist_ok=True)
        if not os.path.exists(args.log_path):
            print(f"Making {args.log_path}")
            Path(args.log_path).mkdir(parents=True, exist_ok=True)

        # only ERM is currently supported
        if not args.train_alg == "ERM":
            raise ValueError("ERM is the only training algorithm currently supported")

        data = get_data(
            args.dataset,
            args.arch,
            args.batch_size,
            args.num_workers,
            root_dir=args.data_dir,
            percentage_train=args.perc_train,
            rank=get_rank(),
            world_size=get_world_size(),
        )
        print(data)

        # model training
        for m in range(args.nruns):
            # create the model
            model, device = make_model(args.arch, n_classes=data.n_classes)
            model = parallelize(model, device, args.arch, sync_bn=args.sync_bn)

            # filenames for checkpoints and log files
            bfname = convert_arch_to_filename(args.arch)
            model_path = f"{args.model_path}/{bfname}_{args.dataset}_ERM_run{m+1}_chkpt_"
            log_dir = f"{args.log_path}/{bfname}_{args.dataset}_ERM_run{m+1}_"
            
            optimizer, scheduler = distributed_lr_warmup(configure_optimizer, n_warmup_epochs=5)(
                args.optimizer,
                model,
                args.lr,
                weight_decay,
            )

            if args.swa_epochs >= args.nepochs:
                raise ValueError(f"SWA epochs ({args.swa_epochs}) must be less than the total number of epochs ({args.nepochs}).")

            model = train_wilds(
                model,
                device,
                data, # DGData object
                optimizer,
                scheduler,
                n_epochs=args.nepochs,
                swa_epochs=args.swa_epochs,
                swa_lr=args.swa_lr,
                eval_freq=args.save_freq_ep,
                train_alg=args.train_alg,
                model_path=model_path,
                log_dir=log_dir,
                progress=args.progress,
            )

    cleanup_distributed()


if __name__ == "__main__":
    main()
