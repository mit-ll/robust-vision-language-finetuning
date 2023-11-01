'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Script for submitting all limited data train jobs as defined in lim_data_config.json

import os
import subprocess
import json
import argparse
from argparse import Namespace

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
        "--data-dir",
        help="Root directory containing the dataset; default = './data'",
        type=str,
        default="./data",
    )
    parser.add_argument(
        "--nepochs",
        help="Number of epochs per experiment",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--swa-epochs",
        help="The number of epochs before the end of training when SWA will begin averaging.",
        type=int,
        default=0
    )
    parser.add_argument(
        "--model-dir-path",
        help="Path to directory that will hold all experiment model subdirectories",
        type=str,
        default="./lim_data_models",
    )
    parser.add_argument(
        "--log-dir-path",
        help="Path to directory that will hold all experiment log subdirectories",
        type=str,
        default="./lim_data_logs",
    )
    parser.add_argument(
        "--distributed-training",
        help="Run experiments using distributed training via DDP",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        action="store_true",
        help="Enable sync-bn to activate synchronized batch norm for distributed training"
    )
    args = parser.parse_args()

    return args


def submit_train_job(
    args: Namespace,
    arch: str,
    model_path: str,
    log_path: str,
    lr: float,
    perc_train: float,
) -> None:
    
    script = "scripts/launch_training_distributed.sh" if args.distributed_training else "scripts/launch_training.sh"
    slurm_outfile = os.path.basename(script) + ".log-%j"

    command = [
        "sbatch", "-o", slurm_outfile,
        script,
        "--dataset", args.dataset,
        "--arch", arch,
        "--data-dir", args.data_dir,
        "--model-dir-path", model_path,
        "--log-dir-path", log_path,
        "--lr", f"{lr}",
        "--perc-train", f"{perc_train}",
        "--nepochs", f"{args.nepochs}",
        "--swa-epochs", f"{args.swa_epochs}"
    ]

    if args.sync_bn: command += ["--sync-bn"]

    subprocess.run(command)


def main():
    
    args = parse_args()

    # Load experiment model configurations
    with open('lim_data_config.json', 'r') as f:
        config = json.load(f)

    # Submit jobs to train all model permutations in parallel
    for arch in config['archs']: 
        for percent in config['percs']:
            if percent > 0:
                for lr in config['lrs']:
                    
                    model_path = f"{args.model_dir_path}/{args.dataset}/perc_train_{percent}/lr_{lr}"
                    log_path = f"{args.log_dir_path}/{args.dataset}/perc_train_{percent}/lr_{lr}"

                    submit_train_job(
                        args=args,
                        arch=arch,
                        model_path=model_path,
                        log_path=log_path,
                        lr=lr,
                        perc_train=percent,
                    )
                    

if __name__=="__main__":
    main()
