'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Script for submitting all limited data test jobs as defined in lim_data_config.json

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
        "--model-dir-path",
        help="Path to directory holding all experiment model subdirectories",
        type=str,
        default="./lim_data_models",
    )
    parser.add_argument(
        "--outfile-dir-path",
        help="Directory in which to store all test result outfiles",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--swa",
        action='store_true',
        help="Raise this flag to test with SWA models instead of best models."
    )
    args = parser.parse_args()

    return args


def submit_test_job(
    args: Namespace,
    arch: str,
    outfile: str,
    model_path: str = None
) -> None:
    
    script = "scripts/launch_testing.sh"
    slurm_outfile = os.path.basename(script) + ".log-%j"

    command = [
        "sbatch", "-o", slurm_outfile,
        script,
        "--dataset", args.dataset,
        "--arch", arch,
        "--data-dir", args.data_dir,
        "--outfile", outfile,
    ]

    if model_path: command += ["--model-dir-path", model_path]
    if args.swa: command += ["--swa"]
    
    subprocess.run(command)


def main():

    args = parse_args()

    # Load experiment model configurations
    with open('lim_data_config.json', 'r') as f:
        config = json.load(f)

    for arch in config['archs']: 
        for percent in config['percs']:
            
            # Submit non zero-shot tests
            if percent > 0:
                for lr in config['lrs']:
                    
                    model_path = f"{args.model_dir_path}/{args.dataset}/perc_train_{percent}/lr_{lr}"
                    exp_name = f"{args.dataset}_{arch}_perctrain_{percent}_lr_{lr}{'_swa' if args.swa else ''}"
                    outfile = f"{args.outfile_dir_path}/"+exp_name+".csv"

                    submit_test_job(args, arch, outfile, model_path)
            
            # Submit zero-shot test for model
            else:
                outfile = f"{args.outfile_dir_path}/{args.dataset}_{arch}_zeroshot.csv"
                
                submit_test_job(args, arch, outfile)
                
if __name__=="__main__":
    main()
