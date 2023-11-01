'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Image retrieval testing script

import argparse
import json
import os

import numpy as np

from fmdg import (
    convert_arch_to_filename,
    evaluate_retrieval,
    get_data,
    loader_funcs,
    load_model,
    make_model,
    pprint_args,
    parallelize,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="Dataset name; default = 'camelyon17'",
        type=str,
        choices=["camelyon17", "camelyon17_0", "fmow", "fmow_0", "iwildcam", "iwildcam_0"],
        required=True,
    )
    parser.add_argument(
        "--arch",
        help="Model architecture; default = 'resnet50'",
        type=str,
        choices=loader_funcs.keys(),
        required=True,
    )
    parser.add_argument(
        "--outfile",
        help="Full path to output file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--k",
        help="k value to use for recall @ k metric",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--batch_size",
        help="Batch size; default = 128",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--nruns",
        help="Number of training runs to evaluate; default = 3",
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
        "--model-path",
        dest="model_path",
        help="Path to directory where models will be saved; default = None",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root directory containing the dataset; default = './data'",
    )
    parser.add_argument(
        "--swa",
        action='store_true',
        help="Raise this flag to test with SWA models instead of best models."
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Raise this flag to display a progress bar during training and evaluation",
    )

    # print all arguments to the screen
    print("Evaluating retrieval with the following parameters")
    args = parser.parse_args()
    pprint_args(args)

    return args


def main():
    args = parse_args()

    data = get_data(
        args.dataset,
        args.arch,
        args.batch_size,
        args.num_workers,
        root_dir=args.data_dir,
    )
    print(data)

    # handle variable input for model loading
    bfname = convert_arch_to_filename(args.arch)
    if args.model_path is not None:
        model_paths = [f"{args.model_path}/{bfname}_{args.dataset}_ERM_run{m+1}_chkpt_{'best' if not args.swa else 'swa'}.pt" for m in range(args.nruns)]
        for p in model_paths:
            if not os.path.exists(p):
                raise ValueError(f"Model path {p} does not exist")
    else:
        model_paths = []
        args.nruns = 1
        print(
            f"No model path specified; will use model as it is instantiated, which may include pretrained weights"
        )

    # initialize results array
    recall_at_k = np.zeros(args.nruns)

    for m in range(args.nruns):
        model, device = make_model(args.arch, data.n_classes)

        # filenames for checkpoints
        if model_paths:
            print("loading model ", model_paths[m])
            model = load_model(model, device, model_paths[m])
        model = parallelize(model, device, args.arch)

        recall_at_k[m] = evaluate_retrieval(model, device, data, args.k, args.progress)

    print(
        (
            f"Recall @ k (k = {args.k}) performance across {args.nruns} training runs: "
            f"mean = {recall_at_k.mean():.4f}, std = {recall_at_k.std():.4f}"
        )
    )

    # save data to disk
    save_data = {
        "arch": args.arch,
        "dataset": args.dataset,
        "recall_at_k": recall_at_k.tolist(),
    }

    dirname = os.path.dirname(args.outfile)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(args.outfile, "w") as f:
        json.dump(save_data, f)


if __name__ == "__main__":
    main()
