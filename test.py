'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Testing script for ID and OOD generalization

import argparse
import os

import pandas as pd

from fmdg import (
    convert_arch_to_filename,
    evaluate_wilds,
    get_data,
    loader_funcs,
    load_model,
    make_model,
    parallelize,
)
from fmdg import pprint_args

arch_choices = loader_funcs.keys()


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
        choices=arch_choices,
        required=True,
    )
    parser.add_argument(
        "--outfile",
        help="Full path to output file",
        type=str,
        default="./results/results.csv",
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size; default = 128",
        type=int,
        default=128,
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
        help="Path to directory where saved models are stored. Model will be tested zero-shot if not provided; default = None",
        type=str,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Root directory containing the dataset; default = './data'",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Raise this flag to display a progress bar during training and evaluation",
    )
    parser.add_argument(
        "--swa",
        action='store_true',
        help="Raise this flag to test with SWA models instead of best models."
    )

    # print all arguments to the screen
    print("Testing with the following parameters")
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
    
    metrics_train_all = [None] * args.nruns
    metrics_id_val_all = [None] * args.nruns
    metrics_id_test_all = [None] * args.nruns
    metrics_ood_val_all = [None] * args.nruns
    metrics_ood_test_all = [None] * args.nruns

    for m in range(args.nruns):
        # create the model
        model, device = make_model(args.arch, n_classes=data.n_classes)

        # filenames for checkpoints
        if model_paths:
            print("loading model ", model_paths[m])
            model = load_model(model, device, model_paths[m])
        model = parallelize(model, device, args.arch)

        # optionally handle text input
        text_list = data.text_list if data.text_input else None

        # evaluate the model using train, ID validation, ID test, OOD validation, OOD test datasets
        metrics_train_all[m] = evaluate_wilds(
            model,
            device,
            data.train_loader,
            text_list=text_list,
            progress=args.progress,
        )
        metrics_id_val_all[m] = evaluate_wilds(
            model,
            device,
            data.id_val_loader,
            text_list=text_list,
            progress=args.progress,
        )
        metrics_id_test_all[m] = evaluate_wilds(
            model,
            device,
            data.id_test_loader,
            text_list=text_list,
            progress=args.progress,
        )
        metrics_ood_val_all[m] = evaluate_wilds(
            model,
            device,
            data.val_loader,
            text_list=text_list,
            progress=args.progress,
        )
        metrics_ood_test_all[m] = evaluate_wilds(
            model,
            device,
            data.test_loader,
            text_list=text_list,
            progress=args.progress,
        )

    metrics_train_all = pd.DataFrame.from_records(metrics_train_all)
    metrics_id_val_all = pd.DataFrame.from_records(metrics_id_val_all)
    metrics_id_test_all = pd.DataFrame.from_records(metrics_id_test_all)
    metrics_ood_val_all = pd.DataFrame.from_records(metrics_ood_val_all)
    metrics_ood_test_all = pd.DataFrame.from_records(metrics_ood_test_all)

    metrics_train_agg = metrics_train_all.agg(['mean', 'std'])
    metrics_id_val_agg = metrics_id_val_all.agg(['mean', 'std'])
    metrics_id_test_agg = metrics_id_test_all.agg(['mean', 'std'])
    metrics_ood_val_agg = metrics_ood_val_all.agg(['mean', 'std'])
    metrics_ood_test_agg = metrics_ood_test_all.agg(['mean', 'std'])

    agg_results_df = pd.concat({
        "Train": metrics_train_agg,
        "ID Val": metrics_id_val_agg,
        "ID Test": metrics_id_test_agg,
        "OOD Val": metrics_ood_val_agg,
        "OOD Test": metrics_ood_test_agg,
    })
    agg_results_df = agg_results_df.transpose()
    print(agg_results_df)

    # save to CSV file
    dirname = os.path.dirname(args.outfile)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    agg_results_df.to_csv(args.outfile)

if __name__ == "__main__":
    main()
