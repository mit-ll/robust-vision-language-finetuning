'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Script for visualizing results from limited data experiments as defined in lim_data_config.json

import json
import os
import argparse
from typing import Tuple, Dict
from argparse import Namespace
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="Dataset used to generate the results being evaluated",
        type=str,
        choices=["camelyon17", "camelyon17_0", "fmow", "fmow_0", "iwildcam", "iwildcam_0"],
        required=True,
    )
    parser.add_argument(
        "--results-dir",
        help="Path of directory containing all non-SWA experiment results",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "--swa-results-dir",
        help="Path of directory containing SWA results; default = None. Currently only supports vitb32_clip_finetune",
        type=str,
    )
    parser.add_argument(
        "--error-bars",
        help="Include error bars in plots",
        action="store_true",
    )
    args = parser.parse_args()

    return args


def _get_results(filename: str, metric: str) -> Tuple[float, float, float, float]:
    df = pd.read_csv(filename, index_col=0)

    id, ood = float(df.loc[metric]['ID Test']), float(df.loc[metric]['OOD Test'])
    id_std, ood_std = float(df.loc[metric]['ID Test.1']), float(df.loc[metric]['OOD Test.1'])

    return id, ood, id_std, ood_std


def aggregate_results(config: dict, args: Namespace) -> Dict[str, Dict[int, list]]:

    # Cycle through permutations in results directory and choose lr per model with best OOD metric
    res = {arch:{p:[] for p in config["percs"]} for arch in config["archs"]}

    metric = "F1-macro_all" if "iwildcam" in args.dataset else "acc_avg"

    for arch in config["archs"]:
        for percent in config["percs"]:

            if percent == 0:
                filename = f"{args.results_dir}/{args.dataset}_{arch}_zeroshot.csv"
                assert os.path.exists(filename)

                res[arch][percent] = _get_results(filename, metric)

            else:
                best_ood = 0
                
                for lr in config["lrs"]:

                    if "swa" in arch:
                        filename = f"{args.swa_results_dir}/{args.dataset}_{arch}_perctrain_{percent}_lr_{lr}_swa.csv"
                    else:
                        filename = f"{args.results_dir}/{args.dataset}_{arch}_perctrain_{percent}_lr_{lr}.csv"
                    assert os.path.exists(filename)

                    curr = _get_results(filename, metric)
                    ood = curr[1]
                    
                    if ood > best_ood:
                        res[arch][percent] = curr
                        best_ood = ood

    return res


def main():

    args = parse_args()

    with open('lim_data_config.json', 'r') as f:
        config = json.load(f)

    # Add ViTB32 CLIP end-to-end finetune SWA results if requested
    if args.swa_results_dir: config["archs"].append("vitb32_clip_finetune_swa")

    # Compile results
    res = aggregate_results(config, args)

    # Output results to json
    outfile = f"./{args.dataset}_lim_data_results.json"
    with open(outfile, 'w') as f:
        json.dump(res, f)

    # Plot ID Test and OOD Test metrics per model as function of data percentage available
    fig, (ax1, ax2) = plt.subplots(1,2, sharey=True, figsize=(20,10))
    fig.subplots_adjust(wspace=0.1)
    fig.supxlabel("Percentage of Training Data Used", fontsize=20)
    if "iwildcam" in args.dataset:
        fig.supylabel("Macro F1 Score", fontsize=20)
    if "fmow" in args.dataset:
        fig.supylabel("Top-1 Accuracy", fontsize=20)

    ax1.set_title("ID Test", fontsize=18)
    ax1.tick_params(axis="both", which="major", labelsize=14)
    ax1.grid(color='tab:gray', linestyle='--', linewidth=0.5)
    ax1.set_xlim([0,1])

    ax2.set_title("OOD Test", fontsize=18)
    ax2.tick_params(axis="both", which="major", labelsize=14)
    ax2.grid(color='tab:gray', linestyle='--', linewidth=0.5)
    ax2.set_xlim([0,1])

    linestyles = {
        "resnet50_pretrained_finetune": '-r',
        "resnet50_pretrained_linearprob": '--r',
        "resnet50_clip_finetune": '-y',
        "resnet50_clip_linearprob": '--y',
        "vitb32_pretrained_finetune": '-b',
        "vitb32_pretrained_linearprob": '--b',
        "vitb32_clip_finetune": '-g',
        "vitb32_clip_linearprob": '--g',
        "vitb32_clip_finetune_swa": '-r',
    }
    
    legend_labels = {
        "resnet50_pretrained_finetune":"ResNet50 Fine-tuned (E2E)",
        "resnet50_pretrained_linearprob":"ResNet50 Linear Probe",
        "resnet50_clip_finetune":"CLIP ResNet50 Fine-tuned (E2E)",
        "resnet50_clip_linearprob":"CLIP ResNet50 Linear Probe",
        "vitb32_pretrained_finetune":"ViT-B/32 Fine-tuned (E2E)",
        "vitb32_pretrained_linearprob":"ViT-B/32 Linear Probe",
        "vitb32_clip_finetune":"CLIP ViT-B/32 Fine-tuned (E2E)",
        "vitb32_clip_linearprob":"CLIP ViT-B/32 Linear Probe",
        "vitb32_clip_finetune_swa":"CLIP ViT-B/32 Fine-tuned (E2E) w/ SWA",
    }

    for arch in config["archs"]:

        percs = list(res[arch].keys())
        vals = list(res[arch].values())

        ids = [x[0] for x in vals]
        oods = [x[1] for x in vals]
        id_stds = [x[2] for x in vals]
        ood_stds = [x[3] for x in vals]
      
        if args.error_bars:
            ax1.errorbar(percs, ids, yerr=id_stds, capsize=10, fmt=f'{linestyles[arch]}o', label=legend_labels[arch])
            ax2.errorbar(percs, oods, yerr=ood_stds, capsize=10, fmt=f'{linestyles[arch]}o', label=legend_labels[arch])
        else:
            ax1.errorbar(percs, ids, yerr=None, fmt=f'{linestyles[arch]}o', label=legend_labels[arch])
            ax2.errorbar(percs, oods, yerr=None, fmt=f'{linestyles[arch]}o', label=legend_labels[arch])

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=14, bbox_to_anchor=(1, 1))

    outfile_plt = f"./{args.dataset}_lim_data_results.png"
    if not os.path.exists(outfile_plt):
        fig.savefig(outfile_plt)
    else:
        raise FileExistsError("Output file exists.")

if __name__=="__main__":
    main()