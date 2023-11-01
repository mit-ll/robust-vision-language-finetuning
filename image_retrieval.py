'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# CLIP image retrieval script

import argparse
import PIL

from clip import tokenize
import numpy as np
import torch
from tqdm import tqdm

from fmdg import (
    get_data_for_image_retrieval,
    load_model,
    make_model,
    parallelize,
    text_suffix,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["camelyon17", "fmow", "iwildcam"],
        default="fmow",
        help="Dataset name; default = 'fmow'",
    )
    parser.add_argument(
        "--default-weights",
        action="store_true",
        dest="default_weights",
        help="Raise this flag to use default OpenAI weights for the model",
    )
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        help="Path to model checkpoint used for image/text comparison",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        help="Text prompt to use for image/text comparison",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        help="Path to output file",
    )
    parser.add_argument(
        "--nimages",
        type=int,
        default=5,
        help="Number of images to retrieve",
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
        dest="progress",
        help="Raise this flag to display the progress bar",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # --- parse CLI arguments
    args = parse_args()
    prompt = " ".join(args.prompt)
    print(f"Dataset: {args.dataset}")
    print(f"Prompt: {prompt}")
    print(f"Output file: {args.outfile}")
    print(f"# Images: {args.nimages}")

    # --- get some data
    rawdata, loader = get_data_for_image_retrieval(args.dataset, root_dir=args.data_dir)

    # --- create model
    arch = "vitb32_clip" if args.default_weights else "vitb32_clip_finetune"
    n_classes = loader.dataset._n_classes
    model, device = make_model(arch, n_classes=n_classes)

    if not args.default_weights:
        print(f"Loading model parameters from {args.model_checkpoint}")
        model = load_model(model, device, args.model_checkpoint)

    model = parallelize(model, device, arch)
    model.eval()

    # --- tokenize the input prompt
    txt_tokens = tokenize(prompt)
    txt_tokens = txt_tokens.to(device)

    # --- loop over batches, saving the top N images
    global_count = 0
    best_n = -1 * np.ones(args.nimages, dtype=int)
    best_n_sim = -np.inf * np.ones(args.nimages)
    dataiter = tqdm(loader) if args.progress else loader

    with torch.no_grad():
        for batch in dataiter:
            # split the batch into raw data
            img, _, _ = batch

            # put data on device
            img = img.to(device)

            # get similarity between each image in the batch and the prompt
            sim, _ = model(img, txt_tokens)
            sim = sim.detach().cpu().numpy().copy()

            # save the current best N
            batch_idx = np.arange(img.shape[0], dtype=int)
            global_idx = batch_idx + global_count

            tmp_idx = np.hstack([best_n, global_idx])
            tmp_sim = np.hstack([best_n_sim, sim.squeeze()])
            idx = np.argsort(tmp_sim)[::-1][: args.nimages]
            best_n = tmp_idx[idx]
            best_n_sim = tmp_sim[idx]

            global_count += img.shape[0]

    # --- form output image
    imgs, labels = [], []
    for idx in best_n:
        im, label, _ = rawdata[idx]
        # add border
        im = PIL.ImageOps.expand(im, border=1, fill="white")
        imgs.append(im)
        labels.append(text_suffix[args.dataset][label])

    total_width = sum([x.size[0] for x in imgs])
    height = max([x.size[1] for x in imgs])

    outim = PIL.Image.new("RGB", (total_width, height))
    x_offset = 0
    for im in imgs:
        outim.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    outim.save(args.outfile)

    print("True labels:")
    for i in range(args.nimages):
        print(
            f"  Image {i} | similarity = {best_n_sim[i]:.2f} | true label = {labels[i]}"
        )
