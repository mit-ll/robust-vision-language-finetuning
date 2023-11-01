'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# Flexible Adaptation module

import os
from socket import gethostname
import datetime
from typing import List, Optional, Tuple
from collections import defaultdict

import clip
import numpy as np
import open_clip
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from tqdm import tqdm

from fmdg._utils import save_log_json
from fmdg._wrappers import CLIP, CLIP_img, OpenCLIP
from fmdg._archs import DGData

__all__ = [
    "make_model",
    "train_wilds",
    "evaluate_wilds",
    "evaluate_retrieval",
    "predict",
    "save_model",
    "load_model",
    "setup_distributed",
    "cleanup_distributed",
    "get_rank",
    "get_world_size",
    "parallelize",
]

RANK = 0
GPU_ID = 0
WORLD_SIZE = 1

def get_rank():
    return RANK

def get_world_size():
    return WORLD_SIZE

def setup_distributed():
    if "MASTER_PORT" not in os.environ or "MASTER_ADDR" not in os.environ:
        raise AssertionError("Missing master address and port environment variables.")
    
    global RANK
    global WORLD_SIZE
    if "SLURM_PROCID" in os.environ and "SLURM_NTASKS" in os.environ:
        RANK = int(os.environ["SLURM_PROCID"])
        WORLD_SIZE = int(os.environ["SLURM_NTASKS"])
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        RANK = int(os.environ["RANK"])
        WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    else:
        raise AssertionError("Missing environment variables for world info; ($SLURM_PROCID and $SLURM_NTASKS) or ($RANK and $WORLD_SIZE) required.")
    
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    global GPU_ID
    
    # Apparently slurm can sometimes switch up the ordering of processes on nodes, possibly based on CUDA_LAUNCH_BLOCKING?
    # num_nodes = int(os.environ["SLURM_NNODES"])
    # GPU_ID = RANK // num_nodes
    GPU_ID = RANK % num_gpus
    
    torch.cuda.set_device(GPU_ID)
    # No backend specified, means both Gloo (cpu) and NCCL (gpu) backends will be created
    # Init method reads MASTER_ADDR and MASTER_PORT from environment,
    # we additionally specify RANK and WORLD_SIZE
    dist.init_process_group(backend="nccl", rank=RANK, world_size=WORLD_SIZE)
    print(f"World Size {WORLD_SIZE}, Rank {RANK} on {gethostname()} using device {torch.device(GPU_ID)}", flush=True)
    return

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

def parallelize(
    model: nn.Module,
    device: torch.device,
    arch: str,
    sync_bn: bool = False
) -> nn.Module:

    # Use DDP when launched with distributed training
    if dist.is_initialized():
        print(f"Using {WORLD_SIZE} GPU(s).")
        if any((p.requires_grad for p in model.parameters())):
            model = nn.parallel.DistributedDataParallel(model)
            if sync_bn:
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            print("All model parameters are frozen. Running without DDP.")
    
    # Use DP if possible
    else:
        e2e_clip = ["resnet50_clip", "resnet50_clip_finetune", "vitb32_clip", "vitb32_clip_finetune"]
        
        if arch not in e2e_clip and torch.cuda.device_count() > 1:  # cannot distribute end-to-end CLIP using DP since have to distribute text inputs also when batch is split over GPUs
            print("Using ", torch.cuda.device_count(), "GPU(s)")
            model = nn.DataParallel(model)
        else:
            print("Using ", int(device != "cpu"), "GPU(s)")
    
    return model


def make_model(
    arch: str = "resnet50",
    n_classes: int = 2,
) -> Tuple[nn.Module, torch.device]:
    device = torch.device(GPU_ID) if torch.cuda.is_available() else torch.device("cpu")
    print(f"Rank {RANK} device {device}")

    if arch == "resnet50":
        model = models.resnet50(pretrained=False, num_classes=n_classes)
    elif arch == "resnet50_pretrained_finetune":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif arch == "resnet50_pretrained_linearprob":
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        # Parameters of newly constructed modules have requires_grad=True by default
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif arch == "resnet50_clip_linearprob":
        # see: https://github.com/openai/CLIP
        clip_model, preprocess = clip.load("RN50", device=device)
        num_ftrs = 1024
        model = CLIP_img(clip_model, num_ftrs, n_classes, device)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif arch == "resnet50_clip":
        clip_model, preprocess = clip.load("RN50", device=device)
        num_ftrs = 1024
        model = CLIP(clip_model)
        for param in model.parameters():
            param.requires_grad = False
    elif arch == "resnet50_clip_finetune":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "RN50",
            pretrained="openai",
        )
        model = OpenCLIP(clip_model)
    elif arch == "vitb32":
        model = models.vit_b_32(pretrained=False, num_classes=n_classes)
    elif arch == "vitb32_pretrained_finetune":
        model = models.vit_b_32(pretrained=True)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, n_classes)
    elif arch == "vitb32_pretrained_linearprob":
        model = models.vit_b_32(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, n_classes)
    elif arch == "vitb32_clip_linearprob":
        # see: https://github.com/openai/CLIP
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        num_ftrs = 512
        model = CLIP_img(clip_model, num_ftrs, n_classes, device)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
    elif arch == "vitb32_clip":
        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        num_ftrs = 512
        model = CLIP(clip_model)
        for param in model.parameters():
            param.requires_grad = False
    elif arch == "vitb32_clip_finetune":
        clip_model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="openai",
        )
        model = OpenCLIP(clip_model)
    else:
        raise ValueError("Invalid architecture.")

    model.to(device=device)
    return model, device


def train_wilds(
    model: nn.Module,
    device: torch.device,
    data: DGData,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.StepLR,
    n_epochs: int,
    swa_epochs: int,
    swa_lr: float,
    eval_freq: int = 5,
    train_alg: str = "ERM",
    model_path: str = "./models/checkpoint_",
    log_dir: str = "./logs/final_",
    progress: bool = True,
):

    print("Training mode:", train_alg)
    print("Group id:", data.group_id)

    loss_fn = nn.CrossEntropyLoss()

    # tokenize text input if it exists
    if data.text_input:
        text_list = data.text_list
        text = torch.cat([clip.tokenize(text_inp) for text_inp in text_list])
        text = text.to(device)

    # choose relevant output metric based on dataset
    metric = "F1-macro_all" if "iwildcam" in data.dataset_name else "acc_avg"
    metric_eval_best = 0.0
    save_data = defaultdict(list)

    ### SWA Section ###
    swa_model = torch.optim.swa_utils.AveragedModel(model, device=device)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer=optimizer, swa_lr=swa_lr, anneal_epochs=10 if swa_epochs >= 10 else max(1, swa_epochs))
    swa_start = (n_epochs - swa_epochs - 1)
    ### SWA Section ###

    # step through training epochs
    for epoch in range(n_epochs):
        model.train()

        loss_train = 0.0

        # wrap loader in tqdm if a progress bar is requested
        train_loader = tqdm(data.train_loader) if progress else data.train_loader
        t_start = datetime.datetime.now()

        for batch in train_loader:
            x, y, metadata = batch

            x = x.to(device=device)
            y = y.to(device=device)

            if data.group_id is not None:  # single-source
                z = data.grouper.metadata_to_group(metadata)
                z = z.to(device=device)
                # select data based on group_id
                x = x[z == data.group_id, :, :, :]
                y = y[z == data.group_id]

            if not data.text_input:
                outputs = model(x)
            else:
                outputs, _ = model(x, text)

            loss = loss_fn(outputs, y)

            optimizer.zero_grad()
            if loss.requires_grad:
                loss.backward()
            optimizer.step()

            loss_train += loss.item()

        t_end = datetime.datetime.now()
        loss = loss_train / len(data.train_loader)
        print(f"Epoch {epoch+1:>3d}; Time: {t_end - t_start}; Loss = {loss:.4f}")

        if swa_epochs > 0 and epoch > swa_start:
            print(f"LR: {swa_scheduler.get_last_lr()}")
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            print(f"LR: {scheduler.get_last_lr()}")
            scheduler.step()

        if ((epoch == 0) or ((epoch + 1) % eval_freq == 0)) and RANK == 0:
            eval_model = model.module if isinstance(model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)) else model
            results = evaluate_wilds(
                eval_model,
                device,
                data.val_loader,
                text_list=data.text_list,
                progress=progress,
            )
            metric_eval = results[metric]

            eval_model.train()
            
            save_model(eval_model, model_path, epoch + 1) 

            save_data["epoch"].append(epoch)
            save_data[metric].append(metric_eval)     
            save_log_json(save_data, log_dir)

            # save the model if this is the highest metric measured so far
            if metric_eval > metric_eval_best:
                metric_eval_best = metric_eval
                save_model(eval_model, model_path + "best")
    if RANK == 0:
        save_log_json(save_data, log_dir)

    if swa_epochs > 0:
        torch.optim.swa_utils.update_bn(train_loader, swa_model)
        average_model = swa_model.module
        average_model = average_model.module if isinstance(average_model, (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)) else average_model
        save_model(average_model, model_path + "swa")
    print("Training complete, model saved:", model_path)
    return model


def evaluate_wilds(
    model: nn.Module,
    device: torch.device,
    val_loader: torch.utils.data.DataLoader,
    text_list: Optional[List[str]] = None,
    progress: bool = False,
):
    model.eval()

    # compute text tokens if the model takes them
    if text_list is not None:
        text = torch.cat([clip.tokenize(text_inp) for text_inp in text_list]).to(device)

    y_true_all, y_pred_all, y_meta_all = [], [], []

    with torch.no_grad():
        data_iter = val_loader if not progress else tqdm(val_loader)
        for batch in data_iter:
            x, y, meta = batch

            xdev = x.to(device=device)

            # model forward pass
            if text_list is None:
                outputs = model(xdev)
            else:
                outputs, _ = model(xdev, text)

            y_pred = outputs.detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            y_true = y.numpy()

            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
            y_meta_all.extend(meta)

    results, results_str = val_loader.dataset.eval(
        torch.tensor(y_pred_all), 
        torch.tensor(y_true_all), 
        torch.stack(y_meta_all))
    
    print(results_str)
    return results


def evaluate_retrieval(model, device, data: DGData, k: int, progress: bool = False):
    model.eval()

    # tokenize text input if it exists
    if data.text_input:
        print("Will use text prompts for inference")
        text_list = data.text_list
        text = torch.cat([clip.tokenize(text_inp) for text_inp in text_list])
        text = text.to(device)

    with torch.no_grad():
        loader = tqdm(data.test_loader) if progress else data.test_loader

        allprobs = []
        alllabels = []
        for batch in loader:
            images, labels, _ = batch

            img_dev = images.to(device)

            if not data.text_input:
                outputs = model(img_dev)
                probs = outputs.softmax(dim=-1)
            else:
                _, probs = model(img_dev, text)

            allprobs.append(probs.detach().cpu())
            alllabels.append(labels)

        # stack the tensors together to get a 2D tensor with shape
        # (nimages) x (nclasses), do the same for the labels
        probs = torch.cat(allprobs, 0)
        labels = torch.cat(alllabels, 0)

        # sort in order of output probability
        _, ind = probs.sort(dim=0, descending=True)

        # for each class, count how many of the top-k retrieved images
        # match that class
        print(f"Recall @ k (k = {k:d}) stats:")
        recall_at_k = np.zeros(data.n_classes)
        for classind in range(data.n_classes):
            topk_labels = labels[ind[:k, classind]]
            ncorrect = sum(topk_labels == classind).item()
            recall_at_k[classind] = ncorrect / k
            print(f" {classind:>3d}: {ncorrect / k:.2f}")

        avg = recall_at_k.mean()
        print(f"Mean: {avg:.3f}")

        return avg


def predict(model, device, val_loader, n_classes=2, text_list=None):
    model.eval()

    if text_list is not None:
        text = torch.cat([clip.tokenize(text_inp) for text_inp in text_list]).to(device)

    total = 0
    softmax = torch.nn.Softmax(dim=1)
    p_pred = torch.zeros((len(val_loader.dataset), n_classes))
    y_true = torch.zeros((len(val_loader.dataset)))
    with torch.no_grad():
        start = 0
        for batch in tqdm(val_loader):
            x, y, _ = batch

            batch_size = x.shape[0]
            x = x.to(device=device)
            y = y.to(device=device)

            # this should change based on image and text inputs
            if text_list is None:
                outputs = model(x)
                # convert logits to softmax scores
                p_pred[start : start + batch_size, :] = softmax(outputs)
            else:  # CLIP multimodal inputs
                outputs, probs = model(x, text)
                p_pred[start : start + batch_size, :] = probs

            y_true[start : start + batch_size] = y
            total += y.shape[0]
            start += batch_size
    print("total examples = %d" % (total))
    return p_pred, y_true


def save_model(model, save_path, ep=None):
    if ep is None:
        full_save_path = save_path + ".pt"
    else:
        full_save_path = save_path + str(ep) + ".pt"
    torch.save(model.state_dict(), full_save_path)


def load_model(model, device, model_path):
    params = torch.load(model_path, map_location=device)
    model.load_state_dict(params, strict=True)
    return model
