'''
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research
and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
or recommendations expressed in this material are those of the author(s) and do not necessarily
reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.
'''

# DataLoader construction for different datasets and architectures

import math
import warnings
import numpy as np
from torch.utils.data.dataset import Dataset, Subset

from torchvision import transforms
from torchvision.transforms import InterpolationMode

from wilds import get_dataset
from wilds.common.data_loaders import get_eval_loader, get_train_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSSubset

__all__ = [
    "get_loaders_CLIP",
    "get_loaders_main_224",
    "get_data_for_image_retrieval",
]

_CLIP_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_IMAGE_TENSOR_NORMALIZATION_STD = [0.26862954, 0.26130258, 0.27577711]

BICUBIC = InterpolationMode.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_loaders_CLIP(dataset_name, b_size=64, nw=64, n_px=224, root_dir="/.data", percentage_train=1.0, rank=0, world_size=1):
    """
    Returns data loaders for ID train, ID val, ID test, OOD val, OOD test sets for use in CLIP model
    Optional argument 'percentage_train' sets a random subset of training data to use
    """
    if dataset_name in ["camelyon17", "camelyon17_0"]:
        dataset_name = "camelyon17"
        grouper_name = "hospital"
    elif dataset_name in ["fmow", "fmow_0"]:
        dataset_name = "fmow"
        grouper_name = "region"
    elif dataset_name in ["iwildcam", "iwildcam_0"]:
        dataset_name = "iwildcam"
        grouper_name = "location"
    else:
        raise ValueError("Dataset not supported.")

    dataset = get_dataset(dataset=dataset_name, download=False, root_dir=root_dir)
    grouper = CombinatorialGrouper(dataset, [grouper_name])

    normalization = transforms.Normalize(
        _CLIP_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _CLIP_IMAGE_TENSOR_NORMALIZATION_STD,
    )

    percentage_train = np.min([np.max([percentage_train, 0.0]), 1.0])

    # Get the training set (shuffled)
    train_data = dataset.get_subset(
        "train",
        frac=percentage_train,
        transform=transforms.Compose(
            [
                # data augmentation transforms
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomRotation(2),
                # CLIP transforms: from https://github.com/openai/CLIP/blob/main/clip/clip.py
                transforms.Resize(n_px, interpolation=BICUBIC),
                transforms.CenterCrop(n_px),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                normalization,
            ]
        ),
    )

    # Get the training set
    train_data = DistributedWILDSSubset(train_data, rank, world_size)

    loaders = []
    # Prepare the train loader
    train_loader = get_train_loader(
        "standard", train_data, uniform_over_groups=True, grouper=grouper, num_workers=nw, pin_memory=True, batch_size=b_size
    )
    loaders.append(train_loader)

    # Get eval loaders
    for split in ["id_val", "id_test", "val", "test"]:
        try:
            # Get the eval set (not shuffled)
            eval_data = dataset.get_subset(
                split,
                transform=transforms.Compose(
                    [
                        # CLIP transforms: from https://github.com/openai/CLIP/blob/main/clip/clip.py
                        transforms.Resize(n_px, interpolation=BICUBIC),
                        transforms.CenterCrop(n_px),
                        _convert_image_to_rgb,
                        transforms.ToTensor(),
                        normalization,
                    ]
                ),
            )
            # Prepare the eval loader
            eval_loader = get_eval_loader(
                "standard", eval_data, num_workers=nw, pin_memory=True, batch_size=b_size
            )
        except ValueError as e:
            warnings.warn(f"{str(e)} Replacing loader with empty iterator.")
            eval_loader = []

        loaders.append(eval_loader)

    return loaders, grouper


_DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
_DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD = [0.229, 0.224, 0.225]


def get_loaders_main_224(dataset_name, b_size=64, nw=64, n_px=224, root_dir="/.data", percentage_train=1.0, rank=0, world_size=1):
    """Returns data loaders for ID train, ID val, ID test, OOD val, OOD test sets"""
    if dataset_name in ["camelyon17", "camelyon17_0"]:
        dataset_name = "camelyon17"
        grouper_name = "hospital"
    elif dataset_name in ["fmow", "fmow_0"]:
        dataset_name = "fmow"
        grouper_name = "region"
    elif dataset_name in ["iwildcam", "iwildcam_0"]:
        dataset_name = "iwildcam"
        grouper_name = "location"
    else:
        raise ValueError("Dataset not supported.")

    dataset = get_dataset(dataset=dataset_name, download=False, root_dir=root_dir)
    grouper = CombinatorialGrouper(dataset, [grouper_name])

    default_normalization = transforms.Normalize(
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _DEFAULT_IMAGE_TENSOR_NORMALIZATION_STD,
    )

    percentage_train = np.min([np.max([percentage_train, 0.0]), 1.0])

    # Get the training set (shuffled)
    train_data = dataset.get_subset(
        "train",
        frac=percentage_train,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.RandomRotation(2),
                transforms.Resize(n_px, interpolation=BICUBIC),
                transforms.CenterCrop(n_px),
                transforms.ToTensor(),
                default_normalization,
            ]
        ),
    )

    # Get the training set
    train_data = DistributedWILDSSubset(train_data, rank, world_size)
    
    loaders = []
    # Prepare the train loader
    train_loader = get_train_loader(
        "standard", train_data, uniform_over_groups=True, grouper=grouper, num_workers=nw, pin_memory=True, batch_size=b_size
    )
    loaders.append(train_loader)

    # Get eval loaders
    for split in ["id_val", "id_test", "val", "test"]:
        try:
            # Get the eval set (not shuffled)
            eval_data = dataset.get_subset(
                split,
                transform=transforms.Compose(
                    [
                        transforms.Resize(n_px, interpolation=BICUBIC),
                        transforms.CenterCrop(n_px),
                        transforms.ToTensor(),
                        default_normalization,
                    ]
                ),
            )

            # Prepare the eval loader
            eval_loader = get_eval_loader(
                "standard", eval_data, num_workers=nw, pin_memory=True, batch_size=b_size
            )
        except ValueError as e:
            warnings.warn(f"{str(e)} Replacing loader with empty iterator.")
            eval_loader = []
        
        loaders.append(eval_loader)

    return loaders, grouper


def get_data_for_image_retrieval(
    dataset_name: str = "fmow",
    subset: str = "test", # default subset is the OOD test data
    b_size: int = 64,
    nw: int = 64,
    n_px: int = 224,
    root_dir: str = "./data",
):
    # image normalization for CLIP
    normalization = transforms.Normalize(
        _CLIP_IMAGE_TENSOR_NORMALIZATION_MEAN,
        _CLIP_IMAGE_TENSOR_NORMALIZATION_STD,
    )

    # get the full dataset from wilds
    dataset = get_dataset(dataset=dataset_name, download=False, root_dir=root_dir)

    # raw data
    rawdata = dataset.get_subset(subset)

    # Get the OOD test set (not shuffled)
    data_subset = dataset.get_subset(
        subset,
        transform=transforms.Compose(
            [
                # CLIP transforms: from https://github.com/openai/CLIP/blob/main/clip/clip.py
                transforms.Resize(n_px, interpolation=BICUBIC),
                transforms.CenterCrop(n_px),
                _convert_image_to_rgb,
                transforms.ToTensor(),
                normalization,
            ]
        ),
    )

    # Prepare the test loader
    loader = get_eval_loader(
        "standard",
        data_subset,
        num_workers=nw,
        pin_memory=True,
        batch_size=b_size,
    )

    return rawdata, loader

class DistributedWILDSSubset(WILDSSubset):
    """A subset of a given WILDS dataset, evenly sliced every `world_size` indices.
    Each rank will have a different, non-overlapping subset of the dataset's indices.
    For compatability with DDP, each subset will be padded with duplicate samples so that all
    ranks have equal size subsets. Thus, padding should not be used for testing.

    Args:
        dataset (Dataset): The original WILDS dataset subset.
        rank (int): The rank of the process in distributed training.
        world_size (int): The size of the world in distributed training.
        pad (bool, optional): Pad all subsets with duplicates to equal length. Defaults to True.

    Raises:
        ValueError: If the given `rank` is not in the range of [0, `world_size`).
    """
    def __init__(self, subset: WILDSSubset, rank: int, world_size: int, pad=True):
        if rank >= world_size:
            raise ValueError(f"Rank should be in [0, {world_size}), but got {rank}.")
        rank_indices = list(range(len(subset)))
        if pad:
            padded_len = math.ceil(len(rank_indices) / world_size) * world_size
            rank_indices = rank_indices + rank_indices[0 : padded_len - len(rank_indices)]
        rank_indices = np.asarray(rank_indices[rank::world_size])
        super().__init__(subset.dataset, subset.indices[rank_indices], subset.transform, subset.do_transform_y)

class DistributedSubset(Subset):
    """A subset of a given dataset, evenly sliced every `world_size` indices.
    Each rank will have a different, non-overlapping subset of the dataset's indices.
    For compatability with DDP, each subset will be padded with duplicate samples so that all
    ranks have equal size subsets. Thus, padding should not be used for testing.

    Args:
        dataset (Dataset): The original dataset.
        rank (int): The rank of the process in distributed training.
        world_size (int): The size of the world in distributed training.
        pad (bool, optional): Pad all subsets with duplicates to equal length. Defaults to True.

    Raises:
        ValueError: If the given `rank` is not in the range of [0, `world_size`).
    """
    def __init__(self, dataset: Dataset, rank: int, world_size: int, pad=True) -> None:
        if rank >= world_size:
            raise ValueError(f"Rank should be in [0, {world_size}), but got {rank}.")
        indices = list(range(len(dataset)))
        if pad:
            padded_len = math.ceil(len(indices) / world_size) * world_size
            indices = indices + indices[0 : padded_len - len(indices)]
        indices = indices[rank::world_size]
        super().__init__(dataset, indices)