DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Under Secretary of Defense for Research and Engineering.

© 2023 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.

-----------------------------------------------

# Robust Fine-Tuning of Vision-Language Models for Domain Generalization
This repository contains code for the IEEE 2023 paper Robust Fine-Tuning of Vision-Language Models for Domain Generalization, by Kevin Vogt-Lowell, Noah Lee, Theodoros Tsiligkaridis, and Marc Vaillant.

## Requirements
Experiments were run on an Anaconda 2022a environment with CUDA 11.3. Distributed training experiments used NCCL-2.9.8.
- Python 3.8.13
- PyTorch 1.11.0
- NumPy 1.20.3
- Pandas 1.5.3
- SciPy 1.7.3
- Scikit-learn 1.0.2
- Pickle 0.7.5
- wilds 2.0.0
- [CLIP](https://github.com/openai/CLIP) 1.0
- [Open-CLIP-Torch](https://github.com/mlfoundations/open_clip) 2.7.0

To install, read the necessary packages from requirements.txt using pip. Then, manually install the torch-scatter and torch-geometric packages, which are needed by wilds.

```console
$ pip install -r requirements.txt
$ pip install torch_geometric torch_scatter
```

For CLIP and Open-CLIP-Torch, please follow the installation instructions found in their linked GitHub repos.

## Datasets
WILDS Datasets download: https://wilds.stanford.edu/datasets/. Information about datasets and their associated splits can be found here as well.

The Camelyon-17, FMoW, and iWildCam datasets are supported in this code base.

To use one of these datasets within the scripts discussed in the [User Guide](#user-guide), provide one of the following options to the  `--dataset` parameter, if available. Options with a "_0" suffix indicate that the group annotations from WILDS metadata should be leveraged to create group-aware data loaders.
- camelyon17
- camelyon17_0 (grouped by "hospital")
- fmow
- fmow_0 (grouped by "region")
- iwildcam
- iwildcam_0 (grouped by "location")

## Supported Model Architectures
- ResNet50 From Scratch (resnet50)
- ResNet50 with ImageNet Pretrained Weights + Linear Probing (resnet50_pretrained_linearprob)
- ResNet50 with ImageNet Pretrained Weights + Fine-Tuning (resnet50_pretrained_finetune)
- ResNet50 with CLIP Pretrained Weights + Linear Probing (resnet50_clip_linearprob)
- ResNet50 with CLIP Pretrained Weights + Fine-Tuning (resnet50_clip_finetune)
- ViT-B-32 From Scratch (vitb32)
- ViT-B-32 with ImageNet Pretrained Weights + Linear Probing (vitb32_pretrained_linearprob)
- ViT-B-32 with ImageNet Pretrained Weights + Fine-Tuning (vitb32_pretrained_finetune)
- ViT-B-32 with CLIP Pretrained Weights + Linear Probing (vitb32_clip_linearprob)
- ViT-B-32 with CLIP Pretrained Weights + Fine-Tuning (vitb32_clip_finetune)

To use these architectures within the scripts discussed in the [User Guide](#user-guide), provide the associated names in parentheses as input to the `--arch` parameter where required.

> :warning: **Note**: Any model can be tested zero-shot by simply instantiating the model architecture (and not loading any trained checkpoint weights). For some architectures that will include either ImageNet or CLIP pretrained weights.

## User Guide
### Training
Models can be trained using the `train.py` script. The basic syntax is:

```console
$ python train.py --dataset <dataset name> --arch <architecture name> [--options]
```

The training script has several configurable options:
| Optional Parameter       | Type    | Description | Default |
| ------------------------ | ------- | ------------| ------- |
| `--batch-size`           | `int`   | The number of samples in each batch | 128 |
| `--train-alg`            | `str`   | The training algorithm to use. Currently only "ERM" is supported | "ERM" |
| `--nepochs`              | `int`   | The number of training epochs to execute (including swa epochs) | 30 |
| `--swa-epochs`           | `int`   | The number of epochs before the end of training over which SWA will occur. Set to 0 to not use SWA. | 10 |
| `--lr`                   | `float` | Learning rate to be used by the optimizer | 1e-5 |
| `--swa-lr`               | `float` | Learning rate to be used during SWA | 0.0001 |
| `--perc-train`           | `float` | Percentage of training data to use during training | 1.0 |
| `--save-freq-ep`         | `int`   | How frequently the model should be evaluated and saved | 1 |
| `--nruns`                | `int`   | How many complete training runs should be executed in order to obtain an averaged measure of model performance | 3 |
| `--num-workers`          | `int`   | Number of workers to use for loading data | 64 |
| `--model-dir-path`       | `str`   | Absolute or relative path to which the model will be saved | "./models" |
| `--log-dir-path`         | `str`   | Absolute or relative path to which validation accuracy logs will be written | "./logs" |
| `--data-dir`             | `str`   | Absolute or relative path pointing to the directory where the WILDS datasets can be found | "./data" |
| `--optimizer`            | `str`   | PyTorch optimizer to use, options are "Adam" or "SGD" | "Adam" |
| `--sync-bn`              | --      | If this option is included when launching distributed training, synchronized batch norm will be activated | -- |
| `--progress`             | --      | If this option is included, then a progress bar will be displayed when iterating over training or validation batches | -- |

#### Distributed Training
If you have access to multiple-GPU compute nodes managed by SLURM, you can speed up training time by submitting distributed training jobs using the following syntax:

```console
$ sbatch -o launch_training_distributed.sh.log-%j scripts/launch_training_distributed.sh --dataset <dataset name> --arch <architecture name> [--options]
```

Optional arguments are the same as those listed for non-distributed training. Please ensure that the SLURM directives found in scripts/launch_training_distributed.sh are appropriate for your compute environment.

> :warning: **Note**: Use of 32 or more GPUs for distributed training can result in degradation of model accuracy. We recommend 16 or fewer GPUs for these experiments.

### Testing
After a model is trained, it can be evaluated on each of five splits of the dataset (train, ID validation, ID test, OOD validation, OOD test). This is done via the `test.py` script. The syntax is:

```console
$ python test.py --dataset <dataset name> --arch <architecture name> [--options]
```

Below is a table describing each of the configurable options for the test script.
| Optional Parameter   | Type  | Description                                                                                                          | Default |
| -------------------- | ----- | -------------------------------------------------------------------------------------------------------------------- | ------- |
| `--outfile`          | `str` | The file to which results will be saved                                                                              | "./results/results.csv" |
| `--batch-size`       | `int` | The number of samples in each batch                                                                                  | 128 |
| `--nruns`            | `int` | How many complete training runs should be executed to build up statistics                                            | 3 |
| `--num-workers`      | `int` | Number of workers to use for loading data                                                                            | 64 |
| `--model-dir-path`   | `str` | Absolute or relative path to the directory in which the model is saved                                               | "./models" |
| `--data-dir`         | `str` | Absolute or relative path pointing to the directory where the WILDS datasets can be found                            | "./data" |
| `--swa`              | --    | If this option is included, testing will be conducted using SWA checkpoints instead of best non-SWA checkpoints      | -- |
| `--progress`         | --    | If this option is included, then a progress bar will be displayed when iterating over training or validation batches | -- |


### Image Retrieval
#### Inspection
The `image_retrieval.py` script can be used to retrieve images from the dataset that are most similar to a given text prompt. Note: the only architectures supported for text-based image retrieval are CLIP-based models with either default weights or weights loaded from a training checkpoint file. The syntax for calling `image_retrieval.py` is:

```console
$ python image_retrieval.py --dataset <dataset name> --prompt <text prompt> --outfile <output image file> [--options]
```

The optional parameters are:
| Optional Parameter   | Type  | Description | Default |
| -------------------- | ----- | ------------| ------- |
| `--default-weights`  | --    | Raise this flag to instantiate a CLIP model using OpenAI's pretrained weights | -- |
| `--model-checkpoint` | `str` | Path to the training checkpoint file to use for loading a model | -- |
| `--nimages`          | `int` | Number of images to retrieve from the dataset | 5 |
| `--data-dir`         | `str` | Absolute or relative path pointing to the directory where the WILDS datasets can be found | "./data" |
| `--progress`         | --    | If this option is included then a progress bar will be displayed when iterating over training or validation batches | -- |

#### Retrieval Metric Evaluation
In addition to inspecting the images retrieved from the dataset based on a text prompt, the image retrieval metric *"recall @ k"* can be evaluated using the `test_retrieval.py` script. The *recall @ k* metric can be interpreted as the fraction of the top-k retrieved images that belong to the class used for the query, averaged over each of the classes in the dataset. This works for both the text-based CLIP models as well as the supervised models. When evaluating a CLIP-based model, auto-generated text prompts for each class are generated and passed into the model. The syntax for calling the script is similar to those prior:

```console
$ python test_retrieval.py --dataset <dataset name> --arch <architecture name> --outfile <results filename> [--options]
```

The optional parameter `--k` can be used to specify the number of images that should be retrieved when computing *recall @ k*; the default is k = 20. The other optional parameters are the same as for training and testing models.

### Limited Data Experiments

> :warning: **Note**: The following instructions for replicating the limited data experiments demonstrated in our paper assume access to compute clusters managed by SLURM. If you have access to such a compute environment, please ensure the SLURM directives within the shell scripts provided in the `scripts` directory are appropriate given your available resources. If you do not have access to such a compute environment, the limited data experiments will have to be run individually using `train.py` as detailed in the [User Guide](#user-guide)

To replicate the experimental results demonstrated in the paper, use the `submit_lim_data_train_jobs.py` and `submit_lim_data_test_jobs.py` scripts with default values to launch parallel training and then testing jobs according to the experimental details outlined within `lim_data_config.json`. `lim_data_config.json` defines all architectures, training data percentages, and learning rate combinations to be evaluated, and it is pre-populated with the values investigated for the paper. These values can be edited to explore different experimental permutations.

The basic syntax for calling `submit_lim_data_train_jobs.py` is:

```console
$ python submit_lim_data_train_jobs.py --dataset <dataset name> [--options]
```

The optional parameters for launching the limited data training jobs are:
| Optional Parameter       | Type    | Description | Default |
| ------------------------ | ------- | ------------| ------- |
| `--data-dir`             | `str`   | Absolute or relative path pointing to the directory where the WILDS datasets can be found | "./data" |
| `--nepochs`              | `int`   | The number of training epochs to execute (including swa epochs) | 20 |
| `--swa-epochs`           | `int`   | The number of epochs before the end of training over which SWA will occur. Set to 0 to not use SWA. | 0 |
| `--model-dir-path`       | `str`   | Absolute or relative path to the directory that will hold all experiment model subdirectories | "./lim_data_models" |
| `--log-dir-path`         | `str`   | Absolute or relative path to the directory that will hold all experiment log subdirectories | "./lim_data_logs" |
| `--distributed-training` | --      | If this option is included, all created training jobs will be launched using distributed training | -- |
| `--sync-bn`              | --      | If this option is included when launching distributed training, synchronized batch norm will be activated | -- |


The syntax for calling `submit_lim_data_test_jobs.py` is very similar, using the following command syntax and optional parameters:

```console
$ python submit_lim_data_test_jobs.py --dataset <dataset name> [--options]
```

| Optional Parameter       | Type    | Description | Default |
| ------------------------ | ------- | ------------| ------- |
| `--data-dir`             | `str`   | Absolute or relative path pointing to the directory where the WILDS datasets can be found | "./data" |
| `--model-dir-path`       | `str`   | Absolute or relative path to the directory that holds all experiment model subdirectories | "./lim_data_models" |
| `--outfile-dir-path`     | `str`   | Absolute or relative path to the directory in which to store all test result outfiles | "./results" |
| `--swa`                  | --      | If this option is included, testing will be conducted using SWA checkpoints instead of best non-SWA checkpoints | -- |

To also obtain the SWA results shown in the paper, edit the architectures field in `lim_data_config.json` to contain only `vitb32_clip_finetune`, set the optional parameter `swa-epochs` to 10 when launching `submit_limited_data_train_jobs.py`, and raise the `--swa` flag when launching `submit_lim_test_jobs.py`.

#### Visualization

To visualize the results of the limited data experiments, use the `visualize_lim_data_results.py` script. The file will generate a line plot and a .json file demonstrating the experimental results.

The syntax and optional parameters used for calling `visualize_lim_data_results.py` are:

```console
$ python visualize_lim_data_results.py --dataset <dataset name> [--options]
```

| Optional Parameter       | Type    | Description | Default |
| ------------------------ | ------- | ------------| ------- |
| `--results-dir`          | `str`   | Absolute or relative path pointing to the directory containing all non-SWA experiment results | "./results" |
| `--swa-results-dir`      | `str`   | Absolute or relative path to the directory containing SWA results for fine-tuned ViT-B-32 with CLIP pretrained weights | -- |
| `--error-bars`           | --      | If this option is included, error bars will be included in the line plot | -- |

During visualization, the ID and OOD test results yielded using each model's best learning rate are plotted against the training data percentages listed in `lim_data_config.json`. Each architecture in the config file will have its own line within the plot, and the SWA results for CLIP ViT-B/32 Fine-tuned (E2E) can be included if the appropriate directory is provided using the `--swa-results-dir` parameter.

> :warning: **Note**: Currently, the visualization script only supports plotting of the SWA results for the ViT-B-32 CLIP model fine-tuned end-to-end.
