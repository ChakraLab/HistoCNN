# HistoCNN

Implementation of a CNN for histology image classification tasks. 

## Table of Contents

1. [Installation](#installation)
2. [Requirements](#Requirements)
3. [Usage](#usage)
4. [References](#references)

## Installation
HistoCNN has been developed using Python 3.7. Clone the repository:

```sh
git clone https://github.com/ChakraLab/HistoCNN
```

## Requirements
It depends on a few Python packages, including:
* torch (1.9.0+cu102)
* google_drive_downloader - [GoogleDriveDownloader](https://pypi.org/project/googledrivedownloader/)

## Usage
The code can be simply run with:

```sh
python resnet_run.py
python densenet_run.py
python squeezenet_run.py
python resnext50_run.py
```

The optional arguments are:

```bash
usage: python3 resnet_run.py [--gpus GPUS] [--nodes NODES]

Arguments:
  --gpu GPU             gpu (-1 for no GPU, 0 otherwise)
  --gpus                number of GPUs
  --nodes               number of nodes
```

## A. Duct segmentation

## B. Duct ROI classification

## C. WSI-level and ROI-level inference

## References
1. Parvatikar, Akash, et al. "Modeling Histological Patterns for Differential Diagnosis of Atypical Breast Lesions." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2020.
2. Parvatikar, Akash, et al. "Prototypical Models for Classifying High-Risk Atypical Breast Lesions." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2021.
