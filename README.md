# UMA-Net: Adaptive Ensemble Loss and Multi-Scale Attention for Breast Ultrasound Segmentation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.8+](https://img.shields.io/badge/TensorFlow-2.8+-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)

This repository contains the official implementation of the paper:

**Adaptive Ensemble Loss and Multi-Scale Attention in Breast Ultrasound Segmentation with UMA-Net**  
*Mohsin Farooq Dar, Avatharam Ganivada*  
Medical & Biological Engineering & Computing (2025)  
[DOI: 10.1007/s11517-025-03301-5](https://doi.org/10.1007/s11517-025-03301-5)

## Overview

UMA-Net (U-Net with Multi-scale Attention) is a novel deep learning architecture designed for accurate breast ultrasound image segmentation. The model incorporates:

- Multi-scale attention mechanisms for better feature extraction
- Adaptive ensemble loss function for improved training stability
- Advanced skip connections with attention gates
- Atrous Convolution for multi-scale feature fusion

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MohsinFurkh/UMA-Net-with-Multi-Scale-Attention.git
cd UMA-Net-with-Multi-Scale-Attention
```

2. Create a conda environment (recommended):
```bash
conda create -n umanet python=3.8
conda activate umanet
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset Preparation

1. Download the dataset and organize it in the following structure:
```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

2. Update the configuration file `configs/train_config.yaml` with your dataset paths.

## Training

To train the model with default settings:

```bash
python train.py --config configs/train_config.yaml
```

## Evaluation

To evaluate the trained model:

```bash
python evaluate.py --model_path models/uma_net.h5 --test_data_path data/test/
```

## Results

Our model achieves state-of-the-art performance on breast ultrasound segmentation:

| Metric       | Value  |
|--------------|--------|
| Dice Score   | 0.921  |
| IoU          | 0.854  |
| Accuracy     | 0.971  |
| Sensitivity  | 0.927  |
| Specificity  | 0.983  |

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{dar2025adaptive,
  title={Adaptive ensemble loss and multi-scale attention in breast ultrasound segmentation with UMA-Net},
  author={Dar, Mohsin Farooq and Ganivada, Avatharam},
  journal={Medical \& Biological Engineering \& Computing},
  volume={63},
  number={6},
  pages={1697--1713},
  year={2025},
  publisher={Springer}
}
```


## Contact

For any questions or suggestions, please open an issue or contact [Mohsin Furkh Dar](mailto:20mcpc02@uohyd.ac.in).
