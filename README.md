# SSIT2023
This repository is the implementation of SSIT: Single-Stream Image-to-Image Translation [Paper](https://www.computer.org/csdl/journal/oj/2024/01/10694773/20wCWTplz7W)
We provide the implement of our SSIT with Pyotrch. 
# Requirements
- Python 3.x
- CPU or NVIDIA GPU
- CUDA and cuDNN (if you activate GPU for training and testing)
# Requirements (Python Libraries)
- pytorch
- munch
# Training
If you train with your custom dataset, please place the dataset under the directory @Dataset. And start train with your code:\n
```
python trainer.py -src "custom dataset" -train_dirA "trainA" -train_dirB "trainB" -test_dirA "testA" -testB "testB"
```
# Citation
If you use our code for your study, please cite: 
```bibtex
@article{article,
author = {Oh, R. and Gonsalves, Tad},
year = {2024},
month = {01},
pages = {624-635},
title = {Photogenic Guided Image-to-Image Translation With Single Encoder},
volume = {5},
journal = {IEEE Open Journal of the Computer Society},
doi = {10.1109/OJCS.2024.3462477}
}
