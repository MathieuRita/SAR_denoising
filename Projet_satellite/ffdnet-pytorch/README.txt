% A PyTorch implementation of FFDNet image denoising.

# ABOUT

* Author    : Matias Tassano <matias.tassano@parisdescartes.fr>
* Copyright : (C) 2018 IPOL Image Processing On Line http://www.ipol.im/
* Licence   : GPL v3+, see GPLv3.txt

# OVERVIEW

This source code provides a PyTorch implementation of FFDNet image denoising,
as in Zhang, Kai, Wangmeng Zuo, and Lei Zhang. "FFDNet: Toward a 
fast and flexible solution for CNN based image denoising." arXiv preprint 
arXiv:1710.04026 (2017).

# USER GUIDE

The code as is runs in Python 3.6 with the following dependencies:
## Dependencies
* [PyTorch v0.3.1](http://pytorch.org/)
* [scikit-image](http://scikit-image.org/)
* [torchvision](https://github.com/pytorch/vision)
* [OpenCV](https://pypi.org/project/opencv-python/)
* [HDF5](http://www.h5py.org/)
* [tensorboardX](https://github.com/lanpa/tensorboard-pytorch)

## Usage

### 1. Testing

If you want to denoise an image using a one of the pretrained models
found under the *models* folder you can execute
```
python test_ffdnet_ipol.py \
	--input input.png \
	--noise_sigma 25 \
	--add_noise True
```
To run the algorithm on CPU instead of GPU:
```
python test_ffdnet_ipol.py \
	--input input.png \
	--noise_sigma 25 \
	--add_noise True \
	--no_gpu
```
**NOTES**
* Models have been trained for values of noise in [0, 75]
* *add_noise* can be set to *False* if the input image is already noisy

### 2. Training

#### Prepare the databases

First, you will need to prepare the dataset composed of patches by executing
*prepare_patches.py* indicating the paths to the directories containing the 
training and validation datasets by passing *--trainset_dir* and
*--valset_dir*, respectively.

Image datasets are not provided with this code, but the following can be downloaded
Training:
[Waterloo Exploration Database](https://ece.uwaterloo.ca/~k29ma/exploration/)
Validation:
[Kodak Lossless True Color Image Suite](http://r0k.us/graphics/kodak/)

**NOTES**
* To prepare a grayscale dataset: ```python prepare_patches.py --gray```
* *--max_number_patches* can be used to set the maximum number of patches
contained in the database

#### Train a model

A model can be trained after having built the training and validation databases 
(i.e. *train_rgb.h5* and *val_rgb.h5* for color denoising, and *train_gray.h5*
and *val_gray.h5* for grayscale denoising).
Only training on GPU is supported.
```
python train.py \
	--batch_size 128 \
	--epochs 80 \
	--noiseIntL 0 75
	--val_noiseL 25
```
**NOTES**
* The training process can be monitored with TensorBoard as logs get saved
in the *log_dir* folder
* By default, models are trained for values of noise in [0, 75] (*--noiseIntL*
flag)
* By default, noise added at validation is set to 20 (*--val_noiseL* flag)
* A previous training can be resumed passing the *--resume_training* flag

# ABOUT THIS FILE

Copyright 2018 IPOL Image Processing On Line http://www.ipol.im/

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved.  This file is offered as-is,
without any warranty.

# ACKNOLEDGMENTS

Some of the code is based on code by Yiqi Yan <yanyiqinwpu@gmail.com>