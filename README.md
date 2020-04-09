# MVA_Remote

![image](https://user-images.githubusercontent.com/35176066/78892619-fc397480-7a69-11ea-8af3-4836aa05880f.png)

The repository is the code of a project done for the master MVA of ENS Paris-Saclay. The aim of the project is to  propose an adaptation of the network FFDNet to denoise SAR images. Therefore, the architecture and main part of the code is taken from __Zhang, Kai and Zuo, Wangmeng and Zhang, Lei FFDNet: Toward a Fast and Flexible Solution for based Image Denoising__ (https://github.com/cszn/FFDNet).

## Short description of the project

In this project we perform transfer learning by using the weights proposed by FFDNet's team (gray level). We then retrain the model with SAR images contaminated with speckle noise and investigate different noise maps. All the information about our study is described in the file `work_report.pdf`

## Experiments

All the experiments are described in the file `denoising_notebook.ipynb`. This is a step-by-step notebook that show how to use our implementation. The searching part has been done using Google Colab. We suggest to use it too.

## Description of the implementation

Then, let's propose a short description of the files that are useful for this project. Since we use the implementation in https://github.com/cszn/FFDNet, most of the files are not used. Here is a description of the files we modified in order to adapt the code to Speckle noise:

- `data/dataset_ffdnet.py`: This file is used for the training procedure. In this file you can modify the type of noise that is added. For this implementation, we remove the addition of gaussian noise and replace it by Speckle multiplication.

- `models/model_plain2.py`: This file is used for the training procedure. It gathers all the functions that are used for the optimization. In particular, it is in this file that we modify the loss in order to test several losses

- `models/network_ffdnet.py`: This file is used both for the training and testing procedures. It implements the architecture of the network. In the default implementation of FFDNet, this file only takes into account uniform noise map (the network receive an integer an noise map). We modified it in order to use also non-uniform noise map: if the network receive an integer, we consider a uniform noise map of the size of the image ; if the network receives a non-uniform noise map of the size of the image, we take it.

- `train_ffdnet.json`: It gathers all the parameters for the training procedure. Feel free to modify it if you want to test different settings.

- `main_train_ffdnet.py`: this is the main file to train the model. File free to update it if you want to test different approaches (eg. train with a noise map that is a Local Means estimate of the input)

- `main_test_ffdnet.py`: the files of ths type are the main files to test the model. `main_test_ffdnet.py` merely apply denoising with a uniform noise map. `main_test_ffdnet_LM.py` apply the Local Means method (see `work_report.pdf`) and `main_test_ffdnet_oracle.py` apply the oracle method (see `work_report.pdf`). Two lines of the code are particularly important: __line 66 where you choose the noise_level__ and __line 68 where you choose the model you want to work with__

- `model_zoo`: In the directory, we put some models (weights) that we use for the project. `ffdnet_gray.pth` are the pre-trained weights provided by FFDNet's team, `loss_L2_view_L1.pth` are the weights of speckle (number of view L=1) training with the L2-loss, `loss_L1_view_L1.pth` are the weights of speckle (number of view L=1) training with the L1-loss, `loss_L1_view_L5.pth` are the weights of speckle (number of view L=5) training with the L2-loss. The last implementation allows to adapt the value of the noise map to find the right balance between smoothness and details preservation (see `work_report.pdf`)

- `results`: once the denoising is performed, the reuslts are stored in this directory.





