# Revisiting Catastrophic Forgetting in Continual Learning via Functional Similarity

This work is under review.


## Requirements 
This project uses the preset environment in Google Colab.


![](https://img.shields.io/badge/python-3.9.16-green.svg)

![](https://img.shields.io/badge/torch-1.13.1-blue.svg)
![](https://img.shields.io/badge/torchvision-0.14.1-blue.svg)
![](https://img.shields.io/badge/scikit--learn-1.2.1-blue.svg)

## Usage
Please open the example file (.ipynb) in colab and copy the preparation files([func_simMNIST](func_simMNIST) or [func_simCIFAR10](func_simCIFAR10)) into `/home`(in your _Colab_ virtual host) as required by the notes in the example file.

Note: Please do not open the example file derictly in your own _Google Drive_. As the data exchange between _Colab_ virtual host and your own _Google Drive_, it will make the data generation module run extremely slow .

## Getting simple
We conduct experiments on two datasets, SplitMNIST and SplitCIFAR10, respectively. 

SplitMNIST experiment: 
<a target="_blank" href="https://colab.research.google.com/github/wang-xulong/Func_sim/blob/main/MNIST_sim_acc_fgt.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>    


SplitCIFAR10 experiment: 
<a target="_blank" href="https://colab.research.google.com/github/wang-xulong/Func_sim/blob/main/CIFAR10_sim_acc_fgt.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>  


## Acknowledgment
Our datasets are based on [the continual learning community project: avalanche](https://avalanche.continualai.org/)
