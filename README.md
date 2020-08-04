# Machine Learning Notes

This repository contains several study notes on Machine Learning.

## Running environment

    conda create --name deep python=python38
    conda install jupyter
    pip install -U \
        tensorflow keras scikit-learn matplotlib \
    conda install pytorch torchvision -c pytorch # check pytorch page for instrucutions
    
## Regression

* [Linear Regression](Regression/Linear-Regression.ipynb)
* [Linear Regression using PyTorch](Regression/regressiion-pytorch.ipynb)
* [Logistic Regression using PyTorch](Regression/logistic-pytorch.ipynb)

## Dimension Reduction

* [Single Value Decomposition](PCA/SVD.ipynb)

## Clustering

## Deep Learning
* [Gradient Descent](DL/gd-general.ipynb)
* [Compute Graph, Back Propagation](DL/autograd.ipynb)
* [Neural Network Concept](DL/ANN-basics.ipynb)
* [2-Layer Gradient Descent](DL/gd_2_layer.ipynb)
* [3-Layer Gradient Descent](DL/gd_3_layer.ipynb)
* [Regularization and dropout](DL/regularization.ipynb)
* [Activation Functions](DL/activation_functions.ipynb)
    * [Look into softmax and cross-entropy in PyTorch context](DL/softmax.ipynb)
  
## Scalable Learning

* [Running Jupyter on Summit](JupyterOnSummit.md)

## Tools

* [numpy](tools/numpy.ipynb), [matplotlib](tools/matplotlib.ipynb)
* [Customized data loader](tools/data_loader.ipynb)
* [Linear Algebra Review](Linear-Algebra-Review.ipynb)
