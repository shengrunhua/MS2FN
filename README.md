# Multiscale Segmentation-Guided Fusion Network for Hyperspectral Image Classification

Published in IEEE Transactions on Image Processing, vol. 34, pp. 6152-6167, 2025

https://doi.org/10.1109/TIP.2025.3611146

----------
![image](https://github.com/shengrunhua/MS2FN/blob/main/Overview%20of%20proposed%20MS2FN.png)
## Hongmin Gao, Runhua Sheng, Yuanchao Su, Zhonghao Chen, Shufang Xu and Lianru Gao
----------
# Files Description

## `main.py`

`main.py` is used for training and testing the classification model.

- Handles data loading, model construction, training and test.

- Training settings and hyperparameters can be configured through `main.py`.

## `draw.py`

`draw.py` loads the trained model parameters and visualizes the classification results.

- Automatically reads the specified checkpoint file.

- Generates classification maps based on the model predictions.
----------
# Environment Requirements
The project has been tested with the following package versions:

- Python 3.9+
- PyTorch 2.8.0+cu129
- torchvision 1.8.0
- numpy 2.1.3
- scipy 1.15.3
- scikit-learn 1.6.1
- skimage 0.25.0
- matplotlib 3.10.0
- thop 0.1.1-2209072238
- torchinfo 1.8.0
- tqdm 4.67.1
----------
# Contact Information
Runhua Sheng: shengrunhua@whu.edu.cn
