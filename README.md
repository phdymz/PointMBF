# PointMBF: A Multi-scale Bidirectional Fusion Network for Unsupervised RGB-D Point Cloud Registration

This repository represents the official implementation of the paper:

[Paper]()

### Instructions
This code has been tested on 
- Python 3.8, PyTorch 1.7.1, CUDA 11.1, GeForce RTX 3090



#### Requirements
To create a virtual environment and install the required dependences please run:
```shell
git clone https://github.com/phdymz/PointMBF.git
conda create --name PointMBF python=3.8
conda activate PointMBF
pip install -r requirements.txt
```

### Make dataset 
#### 3DMatch
You need download the 3DMatch dataset in advance. 

#### ScanNet
You need download the ScanNet dataset in advance. 


### Train on 3DMatch
```shell
python train.py --name RGBD_3DMatch  --RGBD_3D_ROOT 
```

### Train on ScanNet
```shell
python train.py --name ScanNet  --SCANNET_ROOT 
```


### Inference
```shell
python test.py --checkpoint --SCANNET_ROOT
```



