# PointMBF: A Multi-scale Bidirectional Fusion Network for Unsupervised RGB-D Point Cloud Registration

This repository represents the official implementation of the paper:

[PointMBF: A Multi-scale Bidirectional Fusion Network for Unsupervised RGB-D Point Cloud Registration](https://arxiv.org/pdf/2308.04782.pdf)

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
You need to download the RGB-D version of 3DMatch dataset and ScanNet dataset in advance.
Details can refer to [URR](https://github.com/mbanani/unsupervisedRR/blob/main/docs/datasets.md).

#### 3DMatch
```shell
python create_3dmatch_rgbd_dict.py --data_root 3dmatch_train.pkl train
python create_3dmatch_rgbd_dict.py --data_root 3dmatch_valid.pkl valid
python create_3dmatch_rgbd_dict.py --data_root  3dmatch_test.pkl test
```


#### ScanNet
```shell
python create_scannet_dict.py --data_root scannet_train.pkl train
python create_scannet_dict.py --data_root scannet_valid.pkl valid
python create_scannet_dict.py --data_root scannet_test.pkl test 
```


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

### Pretrained Model
We provide the pre-trained model of PointMBF in [BaiDuyun](https://pan.baidu.com/s/1LO94qfYwEiqwj2hUg8Eojw?_at_=1668346563693), Password: pmbf.


### Acknowledgments
In this project we use (parts of) the official implementations of the followin works: 

- [URR](https://github.com/mbanani/unsupervisedRR) (Trainer and dataset)
- [D3Feat](https://github.com/XuyangBai/D3Feat) (Geometric network)
- [ScanNet](https://github.com/ScanNet/ScanNet) (Make dataset)
- [3DMatch](https://github.com/andyzeng/3dmatch-toolbox) (Make dataset)

 We thank the respective authors for open sourcing their methods. 

### Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@article{yuan2023pointmbf,
  title={PointMBF: A Multi-scale Bidirectional Fusion Network for Unsupervised RGB-D Point Cloud Registration},
  author={Yuan, Mingzhi and Fu, Kexue and Li, Zhihao and Meng, Yucong and Wang, Manning},
  journal={arXiv preprint arXiv:2308.04782},
  year={2023}
}
```



