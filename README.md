# ODGNet
Orthogonal Dictionary Guided Shape Completion Network for Point Cloud

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/orthogonal-dictionary-guided-shape-completion/point-cloud-completion-on-shapenet)](https://paperswithcode.com/sota/point-cloud-completion-on-shapenet?p=orthogonal-dictionary-guided-shape-completion)

# Environment

Pytorch 1.12.0 with Nvidia GPUs

Setup Libs

Install pointnet2_ops_lib and Chamfer Distance in the extension Folder:

python3 setup.py install


# Dataset and Python Libs Requirements
Please Check PoinTr (https://github.com/yuxumin/PoinTr/tree/master)

# Pretrained Models

[[Google Drive](https://drive.google.com/file/d/1rXf4ul34QOW_7JBulYaz_pOI4LHUCMCe/view?usp=sharing)]

# Training/Testing
Please check the bash files, e.g., "sh utrain.sh" for the PCN dataset.

Check and modify the "test.sh" for testing.

# Citation
If our method and results are useful for your research, please consider citing:

```
@inproceedings{ODGNet,
    title={Orthogonal Dictionary Guided Shape Completion Network for Point Cloud},
    author={Cai, Pingping and Scott, Deja and Li, Xiaoguang and Wang, Song},
    booktitle={AAAI},
    year={2024},
}
```

# Acknowledgement
Some codes are borrowed from PoinTr and PSCU [https://github.com/corecai163/PSCU]
