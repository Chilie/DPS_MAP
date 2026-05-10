# DPS_MAP
Code for SIIMS paper "Efficient Diffusion Posterior Sampling for Noisy Inverse Problems"

[Efficient Diffusion Posterior Sampling for Noisy Inverse Problems](https://arxiv.org/abs/2503.10237)



## Prerequisites
- python 3.8

- pytorch 1.11.0

- CUDA 11.3.1 (other version is also fine)


## Getting started 

### Step 1: Set environment

Create a new environment and install dependencies

```
conda create -n DPS python=3.8

conda activate DPS

pip install -r requirements.txt

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

If you fail to install mpi4py using the pip install, you can try conda as follows
```
conda install mpi4py
```

In addition, you might need 

```
pip install scikit-image
pip install blobfile
```

Finally, make sure the code is run on GPU, though it can run on cpu as well.  


### Step 2:  Download pretrained checkpoint
For FFHQ and Imagenet datasets, download the pretrained checkpoints "ffhq_10m.pt" and "imagenet256.pt"  from  [checkpoint_link](https://drive.google.com/drive/folders/1jElnRoFv7b31fG0v6pTSQkelbSX3xGZh), and paste it to ./models/


### Step 3:  Prepare the dataset
You need to write your data directory at data.root. Default is ./data/ffhq256_10 which contains three sample images from FFHQ validation set. We also provide other demo data samples in ./data/ used in our paper.

### Step 4: Perform Posterior Sampling for different tasks for FFHQ

```
bash comp_DDIM_ours_test.sh
```

For ImageNet dataset

```
bash comp_DDIM_ours_test_img.sh
```

### Step 5: Perform Posterior Sampling for SVD-free linear operator case (motion deblurring)

```
bash comp_Gaussian_motion_deblur.sh
```

### Step 5: Perform Posterior Sampling for Poisson noise

```
bash comp_DDIM_Poisson.sh
```

## Citation 
If you find the code useful for your research, please consider citing as 

```
@article{meng2022diffusion,
  title={Diffusion Model Based Posterior Samplng for Noisy Linear Inverse Problems},
  author={Meng, Xiangming and Kabashima, Yoshiyuki},
  journal={arXiv preprint arXiv:2211.12343},
  year={2022}
}

```


## References

This repo is developed based on  [DPS code](https://github.com/DPS2022/diffusion-posterior-sampling) and  [DDRM code](https://github.com/bahjat-kawar/ddrm). Please also consider citing them if you use this repo. 
```

@inproceedings{kawar2022denoising,
    title={Denoising Diffusion Restoration Models},
    author={Bahjat Kawar and Michael Elad and Stefano Ermon and Jiaming Song},
    booktitle={Advances in Neural Information Processing Systems},
    year={2022}
}

@article{li2025efficient,
  title={Efficient diffusion posterior sampling for noisy inverse problems},
  author={Li, Ji and Wang, Chao},
  journal={SIAM Journal on Imaging Sciences},
  volume={18},
  number={2},
  pages={1468--1492},
  year={2025},
  publisher={SIAM}
}

@article{chung2022diffusion,
  title={Diffusion Posterior Sampling for General Noisy Inverse Problems},
  author={Chung, Hyungjin and Kim, Jeongsol and Mccann, Michael T and Klasky, Marc L and Ye, Jong Chul},
  journal={arXiv preprint arXiv:2209.14687},
  year={2022}
}

```
