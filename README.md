# encodec-pytorch
This is an unofficial implementation of the paper [High Fidelity Neural Audio Compression](https://arxiv.org/pdf/2210.13438.pdf) in PyTorch.



## Introduction
This repository is based on [encodec](https://github.com/facebookresearch/encodec) and [HiFiGAN](https://github.com/jik876/hifi-gan)).


TODO:
- [ ] support the 48khz model.
- [ ] support wandb or tensorboard to monitor the training process.

## Enviroments
The code is tested on the following environment:
- Python 3.9
- PyTorch 2.0.0 (You can try other versions, but I can't guarantee that it will work. Because torch have changed some api default value (eg: stft). )
- GeForce RTX 3090 x 4

In order to you can run the code, you can install the environment by the help of requirements.txt.

## Usage
### Training
#### 1. Prepare dataset
I use the vctk as the train datasets like HiFiGAN and you can prepare your own dataset.
Also you can use `ln -s` to link the dataset to the `datasets` folder.

#### 2. Train
You can use the following command to train the model using multi gpu:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py 

#### 4. Acknowledgement
Thanks to the following repositories:
- [encodec](https://github.com/facebookresearch/encodec)
- [HiFiGAN](https://github.com/jik876/hifi-gan))
- [melgan-neurips](https://github.com/descriptinc/melgan-neurips): audio_to_mel.py

