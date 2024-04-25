<h1 align="center"> Pytorch ReID </h1>
<h2 align="center"> Strong, Small, Friendly </h2>

### Training 
- Running the code on Google Colab with Free GPU. Check [Here](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/colab) (Thanks to @ronghao233)
- [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) (10x Large Synthetic Dataset from Market **CVPR 2019 Oral**)
- [Swin Transformer](https://github.com/microsoft/Swin-Transformer) / [EfficientNet](https://github.com/lukemelas/EfficientNet-PyTorch) / [HRNet](https://github.com/HRNet)
- ResNet/ResNet-ibn/DenseNet
- Circle Loss, Triplet Loss, Contrastive Loss, Sphere Loss, Lifted Loss, Arcface, Cosface  and Instance Loss
- Float16 to save GPU memory based on [apex](https://github.com/NVIDIA/apex)
- Part-based Convolutional Baseline(PCB)
- Random Erasing
- Linear Warm-up 
- torch.compile (faster training)
- DDP (Multiple GPUs)

### Testing
- TensorRT 
- Pytorch JIT 
- Fuse Conv and BN layer into one Conv layer
- Multiple Query Evaluation
- Re-Ranking (CPU & [GPU Version](https://github.com/layumi/Person_reID_baseline_pytorch/tree/master/GPU-Re-Ranking))
- Visualize Training Curves
- Visualize Ranking Result
- [Visualize Heatmap](https://github.com/layumi/Person_reID_baseline_pytorch/blob/dev/visual_heatmap.py)


Here we provide hyperparameters and architectures, that were used to generate the result. 
Some of them (i.e. learning rate) are far from optimal. Do not hesitate to change them and see the effect. 

P.S. With similar structure, we arrived **Rank@1=87.74% mAP=69.46%** with [Matconvnet](http://www.vlfeat.org/matconvnet/). (batchsize=8, dropout=0.75) 
You may refer to [Here](https://github.com/layumi/Person_reID_baseline_matconvnet).
Different framework need to be tuned in a slightly different way.


   
## Trained Model
I re-trained several models, and the results may be different with the original one. Just for a quick reference, you may directly use these models. 
The download link is [Here](https://drive.google.com/open?id=1XVEYb0TN2SbBYOqf8SzazfYZlpH9CxyE).

|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
| [EfficientNet-b4] | 85.78% | 66.80% |  `python train.py --use_efficient --name eff; python test.py --name eff` |
| [ResNet-50 + adv defense] | 87.77% | 69.83% |  `python train.py  --name adv0.1_40_w10_all --adv 0.1 --aiter 40 --warm 10 --train_all; python test.py  --name adv0.1_40_w10_all` |
| [ConvNeXt] | 88.98% | 71.35% |  `python train.py --use_convnext --name convnext; python test.py --name convnext` |
| [ResNet-50 (fp16)] | 88.03% | 71.40% | `python train.py --name fp16 --fp16 --train_all` |
| [ResNet-50] | 88.84% | 71.59% |  `python train.py --train_all` |
| [ResNet-50-ibn] | 89.13% | 73.40% | `python train.py --train_all --name res-ibn --ibn` |
| [DenseNet-121] | 90.17% | 74.02% | `python train.py --name ft_net_dense --use_dense --train_all` |
| [DenseNet-121 (Circle)] | 91.00% | 76.54% | `python train.py --name ft_net_dense_circle_w5 --circle --use_dense --train_all --warm_epoch 5` |
| [HRNet-18] | 90.83% | 76.65% |  `python train.py --use_hr --name hr18; python test.py --name hr18` |
| [PCB] | 92.64% | 77.47% | `python train.py --name PCB --PCB --train_all --lr 0.02` |
| [PCB + DG] | 92.70% | 78.31% | `python train.py --name PCB_DG --PCB --train_all --lr 0.02 --DG; python test.py --name PCB_DG` |
| [ResNet-50 (all tricks)] | 91.83% | 78.32% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5` |
| [ResNet-50 (all tricks+Circle)] | 92.13% | 79.84% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5_circle  --circle` |
| [ResNet-50 (all tricks+Circle+DG)] | 92.13% | 80.13% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name warm5_s1_b8_lr2_p0.5_circle_DG --circle --DG; python test.py --name warm5_s1_b8_lr2_p0.5_circle_DG` |
| [DenseNet-121 (all tricks+Circle)] | 92.61% | 80.24% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 8 --lr 0.02 --name dense_warm5_s1_b8_lr2_p0.5_circle --circle --use_dense; python test.py --name  dense_warm5_s1_b8_lr2_p0.5_circle` |
| [HRNet-18 (all tricks+Circle+DG)]| 92.19% | 81.00% | `python train.py --use_hr --name  hr18_p0.5_circle_w5_b16_lr0.01_DG --lr 0.01 --batch 16 --DG --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name  hr18_p0.5_circle_w5_b16_lr0.01_DG` |
| [Swin] (224x224) | 92.75% | 79.70% | `python train.py --use_swin --name swin; python test.py --name swin`|
| [SwinV2 (all tricks+Circle 256x128)] | 92.93% | 82.99% | `python train.py --use_swinv2 --name swinv2_p0.5_circle_w5_b16_lr0.03  --lr 0.03 --batch 16 --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name   swinv2_p0.5_circle_w5_b16_lr0.03 --batch 32`|
| [Swin (all tricks+Circle 224x224)] | 94.12% | 84.39% | `python train.py --use_swin --name swin_p0.5_circle_w5  --erasing_p 0.5 --circle --warm_epoch 5;  python test.py --name swin_p0.5_circle_w5`|
| [Swin (all tricks+Circle+b16 224x224)] | 94.00% | 85.21% | `python train.py --use_swin --name swin_p0.5_circle_w5_b16_lr0.01 --lr 0.01 --batch 16  --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swin_p0.5_circle_w5_b16_lr0.01`|
| [Swin (all tricks+Circle+b16+DG 224x224)] | 94.00% | 85.36% | `python train.py --use_swin --name swin_p0.5_circle_w5_b16_lr0.01_DG --lr 0.01 --batch 16 --DG --erasing_p 0.5 --circle --warm_epoch 5; python test.py --name swin_p0.5_circle_w5_b16_lr0.01_DG`|

* More training iterations may lead to better results. 
* Swin costs more GPU memory (11G GPU is needed) to run. 
* The hyper-parameter of [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) `--DG` is not tuned. Better hyper-parameter may lead to better results.

### Different Losses 
   
I do not optimize the hyper-parameters. You are free to tune them for better performance.

|Methods | Rank@1 | mAP| Reference|
| -------- | ----- | ---- | ---- |
| CE | 92.01% | 79.31% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_100 --total 100 ; python test.py  --name  warm5_s1_b32_lr8_p0.5_100`|
| CE + Sphere [[Paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf) | 92.01% | 79.39% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_sphere100 --sphere --total 100; python test.py --name warm5_s1_b32_lr8_p0.5_sphere100` |
| CE + Triplet [[Paper]](https://arxiv.org/pdf/1703.07737) | 92.40%	| 79.71% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_triplet100 --triplet --total 100; python test.py  --name warm5_s1_b32_lr8_p0.5_triplet100` |
| CE + Lifted [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Song_Deep_Metric_Learning_CVPR_2016_paper.pdf)|  91.78% | 79.77% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_lifted100 --lifted --total 100; python test.py --name warm5_s1_b32_lr8_p0.5_lifted100` |
| CE + Instance [[Paper]](https://zdzheng.xyz/files/TOMM20.pdf) | 92.73% | 81.11% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_instance100_gamma64 --instance --ins_gamma 64 --total 100 ; python test.py  --name  warm5_s1_b32_lr8_p0.5_instance100_gamma64`|
| CE + Contrast [[Paper]](https://zdzheng.xyz/files/TOMM18.pdf) | 92.28% | 81.42% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_contrast100 --contrast  --total 100; python test.py  --name warm5_s1_b32_lr8_p0.5_contrast100`|
| CE + Circle [[Paper]](https://arxiv.org/abs/2002.10857) | 92.46% | 81.70% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_circle100 --circle --total 100 ; python test.py  --name  warm5_s1_b32_lr8_p0.5_circle100` |
| CE + Contrast + Sphere | 92.79% | 82.02% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 32 --lr 0.08 --name warm5_s1_b32_lr8_p0.5_cs100 --contrast --sphere --total 100; python test.py --name warm5_s1_b32_lr8_p0.5_cs100`|
| CE + Contrast + Triplet (Long) | 92.61% | 82.01% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 24 --lr 0.062 --name warm5_s1_b24_lr6.2_p0.5_contrast_triplet_133 --contrast --triplet --total 133 ; python test.py  --name  warm5_s1_b24_lr6.2_p0.5_contrast_triplet_133` |
| CE + Contrast + Circle (Long) | 92.19% | 82.07% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 24 --lr 0.08 --name warm5_s1_b24_lr8_p0.5_contrast_circle133 --contrast --circle --total 133 ; python test.py  --name  warm5_s1_b24_lr8_p0.5_contrast_circle133` |
| CE + Contrast + Sphere (Long) | 92.84% | 82.37% | `python train.py --warm_epoch 5 --stride 1 --erasing_p 0.5 --batchsize 24 --lr 0.06 --name warm5_s1_b24_lr6_p0.5_contrast_sphere133 --contrast --sphere --total 133 ; python test.py  --name  warm5_s1_b24_lr6_p0.5_contrast_sphere133` |


### Model Structure
You may learn more from `model.py`. 
We add one linear layer(bottleneck), one batchnorm layer and relu.

## Prerequisites

- Python 3.6+
- GPU Memory >= 6G
- Numpy
- Pytorch 0.3+
- timm `pip install timm` for Swin-Transformer with Pytorch >1.7.0
- pretrainedmodels via `pip install pretrainedmodels`
- [Optional] apex (for float16) 
- [Optional] [pretrainedmodels](https://github.com/Cadene/pretrained-models.pytorch)

**(Some reports found that updating numpy can arrive the right accuracy. If you only get 50~80 Top1 Accuracy, just try it.)**
We have successfully run the code based on numpy 1.12.1 and 1.13.1 .

## Getting started
### Installation
- Install Pytorch from http://pytorch.org/
- Install Torchvision from the source
```
git clone https://github.com/pytorch/vision
cd vision
python setup.py install
```
- [Optional] You may skip it. Install apex from the source
```
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
Because pytorch and torchvision are ongoing projects.

Here we noted that our code is tested based on Pytorch 0.3.0/0.4.0/0.5.0/1.0.0 and Torchvision 0.2.0/0.2.1 .

### Dataset & Preparation

Download [Market1501 Dataset](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op) Or use command line:
```bash
pip install gdown 
pip install --upgrade gdown #!!important!!
gdown 0B8-rUzbwVRk0c054eEozWG9COHM
```

Preparation: Put the images with the same id in one folder. You may use 
```bash
python prepare.py
```
Remember to change the dataset path to your own path.

Futhermore, you also can test our code on [DukeMTMC-reID Dataset]( [GoogleDriver](https://drive.google.com/open?id=1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O) or ([BaiduYun](https://pan.baidu.com/s/1jS0XM7Var5nQGcbf9xUztw) password: bhbh)) Or use command line:
```bash 
gdown 1jjE85dRCMOgRtvJ5RQV9-Afs-2_5dY3O
```
Our baseline code is not such high on DukeMTMC-reID **Rank@1=64.23%, mAP=43.92%**. Hyperparameters are need to be tuned.

- [Optional] [DG-Market](https://github.com/NVlabs/DG-Net#dg-market) is a generated pedestrian dataset of 128,307 images for training a robust model.

### Train
Train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.

Train a model with random erasing by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path --erasing_p 0.5
```

If you want to use **multiple GPUs**, you are suggested to use DDP (`train_DDP.py`) instead of DP (`train.py`). It is because DP lacks the torch supports and may face some [NaN](https://discuss.pytorch.org/t/nan-loss-with-dataparallel/26501).
You could call `train_DDP.py` by running `DDP.sh`. 
```bash
bash DDP.sh 
```

### Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--batchsize` batch size.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.


### Evaluation
```bash
python evaluate.py
```
It will output Rank@1, Rank@5, Rank@10 and mAP results.
You may also try `evaluate_gpu.py` to conduct a faster evaluation with GPU.

For mAP calculation, you also can refer to the [C++ code for Oxford Building](http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/compute_ap.cpp). We use the triangle mAP calculation (consistent with the Market1501 original code).

### re-ranking
```bash
python evaluate_rerank.py
```
**It may take more than 10G Memory to run.** So run it on a powerful machine if possible. 

It will output Rank@1, Rank@5, Rank@10 and mAP results.

