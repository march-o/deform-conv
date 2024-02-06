# Deform-conv
Bachelors theseis project about research and evaluation of modifications to deformable convolution

Note that this repo is still in development


## Getting started
If something does not work, please refer to the [official MMDetection installation guide](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation) and [InternImage installation guide](https://github.com/OpenGVLab/InternImage/tree/master/detection)

### Create a virtual environment
```bash
conda create -n deform-conv python=3.10
conda activate deform-conv
```
### Base requirements
```bash
pip install requirements.txt
```
### MMDetection and friends
```bash
pip install -U openmim
mim install mmengine

mim install "mmcv>=2.0.0" -v
# This can take some time, depending on your cuda version, it might compile from source

mim install mmdet -v
```
### Install deformable convolution v3 CUDA op
```bash
cd ops_dcnv3 && bash make.sh
```

## Download coco2017 dataset
```bash
python tools/download_dataset.py --save-dir data/coco
```

## Training with example model mask-rcnn internimage-tiny
```bash
python train.py configs/coco/mask_rcnn_internimage_t_fpn_1x_coco.py
```