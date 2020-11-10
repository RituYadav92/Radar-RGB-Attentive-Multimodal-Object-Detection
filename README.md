[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-376/)
[![TensorFlow 1.1](https://img.shields.io/badge/tensorflow-1.1.4-blue.svg)](https://github.com/tensorflow/tensorflow/releases/tag/v1.15.2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/RituYadav92/NuScenes_radar_RGBFused-Detection/blob/master/LICENCE)


## Radar+RGB Attentive Fusion For Robust Object Detection in Autonomous Vehicles(ICIP 2020)

## Description: 
Code is for two robust multimodal two-stage object detection networks BIRANet and RANet. The two modalities used in these
architectures are radar signals and RGB camera images. These two networks have the same base architecture with differences
in anchor generation and RPN target generation methods, which are explained in the paper. Evaluation is done on NuScenes dataset[https://www.nuscenes.org],
and results are compared with Faster R-CNN with feature pyramid network for object detection(FFPN)[https://arxiv.org/pdf/1612.03144.pdf].
Both proposed networks proved to be robust in comparison to FFPN. BIRANet performs better than FFPN and also proved to be more robust.
RANet is evaluated to be robust and works reasonably well with fewer anchors, which are merely based on radar points.
For further details, please refer to our paper(https://ieeexplore.ieee.org/document/9191046).

<img src="https://github.com/RituYadav92/NuScenes_radar_RGBFused-Detection/blob/master/Demo/Front.gif" alt="alt text" width="300" height="200"> <img src="https://github.com/RituYadav92/NuScenes_radar_RGBFused-Detection/blob/master/Demo/Back_Cam.gif" alt="alt text" width="300" height="200">

## Packing List: 
The repository includes:
* Source code(which is built on Mask RCNN code base structure but without mask/segmentation branch hence equivalent to FFPN.)
* Training code
* Trained weights for testing/evaluation
* ParallelModel class for multi-GPU training
* Evaluation on MS COCO metrics (AP & AR) with changes mentioned in the paper

## Player Information:  

### Installation
1. Download [modified small NuScenes dataset](https://seafile.rlp.net/d/957d8819906a4d6c8d57/) (size: 2.7 GB).
2. Install pycocotools using https://github.com/cocodataset/cocoapi.
3. Install dependencies from `requirement.txt`
   ```bash
   pip3 install -r requirements.txt
   ```
4. Run setup from the repository root directory.
    ```bash
    python3 setup.py install
    ``` 

### Training and Evaluation
Training and evaluation code is in `samples/coco/nucoco.py`.
You can import this module in Jupyter notebook  or you can run it directly from the command line as such:

```
# Train a new model starting from pre-trained COCO weights
python3 samples/coco/nucoco.py train --dataset=/path/to/nuscenes/ --model=coco

# Train a new model starting from ImageNet weights
python3 samples/coco/nucoco.py train --dataset=/path/to/nuscenes/ --model=imagenet

# Continue training a model that you had trained earlier
python3 samples/coco/nucoco.py train --dataset=/path/to/nuscenes/ --model=/path/to/weights.h5

# Continue training the last model you trained. This will find
# the last trained weights in the model directory.
python3 samples/coco/nucoco.py train --dataset=/path/to/nuscenes/ --model=last

# Run COCO evaluation on the last trained model
python3 samples/coco/nucoco.py evaluate --dataset=/path/to/nuscenes/ --model=last
```
Optional arguments
```
# To select network:
--net= BIRANet/RANet Default = BIRANet.
# To select image resolution:
--resolution=512/1024   Default =1024

# The training schedule, learning rate, and other parameters should be set in `mrcnn/config.py`.
```

## Contact Information: 
Ritu Yadav (Email: er.ritu92@gmail.com)
