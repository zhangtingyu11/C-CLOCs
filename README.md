Code will be released upon accepted

# 实验环境
CUDA 11.8
Pytorch 2.0.1
single RTX4090
Ubuntu 20.04
MMDet 3.0.0

1.拉取仓库
```
git clone https://github.com/zhangtingyu11/C-CLOCs.git
```
# Data Preparation
2. 软链接kitti数据集
```shell
cd data/
ln -s YOUR_KITTI_DATASET_PATH kitti
下載[道路信息](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?pli=1)
```
kitti数据集的目录结构如下
```shell 
# ImagesSets can be found in https://github.com/open-mmlab/OpenPCDet/tree/master/data/kitti/ImageSets
├── ImageSets
├── testing
│   ├── calib
│   ├── image_2
│   └── velodyne
└── training
    ├── calib
    ├── image_2
    ├── label_2
    ├── planes
    └── velodyne
```

3. 通过SAM模型生成数据库
参考[SAM官方仓库](https://github.com/facebookresearch/segment-anything)安装SAM
创建weights文件夹，并下载权重文件([vit_h](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), [vit_l](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth), [vit_b](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth))到weights文件夹下
运行generate_mask.py
```shell
python tools/generate_masks.py #KITTI
python tools/generate_masks_nuscenes.py # nuscenes
```
生成后的kitti数据集结构如下:
```
├── image_gt_database_train
├── ImageSets
├── testing
│   ├── calib
│   ├── image_2
│   └── velodyne
├── training
│   ├── calib
│   ├── image_2
│   ├── label_2
|   ├── planes
│   └── velodyne
```


4. 将KITTI数据集转成COCO的格式
```shell
python tools/kitti2coco.py
```
数据集的结构如下:
```
├── coco
│   ├── annotations
│   ├── coco_label_2
│   ├── labels
│   │   ├── train_labels
│   │   └── val_labels
│   ├── train2017
│   └── val2017
├── kitti
│   ├── image_gt_database_train
│   ├── ImageSets
│   ├── testing
│   │   ├── calib
│   │   ├── image_2
│   │   └── velodyne
│   └── training
│       ├── calib
│       ├── image_2
│       ├── label_2
│       └── velodyne
```

5. 安装mmdetection
```shell
# 在openpcdet主目录下
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
cd mmdetection
git checkout 3.0.0
pip install -v -e .
```

6. 软链接mmdetection中的data目录
```shell
cd mmdetection
ln -s ../data data
```

7. 生成kitti数据
```shell
#在OpenPCDet主目录下
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

8. 用mmdetection训练nuImages
下载nuImages数据集
借助mmdetection3d中的脚本生成coco数据
```python
python -u tools/dataset_converters/nuimage_converter.py --data-root data/nuImages --version v1.0-train v1.0-val v1.0-mini --out-dir data/nuImages/coco
```

# 借助脚本训练2D目标检测器
KITTI
```shell
cd tools
python train_clocs.py
```
nuScenes
```shell
cd tools
python train_clocs_nuscenes.py
```

# 训练
需要先训练SECOND模型:
```
python train.py --cfg_file tools/cfgs/kitti_models/second_car.yaml
```
再训练C-CLOCs模型
```
cd tools
python train.py --cfg_file tools/cfgs/kitti_models/second_car_clocs_contra_fusion_aug.yaml --pretrained_model ${second预训练模型的路径}
```
