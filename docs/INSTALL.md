## Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 16.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.3)
* CUDA 9.0 or higher
* `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634))


### Install `pcdet`
a. Go to the OpenPCDet repository.
```
cd OpenPCDet/
```

b. Install the dependant libraries as follows:

* Install the dependant python libraries: 
```
pip install -r requirements.txt 
```

* Install the SparseConv library, we use the non-official implementation from [`spconv`](https://github.com/traveller59/spconv). 
Note that we use the initial version of `spconv`, make sure you install the `spconv v1.0` ([commit 8da6f96](https://github.com/traveller59/spconv/tree/8da6f967fb9a054d8870c3515b1b44eca2103634)) instead of the latest one.
* For docker image, please refer to:
```
docker pull trn84/repo:latest
```

c. Install this `pcdet` library by running the following command:
```shell
python setup.py develop
```


## Dataset Preparation for KITTI
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):

```
PCDet
├── data
│   ├── kitti
│   │   │──ImageSets
│   │   │──training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes)
│   │   │──testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m pcdet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```
