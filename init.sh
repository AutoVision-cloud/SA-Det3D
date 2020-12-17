#!/bin/bash

# copy files
cp -r src/models/backbones_2d/* OpenPCDet/pcdet/models/backbones_2d/

cp -r src/models/backbones_3d/*.py OpenPCDet/pcdet/models/backbones_3d/
cp -r src/models/backbones_3d/pfe/* OpenPCDet/pcdet/models/backbones_3d/pfe/
cp -r src/models/backbones_3d/cfe OpenPCDet/pcdet/models/backbones_3d/

cp -r src/models/detectors/* OpenPCDet/pcdet/models/detectors/

cp -r src/ops/pointnet2/pointnet2_stack/* OpenPCDet/pcdet/ops/pointnet2/pointnet2_stack/
cp -r src/ops/pointnet2/pointnet2_batch/* OpenPCDet/pcdet/ops/pointnet2/pointnet2_batch/

cp -r src/tools/* OpenPCDet/tools/

cp -r configs/* OpenPCDet/tools/cfgs/kitti_models/

cp -r requirements.txt OpenPCDet/


