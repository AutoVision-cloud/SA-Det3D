from collections import defaultdict
from pathlib import Path

import numpy as np
import torch.utils.data as torch_data

from pcdet.utils import common_utils
from pcdet.datasets.augmentor.data_augmentor import DataAugmentor
from pcdet.datasets.processor.data_processor import DataProcessor
from pcdet.datasets.processor.point_feature_encoder import PointFeatureEncoder

from spconv.utils import VoxelGenerator


def noise_robustness(batch_size, ret):
    """
    Add noise points around every ground truth box for SECOND and Point-RCNN.
    The idea is proposed in TANet (AAAI 2020), https://arxiv.org/pdf/1912.05163.pdf.

    How to use:
    For now, this function should be added to line 179 of pcdet/datasets/dataset.py
    TODO: Integrate with pcdet/datasets/dataset.py
    Change no_of_pts to number of uniformly generated noise points desired
    around each GT bounding box.

    :param batch_size:
    :param ret:
    :return:
    """
    if True:
        np.random.seed(0)  # Numpy module.
        voxels_flag = True
        no_of_pts = 100
        voxel_list = []
        coords_list = []
        num_points_list = []
        for k in range(batch_size):
            for i_box in range(len(ret['gt_boxes'][k])):
                bbox = ret['gt_boxes'][k][i_box]
                cx = bbox[0]
                cy = bbox[1]
                cz = bbox[2]
                l = bbox[3]
                w = bbox[4]
                h = bbox[5]
                z1 = np.random.uniform(cx+l//2, cx+3*l, (no_of_pts//2, 1))
                z2 = np.random.uniform(cx-l//2, cx-3*l, (no_of_pts//2, 1))
                z = np.concatenate([z1, z2], 0)
                y1 = np.random.uniform(cy+w//2, cy+3*w, (no_of_pts//2, 1))
                y2 = np.random.uniform(cy-w//2, cy-3*w, (no_of_pts//2, 1))
                y = np.concatenate([y1, y2], 0)
                x1 = np.random.uniform(cz+h//2, cz+3*h, (no_of_pts//2, 1))
                x2 = np.random.uniform(cz-h//2, cz-3*h, (no_of_pts//2, 1))
                x = np.concatenate([x1, x2], 0)
                r = np.ones([no_of_pts, 1])
                b = np.zeros([no_of_pts, 1]) + k
                noise = np.concatenate([b, z, y, x, r], 1)
                ret['points'] = np.concatenate([ret['points'], noise], 0)

            if voxels_flag:
                voxel_generator = VoxelGenerator(
                    voxel_size=[0.05, 0.05, 0.1],
                    point_cloud_range=[0, -40, -3, 70.4, 40, 1],
                    max_num_points=5,
                    max_voxels=40000,
                )
                batch_mask = ret['points'][:, 0] == k
                points_extract = ret['points'][batch_mask, :]
                voxels, coordinates, num_points = voxel_generator.generate(points_extract[:, 1:])
                voxel_list.append(voxels)
                coords_list.append(coordinates)
                num_points_list.append(num_points)

        if voxels_flag:
            ret['voxels'] = np.concatenate(voxel_list, axis=0)
            ret['voxel_num_points'] = np.concatenate(num_points_list, axis=0)
            coors = []
            for i, coor in enumerate(coords_list):
                coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret['voxel_coords'] = np.concatenate(coors, axis=0)

    return ret
