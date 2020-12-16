import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pcdet.models.backbones_3d.sa_block import SA_block

from ....utils import common_utils

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils


class VoxelContext3D_dsa(nn.Module):
    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.1):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.self_attn1 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)

        # Deform and aggregate local features
        mlps = self.model_cfg.LOCAL_CONTEXT.MLPS
        for k in range(len(mlps)):
            mlps[k] = [self.model_cfg.NUM_BEV_FEATURES] + mlps[k]
        self.adapt_context = pointnet2_stack_modules.StackSAModuleMSGAdapt(
            radii=self.model_cfg.LOCAL_CONTEXT.POOL_RADIUS,
            deform_radii=self.model_cfg.LOCAL_CONTEXT.DEFORM_RADIUS,
            nsamples=self.model_cfg.LOCAL_CONTEXT.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.LOCAL_CONTEXT.POOL_METHOD,
            pc_range=self.point_cloud_range,
        )

        # Self-attention layers
        self.self_attn2 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn3 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)

        # UnPool layers
        mlps_decode = self.model_cfg.DECODE.MLPS
        for k in range(len(mlps_decode)):
            mlps_decode[k] = [self.model_cfg.IN_DIM] + mlps_decode[k]
        self.decode = pointnet2_stack_modules.StackSAModuleMSGDecode(
            radii=self.model_cfg.DECODE.POOL_RADIUS,
            nsamples=self.model_cfg.DECODE.NSAMPLE,
            mlps=mlps_decode,
            use_xyz=True,
            pool_method=self.model_cfg.DECODE.POOL_METHOD,
        )

    def get_keypoints(self, batch_size, coords, src_points):
        """
        Get subset of voxels for deformation and context calculation.
        :param batch_size:
        :param coords:
        :param src_points:
        :return: B x num_keypoints x 3
        """
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (coords[:, 0] == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            cur_pt_idxs = pointnet2_stack_utils.furthest_point_sample(sampled_points[:, :, 0:3],
                                                                      self.model_cfg.NUM_KEYPOINTS).long()
            if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                empty_num = self.model_cfg.NUM_KEYPOINTS - sampled_points.shape[1]
                cur_pt_idxs[0, -empty_num:] = cur_pt_idxs[0, :empty_num]
            keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)
            keypoints_list.append(keypoints)
        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3)
        return keypoints

    def get_local_keypoint_features(self, keypoints, voxel_center, voxel_features, coords):
        """
        :param keypoints:
        :param voxel_center:
        :param voxel_features:
        :param coords:
        :return: B x num_keypoints X C
        """
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        xyz_batch_cnt = torch.zeros([batch_size]).int().cuda()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (coords[:, 0] == bs_idx).sum()

        # Deform and locally aggregate
        def_xyz, local_features = self.adapt_context(
            xyz=voxel_center,
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=voxel_features
        )
        def_xyz = def_xyz.view(batch_size, num_keypoints, -1)
        local_features = local_features.view(batch_size, num_keypoints, -1)
        return def_xyz, local_features

    def get_context_features(self, batch_size, local_features):
        """
        Self-attention on subset of voxels deformed
        :param batch_size:
        :param local_features:
        :return:
        """
        batch_global_features = []
        for batch_idx in range(batch_size):
            init_idx = batch_idx * self.model_cfg.NUM_KEYPOINTS
            local_feat = local_features[init_idx:init_idx + self.model_cfg.NUM_KEYPOINTS, :].unsqueeze(0)
            local_feat = local_feat.permute(0, 2, 1).contiguous()
            global_loc_feat  = self.self_attn1(local_feat)

            # SA-1
            attn_feat_1 = self.self_attn2(global_loc_feat)
            # SA-2
            attn_feat_2 = self.self_attn3(attn_feat_1)
            context_feat = attn_feat_2.permute(0, 2, 1).contiguous().squeeze(0)

            batch_global_features.append(context_feat)
        batch_global_features = torch.cat(batch_global_features, 0)
        return batch_global_features

    def get_context_image(self, batch_size, keypoints, voxel_center, global_features, coords, nx, ny, nz):
        # voxel coordinates
        new_xyz = voxel_center
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        batch_idx = coords[:, 0]
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (batch_idx == k).sum()
        # keypoint coordinates and features
        xyz = keypoints.view(-1, 3)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(keypoints.shape[1])
        # UnPool to get global context enhanced voxel features for every voxel
        voxel_features = self.decode(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz.contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=global_features,
        )  # (M1 + M2 ..., C)

        batch_context_features = []
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            voxel_pillars = voxel_features[batch_mask, :]

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(
                self.model_cfg.NUM_BEV_FEATURES,
                nz * nx * ny,
                dtype=voxel_pillars.dtype,
                device=voxel_pillars.device)
            spatial_feature[:, indices] = voxel_pillars.t()
            batch_context_features.append(spatial_feature)

        voxel_pillar_features = torch.cat(batch_context_features, 0)
        voxel_pillar_features = voxel_pillar_features.view(batch_size, self.model_cfg.NUM_BEV_FEATURES * nz, ny, nx)
        return voxel_pillar_features

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional
        Returns:
            context_pillar_features: (N, C)
        """
        batch_size = batch_dict['batch_size']
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        nz, ny, nx = encoded_spconv_tensor.spatial_shape
        cur_coords = encoded_spconv_tensor.indices
        voxel_feats = encoded_spconv_tensor.features

        xyz = common_utils.get_voxel_centers(
            cur_coords[:, 1:4],
            downsample_times=8,
            voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        # Get keypoints / randomly selected subset of voxels
        keypoints = self.get_keypoints(batch_size, cur_coords, xyz)

        # Get deformed and aggregated keypoint feature from voxels
        def_xyz, local_keypoint_feats = self.get_local_keypoint_features(keypoints, xyz, voxel_feats, cur_coords)

        # get context for subset of selected and deformed voxels
        local_keypoint_feats = local_keypoint_feats.view(batch_size*self.model_cfg.NUM_KEYPOINTS, -1).contiguous()
        context_features = self.get_context_features(batch_size, local_keypoint_feats)

        # Get context enhanced voxels (UnPool step)
        voxel_context_features = self.get_context_image(batch_size, def_xyz, xyz,
                                                        context_features, cur_coords,
                                                        nx, ny, nz)

        # generate down-sampled self-attended-features to concatenate with Conv
        voxel_context = [voxel_context_features,
                         F.interpolate(voxel_context_features, scale_factor=0.5, mode='bilinear'),
                         ]
        batch_dict['voxel_context'] = voxel_context
        return batch_dict
