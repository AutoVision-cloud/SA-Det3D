import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils

from pcdet.models.backbones_3d.sa_block import SA_block, SA_block_def


class PillarContext3D_dsa(nn.Module):
    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.3):
        super().__init__()
        self.model_cfg = model_cfg

        self.nx, self.ny, self.nz = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # layers to deform + aggregate local features
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

        # self-attention layers to operate on deformed pillars
        self.self_full_fast_attn = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.reduce_dim = nn.Sequential(nn.Conv1d(2*self.model_cfg.IN_DIM, self.model_cfg.IN_DIM, kernel_size=1),
                                        nn.BatchNorm1d(self.model_cfg.IN_DIM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(self.model_cfg.IN_DIM, self.model_cfg.IN_DIM, kernel_size=1),
                                        nn.BatchNorm1d(self.model_cfg.IN_DIM),
                                        nn.ReLU(inplace=True)
                                        )
        self.self_attn_ms1 = SA_block(inplanes=2*self.model_cfg.IN_DIM, planes=2*self.model_cfg.IN_DIM)
        self.self_attn_ms2 = SA_block(inplanes=2*self.model_cfg.IN_DIM, planes=2*self.model_cfg.IN_DIM)

    def get_keypoints(self, batch_size, coords, src_points):
        """
        Select keypoints, i.e. a subset of pillar coords to deform, aggregate local features and then attend to.
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

    def get_local_keypoint_features(self, keypoints, pillar_center, pillar_features, coords):
        """
        Get local features of deformed pillar-subset/keypoints.
        :param keypoints:
        :param pillar_center:
        :param pillar_features:
        :param coords:
        :return: B x num_keypoints X C
        """
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        xyz_batch_cnt = torch.zeros([batch_size]).int().cuda()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (coords[:, 0] == bs_idx).sum()

        def_xyz, local_features = self.adapt_context(
            xyz=pillar_center,
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=pillar_features
        )
        def_xyz = def_xyz.view(batch_size, num_keypoints, -1)
        local_features = local_features.view(batch_size, num_keypoints, -1)
        return def_xyz, local_features

    def get_context_features(self, batch_size, local_features):
        batch_global_features = []
        for batch_idx in range(batch_size):
            init_idx = batch_idx * self.model_cfg.NUM_KEYPOINTS
            local_feat = local_features[init_idx:init_idx + self.model_cfg.NUM_KEYPOINTS, :].unsqueeze(0)
            local_feat = local_feat.permute(0, 2, 1).contiguous()
            global_feat = self.self_full_fast_attn(local_feat)

            # SA-1
            ms_feat1 = torch.cat([local_feat, global_feat], dim=1)
            attn_feat1 = self.self_attn_ms1(ms_feat1)
            attn_feat1 = self.reduce_dim(attn_feat1)

            # SA-2
            ms_feat2 = torch.cat([local_feat, attn_feat1], dim=1)
            attn_feat2 = self.self_attn_ms2(ms_feat2)
            attn_feat2 = self.reduce_dim(attn_feat2)
            context_feat = attn_feat2.permute(0, 2, 1).contiguous().squeeze(0)

            batch_global_features.append(context_feat)
        batch_global_features = torch.cat(batch_global_features, 0)
        return batch_global_features

    def get_context_image(self, batch_size, keypoints, pillar_center, global_features, coords):
        # pillar coordinates
        new_xyz = pillar_center
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        batch_idx = coords[:, 0]
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (batch_idx == k).sum()
        # keypoint coordinates and features
        xyz = keypoints.view(-1, 3)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(keypoints.shape[1])
        # UnPool to get global context enhanced pillar features for every pillar
        pillar_features = self.decode(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz.contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=global_features,
        )  # (M1 + M2 ..., C)

        # Create pseudo-image for self-attention pillar features
        batch_context_features = []
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            pillars = pillar_features[batch_mask, :]

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)

            spatial_feature = torch.zeros(
                self.model_cfg.NUM_BEV_FEATURES,
                self.nz * self.nx * self.ny,
                dtype=pillars.dtype,
                device=pillars.device)
            spatial_feature[:, indices] = pillars.t()
            batch_context_features.append(spatial_feature)
        context_pillar_features = torch.cat(batch_context_features, 0)
        context_pillar_features = context_pillar_features.view(batch_size,
                                                               self.model_cfg.NUM_BEV_FEATURES * self.nz, self.ny,
                                                               self.nx)
        return context_pillar_features

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
        pillars = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']

        # Preprocessing for pillar locations
        pillar_center = torch.zeros_like(coords[:, :3])
        # front-back (X); left-right (Y); up-down (Z)
        pillar_center[:, 0] = coords[:, 3] * self.voxel_x + self.x_offset
        pillar_center[:, 1] = coords[:, 2] * self.voxel_y + self.y_offset
        pillar_center[:, 2] = coords[:, 1] * self.voxel_z + self.z_offset

        # Get keypoints
        keypoints = self.get_keypoints(batch_size, coords, pillar_center)

        # Get deformed and aggregated keypoint feature from pillars
        def_xyz, local_keypoint_feats = self.get_local_keypoint_features(keypoints, pillar_center, pillars, coords)
        local_keypoint_feats = local_keypoint_feats.view(batch_size*self.model_cfg.NUM_KEYPOINTS, -1).contiguous()

        # Get context for a subset of selected and deformed  pillars
        context_features = self.get_context_features(batch_size, local_keypoint_feats)

        # Get context enhanced pseudo image - UnPool step here
        context_pillar_features = self.get_context_image(batch_size, def_xyz, pillar_center, context_features, coords)

        # generate down-sampled SA-features to concatenate with Conv in decoder2d
        pillar_context = [F.interpolate(context_pillar_features, scale_factor=0.5, mode='bilinear'),
                          F.interpolate(context_pillar_features, scale_factor=0.25, mode='bilinear'),
                          F.interpolate(context_pillar_features, scale_factor=0.125, mode='bilinear')]
        batch_dict['pillar_context'] = pillar_context
        return batch_dict


class PillarContext3D_def(nn.Module):
    """Up-sampling method based on Set-transformer (ICML 2019)"""
    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.3):
        super().__init__()
        self.model_cfg = model_cfg

        self.nx, self.ny, self.nz = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # layers to deform + aggregate local features
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
        self.self_full_fast_attn = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn1 = SA_block_def(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn2 = SA_block_def(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)

    def get_keypoints(self, batch_size, coords, src_points):
        """
        Select keypoints, i.e. a subset of pillar coords to deform, aggregate local features and then attend to.
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

    def get_local_keypoint_features(self, keypoints, pillar_center, pillar_features, coords):
        """
        Get local features of deformed pillar-subset/keypoints.
        :param keypoints:
        :param pillar_center:
        :param pillar_features:
        :param coords:
        :return: B x num_keypoints X C
        """
        batch_size, num_keypoints, _ = keypoints.shape
        new_xyz = keypoints.view(-1, 3)
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int().fill_(num_keypoints)

        xyz_batch_cnt = torch.zeros([batch_size]).int().cuda()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = (coords[:, 0] == bs_idx).sum()

        def_xyz, local_features = self.adapt_context(
            xyz=pillar_center,
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=pillar_features
        )
        def_xyz = def_xyz.view(batch_size, num_keypoints, -1)
        local_features = local_features.view(batch_size, num_keypoints, -1)
        return def_xyz, local_features

    def get_context_features(self, batch_size, pillars, local_features, coords):
        batch_global_features = []
        for batch_idx in range(batch_size):
            init_idx = batch_idx * self.model_cfg.NUM_KEYPOINTS
            local_feat = local_features[init_idx:init_idx + self.model_cfg.NUM_KEYPOINTS, :].unsqueeze(0)
            local_feat = local_feat.permute(0, 2, 1).contiguous()
            local_sa_feat = self.self_full_fast_attn(local_feat)

            batch_mask = coords[:, 0] == batch_idx
            pillar_feat = pillars[batch_mask, :].unsqueeze(0).permute(0, 2, 1).contiguous()

            attn_feat1 = self.self_attn1(pillar_feat, local_sa_feat)
            attn_feat2 = self.self_attn2(attn_feat1, local_sa_feat)
            context_pillar = attn_feat2.permute(0, 2, 1).contiguous().squeeze(0)

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(
                self.model_cfg.NUM_BEV_FEATURES,
                self.nz * self.nx * self.ny,
                dtype=context_pillar.dtype,
                device=context_pillar.device)
            spatial_feature[:, indices] = context_pillar.t()
            batch_global_features.append(spatial_feature)

        context_pillar_features = torch.cat(batch_global_features, 0)
        context_pillar_features = context_pillar_features.view(batch_size, self.model_cfg.NUM_BEV_FEATURES * self.nz, self.ny, self.nx)
        return context_pillar_features

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
        pillars = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']

        # Preprocessing for pillar locations
        pillar_center = torch.zeros_like(coords[:, :3])
        # front-back (X); left-right (Y); up-down (Z)
        pillar_center[:, 0] = coords[:, 3] * self.voxel_x + self.x_offset
        pillar_center[:, 1] = coords[:, 2] * self.voxel_y + self.y_offset
        pillar_center[:, 2] = coords[:, 1] * self.voxel_z + self.z_offset

        # Get keypoints
        keypoints = self.get_keypoints(batch_size, coords, pillar_center)

        # Get deformed and aggregated keypoint feature from pillars
        def_xyz, local_keypoint_feats = self.get_local_keypoint_features(keypoints, pillar_center, pillars, coords)
        local_keypoint_feats = local_keypoint_feats.view(batch_size * self.model_cfg.NUM_KEYPOINTS, -1).contiguous()

        # Get context for a subset of selected and deformed  pillars
        context_pillar_features = self.get_context_features(batch_size, pillars, local_keypoint_feats, coords)

        # generate down-sampled SA-features to concatenate with Conv in decoder2d
        pillar_context = [F.interpolate(context_pillar_features, scale_factor=0.5, mode='bilinear'),
                          F.interpolate(context_pillar_features, scale_factor=0.25, mode='bilinear'),
                          F.interpolate(context_pillar_features, scale_factor=0.125, mode='bilinear')]
        batch_dict['pillar_context'] = pillar_context
        return batch_dict

