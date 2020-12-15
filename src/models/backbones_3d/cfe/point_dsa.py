import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.models.backbones_3d.sa_block import SA_block


class PointContext3D(nn.Module):
    def __init__(self, model_cfg, IN_DIM, dropout=0.1):
        super().__init__()
        self.model_cfg = model_cfg
        self.pc_range = self.model_cfg.PC_RANGE
        self.voxel_size = self.model_cfg.VOXEL_SIZE
        self.grid_size = self.model_cfg.GRID_SIZE
        self.IN_DIM = IN_DIM

        # Self-attention layers
        self.self_attn1 = SA_block(inplanes=self.model_cfg.ATTN_DIM, planes=self.model_cfg.ATTN_DIM)
        self.self_attn2 = SA_block(inplanes=2 * self.model_cfg.ATTN_DIM, planes=2 * self.model_cfg.ATTN_DIM)
        # MLP layers
        self.reduce_dim = nn.Sequential(nn.Conv1d(IN_DIM, self.model_cfg.ATTN_DIM, kernel_size=1),
                                        nn.BatchNorm1d(self.model_cfg.ATTN_DIM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(self.model_cfg.ATTN_DIM, self.model_cfg.ATTN_DIM, kernel_size=1),
                                        nn.BatchNorm1d(self.model_cfg.ATTN_DIM),
                                        nn.ReLU(inplace=True)
                                        )
        self.reduce_dim_cat = nn.Sequential(nn.Conv1d(2*self.model_cfg.ATTN_DIM, self.model_cfg.ATTN_DIM, kernel_size=1),
                                            nn.BatchNorm1d(self.model_cfg.ATTN_DIM),
                                            nn.ReLU(inplace=True),
                                            nn.Conv1d(self.model_cfg.ATTN_DIM, self.model_cfg.ATTN_DIM, kernel_size=1),
                                            nn.BatchNorm1d(self.model_cfg.ATTN_DIM),
                                            nn.ReLU(inplace=True)
                                            )

    def add_context_to_points(self, point_feats, l_1=None, l_2=None):
        """Add self-attention context across every selected and deformed point"""
        global context_points
        if l_1 is None and l_2 is None:
            context_points = self.self_attn1(point_feats)
            context_points = self.self_attn2(context_points)
        if l_1 is not None and l_2 is None:
            context_points = self.self_attn1(point_feats)
            ms1 = torch.cat([l_1, context_points], dim=1)
            context_points_ms1 = self.self_attn2(ms1)
            context_points = self.reduce_dim_cat(context_points_ms1)
        if l_1 is not None and l_2 is not None:
            ms1 = torch.cat([l_1, point_feats], dim=1)
            ms1 = self.reduce_dim_cat(ms1)
            ms1_context = self.self_attn1(ms1)
            ms2 = torch.cat([l_2, ms1_context], dim=1)
            ms2_context = self.self_attn2(ms2)
            context_points = self.reduce_dim_cat(ms2_context)
        return context_points

    def forward(self, batch_size, l_features, l_xyz, l_conv1=None, l_conv2=None):
        """
        Args:
            :param l_conv2:
            :param l_conv1:
            :param batch_size:
            :param l_xyz:
            :param l_features:
        """
        # reduce dim of selected points
        l_features_red = self.reduce_dim(l_features)
        # get context for every deformed point features input to this module
        point_context_features = self.add_context_to_points(l_features_red, l_conv1, l_conv2)
        return point_context_features
