import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pcdet.models.backbones_3d.sa_block import SA_block


class PointContext3D(nn.Module):
    def __init__(self, model_cfg, IN_DIM, dropout=0.1):
        super().__init__()
        self.model_cfg = model_cfg
        self.IN_DIM = IN_DIM

        # Self attention layers
        self.self_attn1 = SA_block(inplanes=self.model_cfg.ATTN_DIM, planes=self.model_cfg.ATTN_DIM)
        self.self_attn2 = SA_block(inplanes=self.model_cfg.ATTN_DIM, planes=self.model_cfg.ATTN_DIM)
        # MLP layer
        self.reduce_dim = nn.Sequential(nn.Conv1d(IN_DIM, self.model_cfg.ATTN_DIM, kernel_size=1),
                                        nn.BatchNorm1d(self.model_cfg.ATTN_DIM),
                                        nn.ReLU(inplace=True),
                                        nn.Conv1d(self.model_cfg.ATTN_DIM, self.model_cfg.ATTN_DIM, kernel_size=1),
                                        nn.BatchNorm1d(self.model_cfg.ATTN_DIM),
                                        nn.ReLU(inplace=True)
                                        )

    def add_context_to_points(self, point_feats):
        """Full pairwise self-attention for all point features"""
        context_points = self.self_attn1(point_feats)
        context_points = self.self_attn2(context_points)
        return context_points

    def forward(self, batch_size, l_features, l_xyz):
        """
        Args:
            :param batch_size:
            :param l_xyz:
            :param l_features:
        """
        # reduce dim of point features
        l_features_red = self.reduce_dim(l_features)
        # get context for every point feature
        point_context_features = self.add_context_to_points(l_features_red)
        return point_context_features
