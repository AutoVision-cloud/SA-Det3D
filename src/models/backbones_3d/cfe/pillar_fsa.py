import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from pcdet.models.backbones_3d.sa_block import SA_block


class PositionalEncoding(nn.Module):
    """
    Positional encoding from https://github.com/wzlxjtu/PositionalEncoding2D
    """
    def __init__(self, d_model, height, width):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        # Not a parameter
        self.register_buffer('pos_table', self._positionalencoding2d())

    def _positionalencoding2d(self):
        """
        :return: d_model*height*width position matrix
        """
        if self.d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.d_model, self.height, self.width)
        # Each dimension use half of d_model
        d_model = int(self.d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., self.width).unsqueeze(1)
        pos_h = torch.arange(0., self.height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, self.height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, self.width)
        return pe

    def forward(self, x, coords):
        pos_encode = self.pos_table[:, coords[:, 2].type(torch.cuda.LongTensor), coords[:, 3].type(torch.cuda.LongTensor)]
        return x + pos_encode.permute(1, 0).contiguous().clone().detach()


class PillarContext3D_fsa(nn.Module):
    """
    Full pair-wise self-attention module for Pillars.
    """
    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.1):
        super().__init__()
        self.model_cfg = model_cfg

        self.nx, self.ny, self.nz = grid_size
        self.position_enc = PositionalEncoding(self.model_cfg.IN_DIM, height=grid_size[1], width=grid_size[0])
        self.layer_norm = nn.LayerNorm(self.model_cfg.IN_DIM, eps=1e-6)

        self.self_attn1 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn2 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)

    def add_context_to_pillars(self, pillar_features, coords, nx, ny, nz):
        batch_size = coords[:, 0].max().int().item() + 1
        batch_context_features = []
        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            pillars = pillar_features[batch_mask, :].unsqueeze(0)

            # Apply pairwise self-attention on VFE pillar features
            context_pillar = self.self_attn1(pillars.permute(0, 2, 1).contiguous())
            context_pillar = self.self_attn2(context_pillar)
            context_pillar = context_pillar.permute(0, 2, 1).contiguous().squeeze(0)

            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            spatial_feature = torch.zeros(
                self.model_cfg.NUM_BEV_FEATURES,
                nz * nx * ny,
                dtype=context_pillar.dtype,
                device=context_pillar.device)
            spatial_feature[:, indices] = context_pillar.t()
            batch_context_features.append(spatial_feature)

        context_pillar_features = torch.cat(batch_context_features, 0)
        context_pillar_features = context_pillar_features.view(batch_size, self.model_cfg.NUM_BEV_FEATURES * nz, ny, nx)
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
s
        Returns:
            context_pillar_features: (N, C)
        """
        pillars = batch_dict['pillar_features']
        coords = batch_dict['voxel_coords']

        # get position encoding for pillars
        pillar_pos_enc = self.position_enc(pillars, coords)
        pillar_pos_enc = self.layer_norm(pillar_pos_enc)

        # get context for every pillar
        context_features = self.add_context_to_pillars(pillar_pos_enc, coords, self.nx, self.ny, self.nz)

        # generate down-sampled SA-features to concatenate with Conv in decoder_2d module
        pillar_context = [F.interpolate(context_features, scale_factor=0.5, mode='bilinear'),
                          F.interpolate(context_features, scale_factor=0.25, mode='bilinear'),
                          F.interpolate(context_features, scale_factor=0.125, mode='bilinear')]
        batch_dict['pillar_context'] = pillar_context
        return batch_dict
