import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pcdet.models.backbones_3d.sa_block import SA_block
from ....utils import common_utils


class PositionalEncoding(nn.Module):
    """
    Positional encoding from https://github.com/tatp22/multidim-positional-encoding
    """
    def __init__(self, d_model, height, width, depth=2):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.depth = depth
        # Not a parameter
        self.register_buffer('pos_table', self._positionalencoding3d())

    def _positionalencoding3d(self):
        """
        :return: d_model*height*width position matrix
        """
        if self.d_model % 4 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.d_model, self.height, self.width, self.depth)
        # Each dimension use half of d_model
        d_model = int(math.ceil(self.d_model / 3))
        if d_model % 2:
            d_model += 1
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        div_term_depth = torch.exp(torch.arange(0., d_model-2, 2) * -(math.log(10000.0) / d_model-2))
        pos_w = torch.arange(0., self.width).unsqueeze(1)
        pos_h = torch.arange(0., self.height).unsqueeze(1)
        pos_d = torch.arange(0., self.depth).unsqueeze(1)
        pe[0:d_model:2, :, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, self.height, 1, self.depth)
        pe[1:d_model:2, :, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).unsqueeze(-1).repeat(1, self.height, 1, self.depth)
        pe[d_model:2*d_model:2, :, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.width, self.depth)
        pe[d_model + 1:2*d_model:2, :, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).unsqueeze(-1).repeat(1, 1, self.width, self.depth)
        pe[2*d_model::2, :, :, :] = torch.sin(pos_d * div_term_depth).transpose(0, 1).unsqueeze(1).unsqueeze(2).repeat(1, self.height, self.width, 1)
        pe[2*d_model + 1::2, :, :, :] = torch.cos(pos_d * div_term_depth).transpose(0, 1).unsqueeze(1).unsqueeze(2).repeat(1, self.height, self.width, 1)
        return pe

    def forward(self, x, coords):
        pos_encode = self.pos_table[:, coords[:, 2].type(torch.cuda.LongTensor),
                                       coords[:, 3].type(torch.cuda.LongTensor),
                                       coords[:, 1].type(torch.cuda.LongTensor)]
        return x + pos_encode.permute(1, 0).contiguous().clone().detach()


class VoxelContext3D_fsa(nn.Module):
    def __init__(self, model_cfg, grid_size, voxel_size, point_cloud_range, dropout=0.1):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # Positional encoding layers
        self.position_enc = PositionalEncoding(self.model_cfg.IN_DIM,
                                               height=grid_size[1]//self.model_cfg.downsampled,
                                               width=grid_size[0]//self.model_cfg.downsampled,
                                               depth=2)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(self.model_cfg.IN_DIM, eps=1e-6)

        # Self-attention layers
        self.self_attn1 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)
        self.self_attn2 = SA_block(inplanes=self.model_cfg.IN_DIM, planes=self.model_cfg.IN_DIM)

    def add_context_to_voxels(self, voxel_features, coords, nx, ny, nz):
        batch_size = coords[:, 0].max().int().item() + 1
        batch_context_features = []

        for batch_idx in range(batch_size):
            batch_mask = coords[:, 0] == batch_idx
            voxel_pillars = voxel_features[batch_mask, :].unsqueeze(0)
            voxel_pillars = voxel_pillars.permute(0, 2, 1).contiguous()

            voxel_pillars = self.self_attn1(voxel_pillars)
            voxel_pillars = self.self_attn2(voxel_pillars)
            voxel_pillars = voxel_pillars.permute(0, 2, 1).contiguous().squeeze(0)

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
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        nz, ny, nx = encoded_spconv_tensor.spatial_shape
        cur_coords = encoded_spconv_tensor.indices
        voxel_feats = encoded_spconv_tensor.features

        # get position encoding for voxels
        voxel_pos_enc = self.dropout(self.position_enc(voxel_feats, cur_coords))
        voxel_pos_enc = self.layer_norm(voxel_pos_enc)

        # get context for every voxel
        voxel_context_features = self.add_context_to_voxels(voxel_pos_enc, cur_coords, nx, ny, nz)

        # generate down-sampled self-attention-features to concatenate with Conv
        voxel_context = [voxel_context_features,
                         F.interpolate(voxel_context_features, scale_factor=0.5, mode='bilinear'),
                         ]
        batch_dict['voxel_context'] = voxel_context
        return batch_dict
