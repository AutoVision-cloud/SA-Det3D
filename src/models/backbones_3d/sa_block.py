import torch
from torch import nn


class SA_block(nn.Module):
    """Self-Attention block with dot product for point/voxel/pillar context.
    A part of the code is from MLCVNet (CVPR 2020).
    """
    def __init__(self, inplanes, planes, groups=4):
        super().__init__()
        self.groups = groups

        # linear transform to get values
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # linear transform to get keys
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # linear transform to get query
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv linear
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups, bias=False)

        # norm (essentially LayerNorm per group)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        # softmax
        self.softmax = nn.Softmax(dim=-1)

    def kernel(self, t, p, g, b, c, h):
        """Return the output after dot product per head
        Args:
            t: output of linear value
            p: output of linear query
            g: output of linear keys
            b: batch size
            c: no of channels
            h: spatial breadth of feature maps
        """
        proj_query = p.view(b, c, h).permute(0, 2, 1)  # B X H x C
        proj_key = g  # B X C x (H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        total_energy = energy
        attention = self.softmax(total_energy)  # BX (N) X (N)
        proj_value = t
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h)
        return out

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h)
        x = self.z(x)
        x = self.gn(x) + residual
        return x


class SA_block_def(nn.Module):
    """Self-Attention block with dot product for point/voxel/pillar context.
    """

    def __init__(self, inplanes, planes, groups=4):
        super().__init__()
        self.groups = groups

        # linear transform to get values
        self.t = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # linear transform to get keys
        self.p = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # linear transform to get query
        self.g = nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv linear
        self.z = nn.Conv1d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups, bias=False)

        # norm (essentially LayerNorm per group)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        # softmax
        self.softmax = nn.Softmax(dim=-1)

    def kernel(self, t, p, g, b, c, h):
        """Return the output after dot product per head
        Args:
            t: output of linear value
            p: output of linear query
            g: output of linear keys
            b: batch size
            c: no of channels
            h: spatial breadth of feature maps
        """
        proj_query = p.permute(0, 2, 1)  # B X H x C
        proj_key = g  # B X C x (H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        total_energy = energy
        attention = self.softmax(total_energy)  # BX (N) X (N)
        proj_value = t
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        return out

    def forward(self, x, y):
        residual = x

        t = self.t(y)
        p = self.p(x)
        g = self.g(y)

        b, c, h = t.size()

        if self.groups and self.groups > 1:
            _c = int(c / self.groups)

            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h)
        x = self.z(x)
        x = self.gn(x) + residual
        return x

