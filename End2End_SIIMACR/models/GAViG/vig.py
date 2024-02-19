"""
Code is referenced from
https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/vig_pytorch/pyramid_vig.py

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Conv2d


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0), nn.BatchNorm2d(hidden_features),
        )
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0), nn.BatchNorm2d(out_features),
        )

    def forward(self, x):
        x, gaze = x[0], x[1]
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x + shortcut
        return x, gaze


class Stem(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim // 2, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim // 2), nn.GELU(),
            nn.Conv2d(out_dim // 2, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim), nn.GELU(),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1), nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        return self.convs(x)


class Downsample(nn.Module):
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim),
        )
        self.gaze_down2 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x, gaze = x[0], x[1]
        gaze = self.gaze_down2(gaze)
        x = self.conv(x)
        return x, gaze


class Grapher(nn.Module):
    def __init__(self, in_channels, kernel_size=9, dilation=1, r=1):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0), nn.BatchNorm2d(in_channels),
        )
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, r)
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0), nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        x, gaze = x[0], x[1]
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x, gaze)
        x = self.fc2(x)
        x = x + _tmp
        return x, gaze


class GraphConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,):
        super(GraphConv2d, self).__init__()
        self.gconv = MRConv2d(in_channels, out_channels)

    def forward(self, x, edge_index, y=None):
        return self.gconv(x, edge_index, y)


def batched_index_select(x, idx):
    batch_size, num_dims, num_vertices_reduced = x.shape[:3]
    _, num_vertices, k = idx.shape
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_vertices_reduced
    idx = idx + idx_base
    idx = idx.contiguous().view(-1)

    x = x.transpose(2, 1)
    feature = x.contiguous().view(batch_size * num_vertices_reduced, -1)[idx, :]
    feature = feature.view(batch_size, num_vertices, k, num_dims).permute(0, 3, 1, 2).contiguous()
    return feature


class BasicConv(Seq):
    def __init__(self, channels):
        m = []
        for i in range(1, len(channels)):
            m.append(Conv2d(channels[i - 1], channels[i], 1, bias=True, groups=4))
            m.append(nn.BatchNorm2d(channels[-1], affine=True))
            m.append(nn.GELU())
        super(BasicConv, self).__init__(*m)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class MRConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MRConv2d, self).__init__()
        self.nn = BasicConv([in_channels * 2, out_channels])

    def forward(self, x, edge_index, y=None):
        # b x 48 x 3136 x 9
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            # b x 48 x 3136 x 9
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        # b x 48 x 3136 x 1
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        # b x 48 x 2 x 3136 x 1
        xx = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2)
        # b x 96 x 3136 x 1
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        # b x 96 x 3136 x 1
        x = self.nn(x)
        return x


class DyGraphConv2d(GraphConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, r=1):
        # [48, 96, 240, 384][i], 48 x 2, 9, 1, 'mr', 'gelu'
        # 'batch', True, False, 0.2, [4, 2, 1, 1][i]
        super(DyGraphConv2d, self).__init__(in_channels, out_channels)
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation)

    def forward(self, x, gaze, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        gaze_y = None
        if self.r > 1:
            # n x 48 x 14 x 14
            y = F.avg_pool2d(x, self.r, self.r)
            # n x 48 x 196 x 1
            y = y.reshape(B, C, -1, 1).contiguous()
            gaze_y = F.max_pool2d(gaze, self.r, self.r)
            gaze_y = gaze_y.reshape(B, 1, -1, 1).contiguous()
        # n x 48 x 3136 x 1
        x = x.reshape(B, C, -1, 1).contiguous()
        gaze = gaze.reshape(B, 1, -1, 1).contiguous()
        # 2 x n x 3136 x 9
        edge_index = self.dilated_knn_graph(x, y, gaze, gaze_y)
        # n x 96 x 3136 x 1
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        # n x 96 x 56 x 56
        return x.reshape(B, -1, H, W).contiguous()


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=9, dilation=1):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.k = k
        self._dilated = DenseDilated(k, dilation)

    def forward(self, x, y=None, gaze=None, gaze_y=None):
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, gaze, gaze_y, self.k * self.dilation)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = dense_knn_matrix(x, gaze, self.k * self.dilation)
        return self._dilated(edge_index)


def pairwise_distance(x):
    with torch.no_grad():
        x_inner = -2 * torch.matmul(x, x.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        return x_square + x_inner + x_square.transpose(2, 1)


def xy_pairwise_distance(x, y):
    with torch.no_grad():
        xy_inner = -2 * torch.matmul(x, y.transpose(2, 1))
        x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
        y_square = torch.sum(torch.mul(y, y), dim=-1, keepdim=True)
        return x_square + xy_inner + y_square.transpose(2, 1)


def dense_knn_matrix(x, gaze, k=16):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        gaze = gaze.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = pairwise_distance(x.detach())
        gaze_dist = pairwise_distance(gaze.detach()) * 1.
        # ----------------------------------norm--------------------------------------------
        max_values, _ = torch.max(gaze.reshape([gaze.shape[0], -1]), dim=1, keepdim=True)
        min_values, _ = torch.min(gaze.reshape([gaze.shape[0], -1]), dim=1, keepdim=True)
        max_values, min_values = max_values.unsqueeze(-1), min_values.unsqueeze(-1)
        gaze_norm = (gaze - min_values) / (max_values - min_values)
        # ----------------------------------------------------------------------------------
        gaze_dist = gaze_dist * gaze_norm

        dist += gaze_dist
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, gaze, gaze_y, k=16):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        gaze = gaze.transpose(2, 1).squeeze(-1)
        gaze_y = gaze_y.transpose(2, 1).squeeze(-1)

        batch_size, n_points, n_dims = x.shape

        dist = xy_pairwise_distance(x.detach(), y.detach())
        gaze_dist = xy_pairwise_distance(gaze.detach(), gaze_y.detach()) * 1.
        # ----------------------------------norm--------------------------------------------
        max_values, _ = torch.max(gaze.reshape([gaze.shape[0], -1]), dim=1, keepdim=True)
        min_values, _ = torch.min(gaze.reshape([gaze.shape[0], -1]), dim=1, keepdim=True)
        max_values, min_values = max_values.unsqueeze(-1), min_values.unsqueeze(-1)
        gaze_norm = (gaze - min_values) / (max_values - min_values)
        # ----------------------------------------------------------------------------------
        gaze_dist = gaze_dist * gaze_norm

        dist += gaze_dist
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class DenseDilated(nn.Module):
    def __init__(self, k=9, dilation=1):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.k = k

    def forward(self, edge_index):
        edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class ViG_Gaze(torch.nn.Module):
    def __init__(self, num_classes=2):
        super(ViG_Gaze, self).__init__()
        k = 9
        blocks = [2, 2, 6, 2]
        channels = [80, 160, 400, 640]
        reduce_ratios = [4, 2, 1, 1]
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))  # 1 x 48 x 56 x 56

        self.stem = Stem(out_dim=channels[0])
        self.gaze_down4 = nn.MaxPool2d((4, 4), 4)

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(
                        Grapher(channels[i], k, idx // 4 + 1, reduce_ratios[i]),
                        FFN(channels[i], channels[i] * 4)
                    )]
                idx += 1
        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True), nn.BatchNorm2d(1024), nn.GELU(), nn.Dropout(),
                              nn.Conv2d(1024, 1000, 1, bias=True))
        self.pred2 = Seq(nn.BatchNorm2d(1000), nn.GELU(), nn.Dropout(),
                         nn.Conv2d(1000, num_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        inputs, gaze = inputs[0], inputs[1]
        inputs = inputs.repeat(1, 3, 1, 1)
        # b x 48 x 56 x 56
        # pos_embed 1 x 48 x 56 x 56
        gaze = self.gaze_down4(gaze)
        x = self.stem(inputs) + self.pos_embed
        # blocks = [2, 2, 6, 2]
        # channels = [48, 96, 240, 384]
        # b x 48 x 56 x 56 -> b x 96 x 28 x 28 -> b x 240 x 14 x 14 -> b x 384 x 7 x 7
        for i in range(len(self.backbone)):
            x, gaze = self.backbone[i]([x, gaze])
        # b x 384 x 7 x 7 -> b x 384 x 1 x 1
        x = F.adaptive_avg_pool2d(x, 1)
        # b x 384 x 1 x 1 -> b x 1024 x 1 x 1 -> b x n_classes
        x = self.prediction(x)
        return self.pred2(x).squeeze(-1).squeeze(-1)
