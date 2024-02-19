import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Conv2d


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0), nn.BatchNorm2d(hidden_features))
        self.act = nn.GELU()
        self.fc2 = nn.Sequential(nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0), nn.BatchNorm2d(out_features),)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x + shortcut
        return x


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
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1), nn.BatchNorm2d(out_dim))

    def forward(self, x):
        return self.conv(x)


class Grapher(nn.Module):
    def __init__(self, in_channels, kernel_size=9, dilation=1, r=1):
        super(Grapher, self).__init__()
        self.channels = in_channels
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0), nn.BatchNorm2d(in_channels))
        self.graph_conv = DyGraphConv2d(in_channels, in_channels * 2, kernel_size, dilation, r)
        self.fc2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1, stride=1, padding=0), nn.BatchNorm2d(in_channels))

    def forward(self, x):
        _tmp = x
        x = self.fc1(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        x = x + _tmp
        return x


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
        x_i = batched_index_select(x, edge_index[1])
        if y is not None:
            x_j = batched_index_select(y, edge_index[0])
        else:
            x_j = batched_index_select(x, edge_index[0])
        x_j, _ = torch.max(x_j - x_i, -1, keepdim=True)
        b, c, n, _ = x.shape
        x = torch.cat([x.unsqueeze(2), x_j.unsqueeze(2)], dim=2).reshape(b, 2 * c, n, _)
        x = self.nn(x)
        return x


class DyGraphConv2d(GraphConv2d):
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, r=1):
        super(DyGraphConv2d, self).__init__(in_channels, out_channels)
        self.r = r
        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation)

    def forward(self, x, relative_pos=None):
        B, C, H, W = x.shape
        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r)
            y = y.reshape(B, C, -1, 1).contiguous()
        x = x.reshape(B, C, -1, 1).contiguous()
        edge_index = self.dilated_knn_graph(x, y)
        x = super(DyGraphConv2d, self).forward(x, edge_index, y)
        return x.reshape(B, -1, H, W).contiguous()


class DenseDilatedKnnGraph(nn.Module):
    def __init__(self, k=9, dilation=1):
        super(DenseDilatedKnnGraph, self).__init__()
        self.dilation = dilation
        self.k = k
        self._dilated = DenseDilated(k, dilation)

    def forward(self, x, y=None):
        if y is not None:
            x = F.normalize(x, p=2.0, dim=1)
            y = F.normalize(y, p=2.0, dim=1)
            edge_index = xy_dense_knn_matrix(x, y, self.k * self.dilation)
        else:
            x = F.normalize(x, p=2.0, dim=1)
            edge_index = dense_knn_matrix(x, self.k * self.dilation)
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


def dense_knn_matrix(x, k=16):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = pairwise_distance(x.detach())
        _, nn_idx = torch.topk(-dist, k=k)
        center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


def xy_dense_knn_matrix(x, y, k=16):
    with torch.no_grad():
        x = x.transpose(2, 1).squeeze(-1)
        y = y.transpose(2, 1).squeeze(-1)
        batch_size, n_points, n_dims = x.shape
        dist = xy_pairwise_distance(x.detach(), y.detach())
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


class ViG(torch.nn.Module):
    def __init__(self):
        super(ViG, self).__init__()
        k = 9
        blocks = [2, 2, 6, 2]
        channels = [48, 96, 240, 384]
        reduce_ratios = [4, 2, 1, 1]
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224 // 4, 224 // 4))
        self.stem = Stem(out_dim=channels[0])
        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):
            if i > 0:
                self.backbone.append(Downsample(channels[i - 1], channels[i]))
            for j in range(blocks[i]):
                self.backbone += [
                    Seq(Grapher(channels[i], k, idx // 4 + 1, reduce_ratios[i]),
                        FFN(channels[i], channels[i] * 4))]
                idx += 1
        self.backbone = Seq(*self.backbone)
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
        inputs = inputs.repeat(1, 3, 1, 1)
        x = self.stem(inputs) + self.pos_embed
        feature_list = []
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)
            if i in [1, 4, 11, 14]:
                feature_list.append(x)
        return feature_list


class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()
        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim), nn.ReLU(),
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim), nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()
        self.upsample = nn.ConvTranspose2d(input_dim, output_dim, kernel_size=kernel, stride=stride)

    def forward(self, x):
        return self.upsample(x)


class ViGUNet(nn.Module):
    def __init__(self):
        super(ViGUNet, self).__init__()
        filters = [48, 96, 240, 384]
        self.Encoder = ViG()

        self.bridge = None

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)
        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)
        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)
        self.output_reg_layer = nn.Sequential(
            Conv2d(filters[0], filters[0], 3, 1, 1), nn.BatchNorm2d(filters[0]), nn.ReLU(),
            Upsample(filters[0], filters[0]//2, 2, 2), nn.BatchNorm2d(filters[0]//2), nn.ReLU(),
            Conv2d(filters[0]//2, filters[0]//2, 3, 1, 1), nn.BatchNorm2d(filters[0]//2), nn.ReLU(),
            Upsample(filters[0]//2, filters[0]//2, 2, 2),
            Conv2d(filters[0] // 2, 1, 1, 1)
        )

    def forward(self, x):
        # Encode
        f_list = self.Encoder(x)
        # 80x56x56, 160x28x28, 400x14x14, 640x7x7
        x1, x2, x3, x4 = f_list[0], f_list[1], f_list[2], f_list[3]
        # Bridge
        if self.bridge is not None:
            x_bridge = self.bridge(x4)
        else:
            x_bridge = x4
        # Decode for regression
        x4 = self.upsample_1(x_bridge)
        x5 = torch.cat([x4, x3], dim=1)
        x6 = self.up_residual_conv1(x5)
        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)
        x8 = self.up_residual_conv2(x7)
        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)
        x10 = self.up_residual_conv3(x9)
        out_reg = self.output_reg_layer(x10)
        out_reg = torch.nn.ReLU()(out_reg)

        return [x1, x2, x3, x_bridge], out_reg