"""
Code is referenced from
https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/vig_pytorch/pyramid_vig.py

"""

import torch.nn as nn
import torch
from End2End_EGDCXR.models.GAViG.vigunet import ViGUNet
from End2End_EGDCXR.models.GAViG.vig import ViG_Gaze


class GAViG(nn.Module):
    def __init__(self, num_classes):
        super(GAViG, self).__init__()
        self.vig = ViG_Gaze(num_classes=num_classes)
        self.vig.prediction[-1] = nn.Conv2d(self.vig.prediction[-1].in_channels, 3, kernel_size=(1, 1), stride=(1, 1))
        m = self.vig.prediction[-1]
        torch.nn.init.kaiming_normal_(m.weight)
        m.weight.requires_grad = True
        if m.bias is not None:
            m.bias.data.zero_()
            m.bias.requires_grad = True

        self.vigunet = ViGUNet()
        self.gaze = None

    def forward(self, x):
        _, gaze = self.vigunet(x)
        self.gaze = gaze
        pred = self.vig(torch.stack([x, gaze], dim=0))
        return pred

    def get_gaze(self):
        return self.gaze