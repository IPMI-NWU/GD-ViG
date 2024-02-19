"""
Code is referenced from
https://github.com/huawei-noah/Efficient-AI-Backbones/blob/master/vig_pytorch/pyramid_vig.py

"""

import torch.nn as nn
import torch
from End2End_SIIMACR.models.GAViG.vigunet import ViGUNet
from End2End_SIIMACR.models.GAViG.vig import ViG_Gaze


class GAViG(nn.Module):
    def __init__(self, num_classes):
        super(GAViG, self).__init__()
        self.vig = ViG_Gaze(num_classes=num_classes)
        self.vigunet = ViGUNet()
        self.gaze = None

    def forward(self, x):
        _, gaze = self.vigunet(x)
        self.gaze = gaze
        pred = self.vig(torch.stack([x, gaze], dim=0))
        return pred

    def get_gaze(self):
        return self.gaze