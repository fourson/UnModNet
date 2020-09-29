import torch
import torch.nn as nn

from .layer_utils.resnet import NonLocalResnetBackbone
from base.base_model import BaseModel


class DefaultModel(BaseModel):
    """
        Define the network to learn the fold_number_edge
        x1:input_nc  x2:input_nc  -->  input_nc
        output: scores (we shold use sigmoid() and round() to get the fold_number_edge in trainer)
    """

    def __init__(self, input_nc, init_dim=64, n_downsampling=2, n_blocks=6, norm_type='instance', use_dropout=False,
                 mode='residual'):
        super(DefaultModel, self).__init__()

        self.mode = mode

        if self.mode == 'residual':
            self.backbone = nn.Sequential(
                NonLocalResnetBackbone(input_nc * 2, output_nc=init_dim, n_downsampling=n_downsampling,
                                       n_blocks=n_blocks,
                                       norm_type=norm_type, use_dropout=use_dropout),
                nn.Conv2d(init_dim, input_nc, kernel_size=7, stride=1, padding=3),
                nn.Tanh()
            )
        elif self.mode == 'end2end':
            self.backbone = nn.Sequential(
                NonLocalResnetBackbone(input_nc, output_nc=init_dim, n_downsampling=n_downsampling,
                                       n_blocks=n_blocks,
                                       norm_type=norm_type, use_dropout=use_dropout),
                nn.Conv2d(init_dim, input_nc, kernel_size=7, stride=1, padding=3),
                nn.Tanh()
            )
        elif self.mode == 'end2end_without_tanh':
            self.backbone = nn.Sequential(
                NonLocalResnetBackbone(input_nc, output_nc=init_dim, n_downsampling=n_downsampling,
                                       n_blocks=n_blocks,
                                       norm_type=norm_type, use_dropout=use_dropout),
                nn.Conv2d(init_dim, input_nc, kernel_size=7, stride=1, padding=3),
            )
        else:
            raise NotImplementedError('mode [%s] is not found' % self.mode)

    def forward(self, x1, x2):
        # x1: (N, input_nc, H, W) input modulo img([0, 1] float, as float32)
        # x2: (N, input_nc, H, W) laplacian of input modulo img([0, 1] float, as float32)
        if self.mode == 'residual':
            out = self.backbone(torch.cat([x1, x2], dim=1)) + x2
        else:
            out = self.backbone(x1)
        return out
