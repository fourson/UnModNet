import torch
import torch.nn as nn

from .layer_utils.unet import AttentionUnetBackbone
from .layer_utils.se_block import SEBlock
from .layer_utils.region_non_local_block import RegionNonLocalEnhancedDenseBlock
from base.base_model import BaseModel


class DefaultModel(BaseModel):
    """
        Define the network to learn the binary mask
        x1:input_nc  x2:input_nc  -->  input_nc
        output: scores (we shold use sigmoid() and round() to get the binary mask in trainer)
    """

    def __init__(self, input_nc, init_dim=64, n_downsampling=4, use_conv_to_downsample=True, norm_type='instance',
                 use_dropout=False, mode='res-bottleneck'):
        super(DefaultModel, self).__init__()

        self.modulo_feature_extraction = nn.Sequential(
            nn.Conv2d(input_nc, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(init_dim // 2),
            nn.ReLU(True)
        )
        self.edge_feature_extraction = nn.Sequential(
            nn.Conv2d(input_nc, init_dim // 2, kernel_size=7, stride=1, padding=3, bias=True),
            nn.InstanceNorm2d(init_dim // 2),
            nn.ReLU(True),
            RegionNonLocalEnhancedDenseBlock(in_channel=init_dim // 2, inter_channel=init_dim // 4, n_blocks=3,
                                             latent_dim=2, subsample=True, grid=(8, 8))
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(init_dim, init_dim, kernel_size=1, stride=1),
            SEBlock(init_dim, 8)
        )
        self.backbone = AttentionUnetBackbone(init_dim, output_nc=init_dim, n_downsampling=n_downsampling,
                                              use_conv_to_downsample=use_conv_to_downsample, norm_type=norm_type,
                                              use_dropout=use_dropout, mode=mode)
        self.out_block = nn.Sequential(
            nn.Conv2d(init_dim, input_nc, kernel_size=1, stride=1)
        )

    def forward(self, x1, x2):
        # x1: (N, input_nc, H, W) input modulo img([0, 1] float, as float32)
        # x2: (N, input_nc, H, W) fold number edge of input modulo img(binary, as float32)
        modulo_feature = self.modulo_feature_extraction(x1)
        edge_feature = self.edge_feature_extraction(x2)
        fusion_out = self.fusion(torch.cat([modulo_feature, edge_feature], dim=1))
        backbone_out = self.backbone(fusion_out)
        out = self.out_block(backbone_out)
        return out
