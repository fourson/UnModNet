import torch
import torch.nn.functional as F

# should be torch.ones([1, 256, 256]).cuda() while training with the grayscale images
pos_weight_dummy = torch.ones([3, 256, 256]).cuda()


def bce_with_logits(output, target, **kwargs):
    pos_weight = kwargs.get('pos_weight', 1) * pos_weight_dummy
    loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
    return loss
