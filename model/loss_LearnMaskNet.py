import torch
import torch.nn.functional as F

from .loss_utils.total_variation_loss import TVLoss

# should be torch.ones([1, 256, 256]).cuda() while training with the grayscale images
pos_weight_dummy = torch.ones([3, 256, 256]).cuda()

def bce_with_logits(output, target, **kwargs):
    pos_weight = kwargs.get('pos_weight', 1) * pos_weight_dummy
    loss = F.binary_cross_entropy_with_logits(output, target, pos_weight=pos_weight)
    return loss


def bce_with_logits_and_tv(output, target, **kwargs):
    bce_loss_lambda = kwargs.get('bce_loss_lambda', 1)
    bce_loss = F.binary_cross_entropy_with_logits(output, target)
    tv_loss_func = TVLoss(TVLoss_weight=kwargs.get('TVLoss_weight', 1))
    tv_loss = bce_loss + tv_loss_func(F.sigmoid(output))
    print('\nbce_loss:', bce_loss.item() * bce_loss_lambda)
    print('tv_loss :', tv_loss.item())
    return bce_loss * bce_loss_lambda + tv_loss
