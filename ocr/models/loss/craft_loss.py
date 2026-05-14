from collections import OrderedDict
import torch.nn as nn
from .bce_loss import BCELoss
from .dice_loss import DiceLoss


class CRAFTLoss(nn.Module):
    def __init__(self, negative_ratio=3.0, eps=1e-6, region_weight=1.0):
        super(CRAFTLoss, self).__init__()
        self.dice_loss = DiceLoss(eps)
        self.bce_loss = BCELoss(negative_ratio, eps)
        self.region_weight = region_weight

    def forward(self, pred, **kwargs):
        pred_prob = pred['prob_maps']
        gt_prob_maps = kwargs.get('prob_maps')
        gt_prob_mask = kwargs.get('prob_mask')

        loss_dice = self.dice_loss(pred_prob, gt_prob_maps, gt_prob_mask)
        loss_bce = self.bce_loss(pred_prob, gt_prob_maps, gt_prob_mask)
        loss = self.region_weight * (loss_dice + loss_bce)

        return loss, OrderedDict(loss_dice=loss_dice, loss_bce=loss_bce)
