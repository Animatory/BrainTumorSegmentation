import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor as T

__all__ = ['CrossEntropyLoss', 'WeightedBCEWithLogitsLoss', 'WeightedCrossEntropyLoss',
           'MultiScaleCrossEntropyLoss', 'LabelSmoothingCrossEntropyLoss', 'SoftTargetCrossEntropyLoss',
           'OhemCrossEntropyLoss', 'BCEWithLogitsLoss', 'SelfCrossEntropyLoss', 'SelfBCELoss']


class CrossEntropyLoss(nn.CrossEntropyLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float)
        super(CrossEntropyLoss, self).__init__(weight=weight, ignore_index=ignore_index, reduction=reduction)


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, reduction='mean', pos_weight=None, ignore_all_zeros=False):
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)
        super().__init__(weight=weight, reduction=reduction, pos_weight=pos_weight)
        self.ignore_all_zeros = ignore_all_zeros

    def forward(self, input, target):
        if self.ignore_all_zeros and target.ndim == 4:
            non_zeros = target.sum(dim=1) > 0
            target = target[non_zeros]
            input = input[non_zeros]
        return F.binary_cross_entropy_with_logits(input, target.float(),
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)


class SelfCrossEntropyLoss(nn.Module):
    def __init__(self, confidence_thresh=0.96837722):
        super().__init__()
        self.confidence_thresh = confidence_thresh

    @staticmethod
    def robust_binary_crossentropy(pred, tgt):
        return -(tgt * torch.log(pred + 1.0e-6) + (1.0 - tgt) * torch.log(1.0 - pred + 1e-6))

    def forward(self, input: T, target: T) -> T:
        input = F.softmax(input, dim=1)
        target = F.softmax(target, dim=1)
        conf_tea = torch.max(target, 1)[0]
        unsup_mask = (conf_tea > self.confidence_thresh).float()

        aug_loss = self.robust_binary_crossentropy(input, target)
        aug_loss = aug_loss.mean(dim=1)
        unsup_loss = (aug_loss * unsup_mask).mean()

        return unsup_loss


class SelfBCELoss(nn.Module):
    def __init__(self, confidence_thresh=0.96837722):
        super().__init__()
        self.confidence_thresh = confidence_thresh

    @staticmethod
    def robust_binary_crossentropy(pred, tgt):
        return -(tgt * torch.log(pred + 1.0e-6) + (1.0 - tgt) * torch.log(1.0 - pred + 1e-6))

    def forward(self, input: T, target: T) -> T:
        input = torch.sigmoid(input)
        target = torch.sigmoid(target)
        input = torch.stack([1 - input, input], 1)
        target = torch.stack([1 - target, target], 1)
        conf_tea = torch.max(target, 1)[0]
        unsup_mask = (conf_tea > self.confidence_thresh).float()

        aug_loss = self.robust_binary_crossentropy(input, target)
        aug_loss = aug_loss.mean(dim=1)
        unsup_loss = (aug_loss * unsup_mask).mean()

        return unsup_loss


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=weight, ignore_index=ignore_index)
        self.reduction = reduction

    def forward(self, input: T, target: T, weight: T = None, **kwargs) -> T:
        loss = self.ce(input, target)
        if weight is not None:
            loss = loss * weight
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', pos_weight=None):
        super().__init__()
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)

        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input: T, target: T, weights: T = None):
        loss = F.binary_cross_entropy_with_logits(input, target.float(), self.weight,
                                                  pos_weight=self.pos_weight, reduction='none')
        if weights is not None:
            loss = loss * weights
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiScaleCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, scale_factor=1, ignore_index=-100, reduction='mean'):
        super(MultiScaleCrossEntropyLoss, self).__init__()
        self.ce = WeightedCrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction=reduction)
        self.reduction = reduction
        self.scale = scale_factor

    def forward(self, input: T, target: T, weight: T = None, **kwargs) -> T:
        if not isinstance(input, tuple):
            return self.cross_entropy2d(input, target, weight)
        else:
            n_inp = len(input)
            # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
            scale_weight = torch.pow(self.scale * torch.ones(n_inp), torch.arange(n_inp).float()).to(target.device)
            loss = sum([scale_weight[i] * self.cross_entropy2d(inp, target, weight) for i, inp in enumerate(input)])
            return loss / scale_weight.sum()

    def cross_entropy2d(self, input: T, target: T, weight=None) -> T:
        n, c, h, w = input.size()
        nt, ht, wt = target.size()

        # Handle inconsistent size between input and target
        if h != ht and w != wt:  # upsample labels
            input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)

        return self.ce.forward(input, target, weight)


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, input: T, target: T) -> T:
        logprobs = F.log_softmax(input, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropyLoss(nn.Module):

    def __init__(self):
        super(SoftTargetCrossEntropyLoss, self).__init__()

    def forward(self, input: T, target: T) -> T:
        loss = torch.sum(-target * F.log_softmax(input, dim=-1), dim=-1)
        return loss.mean()


class OhemCrossEntropyLoss(nn.Module):
    """
    Online bootstrapping of hard training pixels
    paper: https://arxiv.org/abs/1604.04339
    """

    def __init__(self, thres=0.7, min_kept=100000, ignore_label=-1, balance_weight=None, weight=None):
        super(OhemCrossEntropyLoss, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=ignore_label,
            reduction='none'
        )
        self.balance_weight = balance_weight

    def _ce_forward(self, input: T, target: T) -> T:
        ph, pw = input.size(2), input.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            input = F.interpolate(input=input, size=(
                h, w), mode='bilinear', align_corners=True)

        loss = self.criterion(input, target)

        return loss

    def _ohem_forward(self, input: T, target: T) -> T:
        ph, pw = input.size(2), input.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            input = F.interpolate(input=input, size=(
                h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input, dim=1)
        pixel_losses = self.criterion(input, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.view(-1, )[mask].sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

    def forward(self, input: T, target: T) -> T:

        if not isinstance(input, tuple):
            return self._ohem_forward(input, target)

        loss = self._ohem_forward(input[0], target)
        loss += self._ce_forward(input[1], target)
        return loss
