from functools import partial
from typing import Optional

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

# Copypasted from https://github.com/BloodAxe/pytorch-toolbelt

__all__ = ["FocalLoss"]

BINARY_MODE = "binary"
MULTICLASS_MODE = "multiclass"


def focal_loss_with_logits(
        input: torch.Tensor,
        target: torch.Tensor,
        gamma=2.0,
        alpha: Optional[float] = 0.25,
        reduction="mean",
        normalized=False,
        reduced_threshold: Optional[float] = None,
) -> torch.Tensor:
    """Compute binary focal loss between target and output logits.
    See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
    Args:
        input: Tensor of arbitrary shape
        target: Tensor of the same shape as input
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
        normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
        reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
    References::
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
    """
    target = target.type(input.type())

    logpt = F.binary_cross_entropy_with_logits(input, target, reduction="none")
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum() + 1e-5
        loss /= norm_factor

    if reduction == "mean":
        loss = loss.mean()
    if reduction == "sum":
        loss = loss.sum()
    if reduction == "batchwise_mean":
        loss = loss.sum(0)

    return loss


class FocalLoss(_Loss):
    def __init__(self, mode=MULTICLASS_MODE, alpha=None, gamma=2, ignore_index=None,
                 reduction="mean", normalized=False, reduced_threshold=None):
        """
        Focal loss for multi-class problem.
        :param mode: Metric mode {'binary', 'multiclass'}
        :param alpha:
        :param gamma:
        :param ignore_index: If not None, targets with given index are ignored
        :param reduced_threshold: A threshold factor for computing reduced focal loss
        """
        super().__init__()
        self.mode = mode
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction=reduction,
            normalized=normalized,
        )

    def forward(self, input, target):
        if self.mode == BINARY_MODE:
            target = target.view(-1)
            input = input.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = target != self.ignore_index
                input = input[not_ignored]
                target = target[not_ignored]

            loss = self.focal_loss_fn(input, target)
            return loss
        elif self.mode == MULTICLASS_MODE:
            num_classes = input.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = target != self.ignore_index

            for cls in range(num_classes):
                cls_label_target = (target == cls).long()
                cls_label_input = input[:, cls, ...]

                if self.ignore_index is not None:
                    cls_label_target = cls_label_target[not_ignored]
                    cls_label_input = cls_label_input[not_ignored]

                loss += self.focal_loss_fn(cls_label_input, cls_label_target)
            return loss
        else:
            raise NotImplementedError("Mode you are asking for is not implemented.")
