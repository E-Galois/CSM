import torch
import torch.nn as nn
import torch.nn.functional as F


class CeL2Loss(nn.Module):
    def __init__(self, num_classes: int = 100,
                 ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.cross_entropy = nn.CrossEntropyLoss()

    @property
    def names(self):
        return "loss", "loss_ce", "loss_mixup"

    def l2_loss(self, p, q):
        return (p - q).square().sum(-1).mean()

    def forward(self, inputs, targets, mixup, target_re):
        loss_ce = 0
        loss_mixup = 0
        if inputs is not None:
            if inputs.dim() > 2:
                inputs = inputs.view(inputs.size(0), inputs.size(1), -1)  # N,C,H,W => N,C,H*W
                inputs = inputs.transpose(1, 2)  # N,C,H*W => N,H*W,C
                inputs = inputs.contiguous().view(-1, inputs.size(2))  # N,H*W,C => N*H*W,C
                targets = targets.view(-1)

            if self.ignore_index >= 0:
                index = torch.nonzero(targets != self.ignore_index).squeeze()
                inputs = inputs[index, :]
                targets = targets[index]
            loss_ce = self.cross_entropy(inputs, targets)
        if mixup is not None:
            mixup = mixup.flatten(0, 1)
            target_re = target_re.flatten(0, 1)
            mixup = torch.log_softmax(mixup, dim=-1)
            mixup = torch.exp(mixup)
            loss_mixup = self.l2_loss(mixup, target_re)
        else:
            loss_mixup = 0.0
        loss = loss_ce + loss_mixup
        return loss, loss_ce, loss_mixup
