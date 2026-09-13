"""Fixed 4.3 networks with training-only interpolation of hidden representations."""
from __future__ import annotations

import torch
from torch import nn

from FMT_Utils.Task4C_LinePooling_4_3 import make_model as base_model
from FMT_Utils.Task4C_Regularized_4_2 import classifier_loss, classifier_probability


class MixupClassifier(nn.Module):
    def __init__(self, method, candidate):
        super().__init__()
        if candidate["architecture"] != "learned_pool_wider_conv_4.3":
            raise ValueError("4.4 freezes the 4.3 learned-pooling/wider-convolution models")
        self.method = method
        self.base = base_model(method, candidate)

    def forward(self, x):
        return self.base(x)

    def mixed_logits(self, x, permutation, coefficient):
        """Interpolate after the 64D hidden layer; infer on original samples."""
        if self.method == "fmt_mlp":
            mask = x[..., -1] > .5
            features = self.base.line(x[..., :-1])
            mean = (features*mask[..., None]).sum(1)/mask.sum(1, keepdim=True)
            maximum = features.masked_fill(~mask[..., None], -torch.inf).amax(1)
            hidden = self.base.head[:-1](torch.cat((mean, maximum), -1))
            head = self.base.head[-1]
        else:
            hidden = self.base.network[:-1](x)
            head = self.base.network[-1]
        mixed = coefficient*hidden + (1-coefficient)*hidden[permutation]
        return head(mixed)


def make_model(method, candidate):
    return MixupClassifier(method, candidate)


def training_loss(model, x, labels, candidate, positive_weight, mix_rng):
    """A full-batch permutation preserves the weighted CE denominator."""
    alpha = candidate["mixup_alpha"]
    if alpha == 0:
        return classifier_loss(model(x), labels, candidate, positive_weight)
    coefficient = float(mix_rng.beta(alpha, alpha))
    permutation = torch.as_tensor(mix_rng.permutation(len(x)), device=x.device)
    logits = model.mixed_logits(x, permutation, coefficient)
    return (coefficient*classifier_loss(logits, labels, candidate, positive_weight)
            + (1-coefficient)*classifier_loss(logits, labels[permutation], candidate, positive_weight))
