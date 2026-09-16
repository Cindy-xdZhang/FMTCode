"""Shrink the frozen per-line BiLSTM and MLP to approximately 56,786 total parameters."""
from __future__ import annotations
import torch
from torch import nn
from FMT_Utils.Task4C_GeometricBaselines_1_1 import mlp

PARAMETERS = 56753
RECURRENT_PARAMETERS = 43168
CLASSIFIER_PARAMETERS = 13585


class BundleBiLSTMSmall(nn.Module):
    """Same one-layer bidirectional sequence model and symmetric line pooling, smaller widths."""
    def __init__(self, dropout=.15):
        super().__init__()
        self.encoder = nn.LSTM(3,71,batch_first=True,bidirectional=True)
        self.classifier = mlp((284,47,2),dropout)

    def forward(self, batch):
        geometry,counts=batch
        b,lines,points,_=geometry.shape
        _,(hidden,_)=self.encoder(geometry.reshape(b*lines,points,3))
        features=hidden.transpose(0,1).reshape(b,lines,142)
        valid=torch.arange(lines,device=geometry.device)[None,:,None]<counts[:,None,None]
        average=torch.where(valid,features,0).sum(1)/counts[:,None]
        maximum=features.masked_fill(~valid,-torch.inf).max(1).values
        return self.classifier(torch.cat((average,maximum),-1))


def parameter_counts(model):
    recurrent=sum(p.numel() for p in model.encoder.parameters())
    classifier=sum(p.numel() for p in model.classifier.parameters())
    total=sum(p.numel() for p in model.parameters())
    assert (recurrent,classifier,total)==(RECURRENT_PARAMETERS,CLASSIFIER_PARAMETERS,PARAMETERS)
    assert total==recurrent+classifier and total<=56786
    return dict(recurrent=recurrent,classifier=classifier,total=total)


def make_model(method='baseline2',dropout=.15):
    assert method=='baseline2'
    model=BundleBiLSTMSmall(dropout);parameter_counts(model)
    return model
