"""Matched low-grid residual decoders with optional Fourier geometry."""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self,width,dilation=1):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(width,width,3,padding=dilation,dilation=dilation),nn.SiLU(),
                               nn.Conv2d(width,width,3,padding=dilation,dilation=dilation))
    def forward(self,x):return x+.1*self.net(x)


class FusionSR(nn.Module):
    def __init__(self,scale,architecture='pyramid',width=48,padded=672):
        super().__init__()
        self.scale,self.architecture,self.padded=scale,architecture,padded
        self.scalar=nn.Conv2d(2,width,3,padding=1)
        self.geometry=nn.Conv2d(padded,width,1)
        self.first=ResidualBlock(width)
        self.context=nn.Sequential(nn.Conv2d(2*width,width,1),nn.SiLU(),nn.Conv2d(width,width,1))
        if architecture=='pyramid':
            self.blocks=nn.Sequential(ResidualBlock(width,2),ResidualBlock(width,4),ResidualBlock(width))
        elif architecture=='unet':
            self.down1=nn.Sequential(nn.Conv2d(width,2*width,3,padding=1),nn.SiLU(),ResidualBlock(2*width))
            self.down2=nn.Sequential(nn.Conv2d(2*width,4*width,3,padding=1),nn.SiLU(),ResidualBlock(4*width))
            self.up1=nn.Sequential(nn.Conv2d(6*width,2*width,3,padding=1),nn.SiLU(),ResidualBlock(2*width))
            self.up2=nn.Sequential(nn.Conv2d(3*width,width,3,padding=1),nn.SiLU(),ResidualBlock(width))
        else:raise ValueError(architecture)
        self.head=nn.Conv2d(width,scale*scale,3,padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,low,geometry,valid,bicubic,low_target):
        if geometry is None:
            geometry=low.new_zeros((len(low),self.padded,*low.shape[-2:]))
        else:
            geometry=F.pad(geometry,(0,0,0,0,0,self.padded-geometry.shape[1]))
        x=self.first(self.scalar(torch.cat((low,valid),1))+self.geometry(geometry))
        mean=(x*valid).sum((-2,-1),keepdim=True)/valid.sum((-2,-1),keepdim=True).clamp_min(1)
        maximum=x.masked_fill(valid==0,-torch.inf).amax((-2,-1),keepdim=True)
        x=x+.1*self.context(torch.cat((mean,maximum),1))
        if self.architecture=='pyramid':
            x=self.blocks(x)
        else:
            x1=self.down1(F.avg_pool2d(x,2))
            x2=self.down2(F.avg_pool2d(x1,2))
            y=self.up1(torch.cat((F.interpolate(x2,size=x1.shape[-2:],mode='bilinear',align_corners=True),x1),1))
            x=self.up2(torch.cat((F.interpolate(y,size=x.shape[-2:],mode='bilinear',align_corners=True),x),1))
        residual=F.pixel_shuffle(self.head(x),self.scale)[...,:bicubic.shape[-2],:bicubic.shape[-1]]
        output=bicubic+residual
        # Known physical FTLE samples are exact observations, shared by all new arms.
        output=output.clone()
        output[...,::self.scale,::self.scale]=low_target
        return output
