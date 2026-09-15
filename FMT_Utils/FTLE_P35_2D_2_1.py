"""P35 semantic pooling and signed five-line Fourier features for FTLE SR."""
import torch

VERSION='2.1'
WIDTHS={'none':0,'raw':321,'p35':103,'signed':121,'p35_signed':223,'p35_multitime':667}


def descriptor(delta, frequencies=6):
    spectrum=torch.fft.rfft(delta,dim=-2)[...,:frequencies,:]
    a,b=spectrum.real,spectrum.imag
    an,bn=a.norm(dim=-1),b.norm(dim=-1)
    cosine=(a*b).sum(-1)/(an*bn).clamp_min(1e-8)
    # A 3D scalar triple product is identically zero in 2D; omit those slots.
    return torch.stack((an,bn,cosine),-1).flatten(-2)


def window(q, relative, start, stop, frequencies=6):
    center=q[:,0,start:stop]
    neighbors=relative[:,:,start:stop]
    delta=center.diff(dim=-2)
    descriptors=descriptor(neighbors.diff(dim=-2),frequencies)
    tangent=delta/delta.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    tangent=torch.cat((tangent,tangent[:,-1:]),-2)
    direction=torch.view_as_real(torch.fft.rfft(torch.cat((center,tangent),-1),
                                               dim=-2,norm='ortho')[:,:frequencies]).flatten(1)
    p35=torch.cat((descriptor(delta,frequencies),direction,
                   descriptors.mean(1),descriptors.amax(1)),-1)
    sequences=torch.cat((delta[:,None],neighbors.diff(dim=-2)),1)
    signed=torch.view_as_real(torch.fft.rfft(sequences,dim=-2)[:,:,:frequencies]).flatten(1)
    return p35,signed


@torch.no_grad()
def encode(paths):
    x=torch.as_tensor(paths,dtype=torch.float64)
    if x.ndim!=4 or x.shape[1:]!=(5,32,2) or not torch.isfinite(x).all():
        raise ValueError('Require finite [N,5,32,2] physical paths; time is not z.')
    center=x.mean((1,2),keepdim=True)
    radius=(x-center).norm(dim=-1).amax((1,2))
    relative=x[:,1:]-x[:,:1]
    initial_radius=relative[:,:,0].norm(dim=-1).mean(1)
    if torch.any(radius<=1e-12) or torch.any(initial_radius<=1e-12):
        raise ValueError('Degenerate primitive')
    q=(x-center)/radius[:,None,None,None]
    relative=relative/initial_radius[:,None,None,None]
    scale=torch.log(radius/initial_radius)[:,None]
    p,s=window(q,relative,0,32)
    early=window(q,relative,0,17)
    late=window(q,relative,15,32)
    raw=torch.cat((q[:,:1]-q[:,:1,:1],relative),1).flatten(1)
    features={'raw':torch.cat((raw,scale),-1),'p35':torch.cat((p,scale),-1),
              'signed':torch.cat((s,scale),-1),'p35_signed':torch.cat((p,s,scale),-1),
              'p35_multitime':torch.cat((p,s,*early,*late,scale),-1)}
    for key,value in features.items():
        if value.shape[1]!=WIDTHS[key] or not torch.isfinite(value).all():
            raise ValueError('Invalid feature width or values')
    return {key:value.float() for key,value in features.items()}
