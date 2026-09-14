"""Version 8.2: fixed-source neighbor pooling and explicit search recipes."""
from __future__ import annotations

import copy
import numpy as np
import torch
from torch import nn

from FMT_Utils.fmt_v8 import coordinate_tangent_spectrum
from FMT_Utils.Task4C_Encoders_4_15 import EncoderLinePooling
from FMT_Utils.PathlineClassifier_3D import _BlockwiseAuxiliaryProjection


def pooling_candidates():
    result = []
    counts = [4, 1, 8, 16, 24, 32, 48, 64, 96, 138]
    for kind in ('absolute_top', 'signed_extremes', 'indexed_top'):
        for k in counts:
            result.append(dict(kind=kind, k=k, pooled_dimensions=138 if kind == 'indexed_top' else k+1))
    for statistics in [('mean',), ('max',), ('min',), ('std',), ('rms',),
                       ('mean','max'), ('mean','std'), ('min','max'),
                       ('mean','max','std'), ('mean','min','max','std')]:
        result.append(dict(kind='semantic_statistics', statistics=list(statistics), pooled_dimensions=23*len(statistics)))
    for ranks in [[0], [0,5], [0,2,5], [0,1,4,5], [0,1,2,4,5], list(range(6))]:
        result.append(dict(kind='semantic_ranks', ranks=ranks, pooled_dimensions=23*len(ranks)))
    for k in (1,2,3,4):
        result.append(dict(kind='semantic_absolute_top', k=k, pooled_dimensions=23*k))
    for i, item in enumerate(result):
        item.update(id=f'p{i:02d}', feature_dimensions=95+item['pooled_dimensions'])
    assert len(result) == 50
    return result


PROFILES = {
    'h0': dict(description='Frozen baseline training and projection',
               residual_model={}, residual_training={}, task4={}),
    'h1': dict(description='Stronger dropout and weight decay',
               residual_model=dict(auxiliary_dropout=.15), residual_training=dict(weight_decay=.001),
               task4=dict(dropout=.30, weight_decay=.001)),
    'h2': dict(description='SiLU activation with lower learning rate',
               residual_model=dict(auxiliary_projection_activation_override='silu', auxiliary_dropout=.05),
               residual_training=dict(learning_rate=.0005),
               task4=dict(activation='silu', learning_rate=.0005)),
    'h3': dict(description='Separate learned projections of center, direction and pooled neighbors',
               residual_model=dict(auxiliary_projection='blockwise_layernorm_gelu',
                                   auxiliary_block_dims=[23,72,338]), residual_training={},
               task4=dict(blockwise=True)),
}


def pool_neighbors(values, definition):
    """Input is rank-major [6,23], already independently sorted per descriptor."""
    if values.shape[-1] != 138:
        raise ValueError('Expected 138 frozen neighbor descriptors')
    kind = definition['kind']
    if kind in ('absolute_top','indexed_top'):
        ids = torch.argsort(values.abs(), dim=-1, descending=True, stable=True)[..., :definition['k']]
        selected = torch.gather(values, -1, ids)
        if kind == 'indexed_top':
            return torch.zeros_like(values).scatter(-1, ids, selected)
        return torch.cat((selected, values.mean(-1,keepdim=True)), -1)
    if kind == 'signed_extremes':
        k = definition['k']; ordered = values.sort(dim=-1, descending=True, stable=True).values
        selected = torch.cat((ordered[..., :(k+1)//2], ordered[..., 138-k//2:]), -1)
        return torch.cat((selected, values.mean(-1,keepdim=True)), -1)
    grouped = values.reshape(*values.shape[:-1], 6, 23)
    if kind == 'semantic_ranks':
        return grouped[..., definition['ranks'], :].flatten(-2)
    if kind == 'semantic_absolute_top':
        ids = torch.argsort(grouped.abs(), dim=-2, descending=True, stable=True)[..., :definition['k'], :]
        return torch.gather(grouped, -2, ids).flatten(-2)
    if kind == 'semantic_statistics':
        operations = {'mean':lambda:grouped.mean(-2), 'max':lambda:grouped.amax(-2),
                      'min':lambda:grouped.amin(-2), 'std':lambda:grouped.std(-2,unbiased=False),
                      'rms':lambda:grouped.square().mean(-2).sqrt()}
        return torch.cat([operations[name]() for name in definition['statistics']], -1)
    raise ValueError(kind)


def pool_numpy(values, definition):
    """Independent reference for every pooling family, including stable ties."""
    kind = definition['kind']
    if kind in ('absolute_top','indexed_top'):
        ids=np.argsort(-np.abs(values),axis=-1,kind='stable')[..., :definition['k']]
        chosen=np.take_along_axis(values,ids,-1)
        if kind=='indexed_top':
            result=np.zeros_like(values);np.put_along_axis(result,ids,chosen,-1);return result
        return np.concatenate((chosen,values.mean(-1,keepdims=True)),-1)
    if kind=='signed_extremes':
        k=definition['k'];ordered=np.sort(values,axis=-1)[...,::-1]
        return np.concatenate((ordered[..., :(k+1)//2],ordered[..., 138-k//2:],values.mean(-1,keepdims=True)),-1)
    grouped=values.reshape(*values.shape[:-1],6,23)
    if kind=='semantic_ranks':return grouped[...,definition['ranks'],:].reshape(*values.shape[:-1],-1)
    if kind=='semantic_absolute_top':
        ids=np.argsort(-np.abs(grouped),axis=-2,kind='stable')[...,:definition['k'],:]
        return np.take_along_axis(grouped,ids,-2).reshape(*values.shape[:-1],-1)
    operations={'mean':lambda:grouped.mean(-2),'max':lambda:grouped.max(-2),
                'min':lambda:grouped.min(-2),'std':lambda:grouped.std(-2),
                'rms':lambda:np.sqrt(np.mean(grouped**2,axis=-2))}
    return np.concatenate([operations[name]() for name in definition['statistics']],-1)


@torch.no_grad()
def select_features(original233, definition):
    x=torch.as_tensor(original233)
    if x.shape[-1]!=233:raise ValueError('No input beyond frozen 233 geometry features is allowed')
    result=torch.cat((x[...,:23],x[...,161:233],pool_neighbors(x[...,23:161],definition)),-1)
    assert result.shape[-1]==definition['feature_dimensions'] and torch.isfinite(result).all()
    return result


@torch.no_grad()
def pathline_source(raw, cached161):
    """Exactly the previous 233D input; avoid computing the unused angle branch."""
    raw=np.asarray(raw,dtype=np.float32);cached161=np.asarray(cached161,dtype=np.float32)
    assert raw.shape[1:]==(7,32,3) and cached161.shape==(len(raw),161)
    output=np.empty((len(raw),233),np.float32);output[:,:161]=cached161
    for start in range(0,len(raw),1024):
        sl=slice(start,start+1024);x=torch.from_numpy(raw[sl]).double()
        x=x-x.mean((1,2),keepdim=True);radius=x.norm(dim=-1).amax((1,2),keepdim=True)[...,None]
        assert torch.all(radius>0)
        output[sl,161:]=coordinate_tangent_spectrum((x/radius).float()[:,0]).numpy()
    assert np.isfinite(output).all()
    return output


def task4_model(width, profile):
    options=PROFILES[profile]['task4'];model=EncoderLinePooling(width,options.get('dropout',.15))
    if options.get('blockwise'):
        model.line=nn.Sequential(_BlockwiseAuxiliaryProjection(width,128,[23,72,width-95],
            'blockwise_layernorm_gelu'),*list(model.line.children())[3:])
    if options.get('activation')=='silu':
        def replace(parent):
            for name,child in tuple(parent.named_children()):
                if isinstance(child,nn.GELU):setattr(parent,name,nn.SiLU())
                else:replace(child)
        replace(model)
    return model


def residual_spec(base, task, profile, raw_directory, seed):
    model=copy.deepcopy(base['model']);model.update(PROFILES[profile]['residual_model'])
    training=copy.deepcopy(base['training']);training.update(PROFILES[profile]['residual_training'])
    return dict(experiment='Ablation_FMTv8_Search_2.1',model=model,training=training,
                fusion=base[task.lower()]['fusion'],raw_checkpoint_dir=str(raw_directory),
                raw_backbone_seed=seed,raw_wide_parameter_count=148225,auxiliary_source='fmt',
                evaluation={'test_enabled':False},split=base[task.lower()])
