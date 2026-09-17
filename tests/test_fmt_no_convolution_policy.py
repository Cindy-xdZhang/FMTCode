"""Regression checks for full-path FMT convolution prohibition."""
import ast
import json
from pathlib import Path
import subprocess
import types

import pytest
import torch
from torch import nn
from FMT_Utils.FMTNoConvolution_1_1 import (
    ForbiddenFMTConvolutionError, assert_no_fmt_convolution,
)
from FMT_Utils.Task4C_FPSAugmentSearch_1_1 import FourierClassifier
from FMT_Utils.Task4B_Classifier_3D import PathlineMulticlassClassifier3D

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('layer', [nn.Conv1d, nn.Conv2d, nn.Conv3d,
                                   nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d])
def test_hidden_frozen_convolution_is_rejected(layer):
    model = nn.ModuleDict({'frozen_raw': nn.Sequential(nn.Linear(3, 3), layer(3, 3, 1))})
    model.requires_grad_(False)
    with pytest.raises(ForbiddenFMTConvolutionError, match='frozen_raw'):
        assert_no_fmt_convolution(model)


@pytest.mark.parametrize('architecture,parameters', [('original', 76738), ('wide_residual', 992386)])
def test_retained_task4_models_are_identical_to_previous_revision(architecture, parameters):
    source = subprocess.check_output(['git', 'show',
        '52f6284c:FMT_Utils/Task4C_FPSAugmentSearch_1_1.py'], cwd=ROOT).decode('utf-8')
    previous = types.ModuleType('previous_fmt_classifier')
    exec(compile(source, 'previous_fmt_classifier', 'exec'), previous.__dict__)
    torch.manual_seed(12)
    old = previous.FourierClassifier('p35', architecture, 'h0').eval()
    torch.manual_seed(12)
    new = FourierClassifier('p35', architecture, 'h0').eval()
    assert_no_fmt_convolution(new)
    assert sum(p.numel() for p in new.parameters()) == parameters
    assert old.state_dict().keys() == new.state_dict().keys()
    for key in old.state_dict():
        assert torch.equal(old.state_dict()[key], new.state_dict()[key])
    tokens = torch.randn(3, 27, 142)
    tokens[..., -1] = 1
    tokens[:, 20:, -1] = 0
    assert torch.equal(old(tokens), new(tokens))


def test_task4b_only_the_raw_fmt_variant_is_removed():
    with pytest.raises(ForbiddenFMTConvolutionError):
        PathlineMulticlassClassifier3D(variant='raw_fmt', fmt_dim=141)
    model = PathlineMulticlassClassifier3D(variant='fmt_only', fmt_dim=141).eval()
    assert_no_fmt_convolution(model)
    assert model(torch.randn(2, 7, 32, 3), torch.randn(2, 141)).shape == (2, 4)


@pytest.mark.parametrize('module_name', ['FTLE_Upsampling_2D_1_1', 'FTLE_Upsampling_2D_1_2'])
def test_2d_fmt_unet_removed_before_config_or_device_access(module_name):
    import importlib
    module = importlib.import_module('experiments.'+module_name)
    assert module.GEOMETRY == ('raw',)
    for name in ('fmt', 'fmt_v5', 'fmt_objective_ntod_v2', 'p35'):
        with pytest.raises(ForbiddenFMTConvolutionError):
            module.Model({}, name, 2, 141)
    with pytest.raises(ForbiddenFMTConvolutionError):
        module.submit({}, 'unused.json')


def test_p35_spatial_fusion_removed():
    from FMT_Utils.FTLE_Fusion_2D_2_1 import FusionSR
    with pytest.raises(ForbiddenFMTConvolutionError):
        FusionSR(2)


def test_model_zoo_retired_classes_have_no_implementation():
    # Avoid importing unrelated legacy point-cloud extensions.
    source = (ROOT/'FMT_Utils/model_zoo.py').read_text(encoding='utf-8')
    names = {'AttentionFusion', 'FTLEUpsamplingFMT_Unet', 'FTLEupsamplingFMT_UnetV3',
             'FTLEupsamplingFMT_UnetV2', 'FTLEupsamplingDCT_FMT_UnetV2'}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in names]
    assert len(nodes) == 5
    from FMT_Utils.FMTNoConvolution_1_1 import reject_retired_fmt
    for node in nodes:
        namespace = dict(nn=nn, reject_retired_fmt=reject_retired_fmt)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'retired_zoo', 'exec'), namespace)
        with pytest.raises(ForbiddenFMTConvolutionError):
            namespace[node.name]()


def test_direct_task35_models_have_no_convolution():
    from FMT_Utils.ASAPFMT_Task35_1_1 import ARMS, classifier
    for arm in ARMS:
        assert_no_fmt_convolution(classifier(arm))


def test_archival_table_bytes_are_hash_locked():
    import hashlib
    manifest = json.loads((ROOT/'docs/archive/retired_fmt_convolution_20260917/manifest.json').read_text())
    for row in manifest:
        assert hashlib.sha256((ROOT/row['archive']).read_bytes()).hexdigest() == row['sha256']
    active = (ROOT/'docs/paper_tables_tasks_3d.md').read_text(encoding='utf-8')
    assert 'Task3：监督 IVD 二分类' not in active
    assert '| fmt_c156 | 0.697036' in active
