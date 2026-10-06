"""Strict same-seed initialization for GDC and matched continuation."""
from pathlib import Path
import torch
from .utils import sha256_file, state_dict_sha256
from .physics import Normalization


def backbone_state(model):
    return {name: value for name, value in model.state_dict().items()
            if not name.startswith('residual_refiner.')}


def backbone_hash(model):
    return state_dict_sha256(backbone_state(model))


def verify_frozen(model, expected_hash):
    """Catch optimizer leakage and changes to frozen model buffers."""
    for name, parameter in model.named_parameters():
        if not name.startswith('residual_refiner.'):
            if parameter.requires_grad or parameter.grad is not None:
                raise AssertionError('Initial predictor became trainable: ' + name)
    if backbone_hash(model) != expected_hash:
        raise AssertionError('Frozen initial predictor changed')


def initialize_stage1(model, config):
    stage = config['staged_refinement']
    path = Path(stage['checkpoint'])
    if sha256_file(path) != stage['checkpoint_sha256']:
        raise ValueError('Initial checkpoint hash changed')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    source = saved['config']
    if saved['normalization'] != Normalization.from_json(config['data']['normalization_path']).to_dict():
        raise ValueError('Initial checkpoint uses a different training normalization')
    if source['seed'] != config['seed'] or int(saved['epoch']) != stage['source_epoch']:
        raise ValueError('Initial seed or epoch mismatch')
    for key in ('loss', 'noise', 'task', 'initialization'):
        if source[key] != config[key]:
            raise ValueError('Initial scientific protocol mismatch: ' + key)
    expected = dict(config['model']['architecture'], residual_refiner='none')
    actual = dict(source['model']['architecture'], residual_refiner='none')
    if expected != actual:
        raise ValueError('Wrong initial architecture')
    current = backbone_state(model)
    if set(current) != set(saved['model']):
        raise ValueError('Partial backbone loading is forbidden')
    for name, value in current.items():
        other = saved['model'][name]
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError('Backbone tensor mismatch: ' + name)
    state = model.state_dict()
    state.update(saved['model'])
    model.load_state_dict(state, strict=True)
    expected_hash = state_dict_sha256(saved['model'])
    if stage['freeze_backbone']:
        model.freeze_initial_predictor()
        verify_frozen(model, expected_hash)
    elif model.uses_residual_refiner or not all(p.requires_grad for p in model.parameters()):
        raise ValueError('Continuation must train the entire initial predictor')
    return dict(**stage, frozen_state_sha256=expected_hash,
                total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
