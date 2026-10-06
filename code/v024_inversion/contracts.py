"""Portable contracts: scientific settings frozen, machine paths configurable.

Reruns are NOT marked as historical formal experiments. The checkpoint registry
retains original hashes; each new run records its own code/configuration snapshot.
"""
from pathlib import Path
import math
from .utils import read_json, sha256_file

VERSION = 'tarf_gdc_portable_v1'
ROOT = Path(__file__).resolve().parents[2]


def validate_config(config):
    if config.get('code_version') != VERSION:
        raise ValueError('Use reproduce.py to resolve a release recipe')
    if config.get('phase') not in ('release', 'smoke'):
        raise ValueError('Release reruns cannot claim historical formal status')
    identifier = config['release']['recipe']
    reference = read_json(ROOT / 'configs' / (identifier + '.json'))
    for key in ('seed', 'model', 'task', 'loss', 'noise', 'initialization', 'evaluation'):
        if config[key] != reference[key]:
            raise ValueError('Scientific recipe changed: ' + key)
    mutable = {'cpu_threads', 'num_workers', 'prefetch_factor', 'log_interval'}
    if config['phase'] == 'smoke':
        mutable |= {'epochs', 'max_steps', 'batch_size', 'amp_dtype'}
    for key, value in reference['training'].items():
        if key not in mutable and config['training'].get(key) != value:
            raise ValueError('Training protocol changed: ' + key)
    if config['phase'] == 'release' and any(config['data'].get(k) for k in
            ('train_limit', 'validation_limit', 'evaluation_limits')):
        raise ValueError('Full reproduction cannot truncate the dataset')
    if not math.isfinite(config['loss']['physics_weight']):
        raise ValueError('Nonfinite physical loss weight')
    for key in ('directory', 'normalization_path', 'cache_directory'):
        if not Path(config['data'][key]).is_absolute():
            raise ValueError('Resolve data paths before training: ' + key)
    refined = config['model'].get('architecture', {}).get('residual_refiner') == 'transformer'
    followup = refined or identifier.startswith(('0001C_', '1001C_'))
    if followup != bool(config.get('staged_refinement')):
        raise ValueError('Follow-up runs require the matching initial model')
    if followup:
        stage = config['staged_refinement']
        if stage['freeze_backbone'] != refined or not stage['epoch_zero_candidate']:
            raise ValueError('Changed backbone/epoch-zero protocol')


def source_hashes(project):
    return {str(p.relative_to(project)): sha256_file(p)
            for p in sorted((project / 'code').rglob('*.py'))}


def verify_implementation_freeze(project, config_path, config):
    raise RuntimeError('Historical formal execution is not a release entry point')
