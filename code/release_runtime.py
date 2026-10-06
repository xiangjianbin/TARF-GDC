"""Path-independent access to frozen recipes and verified scientific assets."""
from pathlib import Path
import copy
import torch
from v024_inversion.utils import read_json, sha256_file, seed_everything
from v024_inversion.physics import Normalization, build_operator
from v024_inversion.model import build_model

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'0001': 'Base', '1001': 'Base-TARF', '0101': 'Base-Correction',
          '1101': 'TARF-GDC', '0001C': 'Base-FT', '1001C': 'Base-TARF-FT',
          'T0P': 'SC', 'B0P': 'MC', 'Obs-only': 'Obs-only'}


def registry():
    return {row['id']: row for row in read_json(ROOT / 'assets/model_registry.json')}


def config_for(identifier, assets, device='cuda'):
    """Resolve all run-time paths; never fall back to the author's workstation."""
    row = registry()[identifier]
    cfg = copy.deepcopy(read_json(ROOT / row['config']))
    assets = Path(assets).resolve()
    cfg['data'].update(directory=str(assets / 'dataset'),
                       cache_directory=str(assets / 'cache'),
                       normalization_path=str(ROOT / 'assets/normalization.json'))
    cfg['operator'].update(g6_path=str(assets / 'operator/g6.npy'),
        audit_path=str(ROOT / 'assets/operator_audit.json'),
        sensitivity_role_path=str(ROOT / 'assets/sensitivity_role.npy'),
        sensitivity_tzz_path=str(ROOT / 'assets/sensitivity_tzz.npy'))
    cfg.update(device=device, code_version='tarf_gdc_portable_v1', phase='release',
               release={'recipe': identifier, 'asset_root': str(assets)},
               experiment_id=identifier, output_root=str(ROOT / 'outputs/training'))
    cfg.pop('reuse_source', None)
    cfg.pop('provenance', None)
    if cfg.get('control'):
        cfg['control'].pop('scientific_protocol', None)
    code, seed = row['code'], row['seed']
    if code in ('0101', '1101', '0001C', '1001C', 'Obs-only'):
        source_code = '0001' if code in ('0101', '0001C') else '1001'
        source = registry()[f'{source_code}_s{seed}']
        cfg['staged_refinement'] = dict(checkpoint=str(assets / source['checkpoint']),
            checkpoint_sha256=source['sha256'], source_epoch=source['epoch'],
            source_code=source_code, freeze_backbone=code not in ('0001C', '1001C'),
            backbone_eval_mode=code not in ('0001C', '1001C'),
            epoch_zero_candidate=True, stage1_training_epochs=50, stage2_training_epochs=50)
    else:
        cfg.pop('staged_refinement', None)
    return cfg


def verify_operator(assets):
    expected = read_json(ROOT / 'assets/operator_specification.json')['g6_sha256']
    if sha256_file(Path(assets) / 'operator/g6.npy') != expected:
        raise ValueError('Forward matrix checksum mismatch')
    audit = read_json(ROOT / 'assets/operator_audit.json')
    for mode in ('role', 'tzz'):
        if sha256_file(ROOT / f'assets/sensitivity_{mode}.npy') != audit[f'sensitivity_{mode}_sha256']:
            raise ValueError('Sensitivity checksum mismatch: ' + mode)
    normalization = read_json(ROOT / 'assets/normalization.json')
    if normalization['normalization'] != next(iter(registry().values()))['normalization']:
        raise ValueError('Normalization differs from frozen checkpoint registry')
    if sha256_file(Path(assets) / 'dataset/SHA256SUMS') != normalization['source_sha256sums']:
        raise ValueError('Wrong synthetic dataset freeze; run the full release check before training')


def load_model(identifier, assets, device='cuda', checkpoint=None):
    """Check hashes before deserializing trusted historical checkpoint metadata."""
    row = registry()[identifier]
    historical = checkpoint is None
    checkpoint = Path(assets) / row['checkpoint'] if historical else Path(checkpoint).resolve()
    actual_hash = sha256_file(checkpoint)
    if historical and actual_hash != row['sha256']:
        raise ValueError('Checkpoint checksum mismatch: ' + identifier)
    verify_operator(assets)
    cfg = config_for(identifier, assets, device)
    seed_everything(cfg['seed'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if not historical and saved['config'].get('phase') == 'smoke':
        raise ValueError('A smoke checkpoint cannot be used for manuscript evaluation')
    for key in ('seed', 'model', 'loss', 'noise', 'task'):
        if saved['config'][key] != cfg[key]:
            raise ValueError('Checkpoint/recipe mismatch: ' + key)
    norm = Normalization.from_json(ROOT / 'assets/normalization.json')
    if norm.to_dict() != saved['normalization']:
        raise ValueError('Checkpoint normalization differs from release')
    op = build_operator(cfg, norm, ROOT).to(device)
    model = build_model(cfg, norm, op).to(device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    cfg['loaded_checkpoint_sha256'] = actual_hash
    cfg['loaded_historical_checkpoint'] = historical
    return cfg, model, op
