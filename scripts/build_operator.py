"""Rebuild the exact receiver-major G6 used by the paper with SimPEG.

Adapted from the original build_vinton_operator_v0003.py (local provenance).
The historical Q matrix is not used by TARF-GDC: its role transform is defined
in physics.role_matrix. Never replace the supplied frozen operator in place.
"""
from pathlib import Path
import argparse
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'code'))
from v024_inversion.utils import read_json, write_json, sha256_file


def main():
    from discretize import TensorMesh
    from simpeg import maps
    from simpeg.potential_fields import gravity
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to replace an operator: ' + str(args.output))
    spec = read_json(ROOT / 'assets/operator_specification.json')
    nz, ny, nx = spec['density_shape_zyx']
    dx, dy, dz = spec['cell_size_xyz_m']
    mesh = TensorMesh([np.full(nx, dx), np.full(ny, dy), np.full(nz, dz)],
                      origin=spec['origin_xyz_m'])
    ry, rx = spec['receiver_shape_yx']
    xx, yy = np.meshgrid(np.linspace(0, nx * dx, rx), np.linspace(0, ny * dy, ry), indexing='xy')
    locations = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, spec['receiver_height_m'])]
    receiver = gravity.receivers.Point(locations, components=spec['components'])
    survey = gravity.survey.Survey(gravity.sources.SourceField([receiver]))
    simulation = gravity.simulation.Simulation3DIntegral(mesh, survey=survey,
        rhoMap=maps.IdentityMap(nP=mesh.nC), store_sensitivities='ram', sensitivity_dtype=np.float32)
    matrix = np.ascontiguousarray(simulation.G, dtype='<f4')
    if list(matrix.shape) != spec['g6_shape']:
        raise ValueError('Operator layout mismatch')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, matrix, allow_pickle=False)
    actual = sha256_file(args.output)
    result = dict(shape=list(matrix.shape), sha256=actual, expected=spec['g6_sha256'],
                  byte_identical=actual == spec['g6_sha256'])
    write_json(args.output.with_suffix('.verification.json'), result)
    if not result['byte_identical']:
        raise RuntimeError('Operator differs from the frozen artifact; inspect library versions')
    print('PASS: rebuilt G6 is byte-identical to the paper operator')


if __name__ == '__main__':
    main()
