"""Fast CPU contracts independent of the full data bundle."""
from pathlib import Path
import ast
import io
import re
import sys
import tokenize
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'scripts')]
from release_runtime import registry, config_for
from v024_inversion.physics import tensor_roles
from v024_inversion.evaluation import pair_difference_metrics
from v024_inversion.contracts import validate_config
from check_release import check_tables


def test_registry_complete():
    rows = registry()
    assert len(rows) == 27
    assert all(sum(r['seed'] == seed for r in rows.values()) == 9 for seed in (1107,1108,1109))
    assert len({r['sha256'] for r in rows.values()}) == 27


def test_portable_recipes(tmp_path):
    for name in registry():
        cfg = config_for(name, tmp_path)
        validate_config(cfg)
        assert Path(cfg['data']['directory']).parent == tmp_path
        assert str(tmp_path) in cfg['operator']['g6_path']


def test_role_order_and_trace():
    raw = torch.tensor([[3.,2.,4.,6.,5.,-9.]])
    expected = torch.tensor([[-9.,4.,5.,-1.5,2.]])
    assert torch.equal(tensor_roles(raw), expected)


def test_pair_metrics():
    a, b = np.array([1.,-1.]), np.array([0.,0.])
    good = pair_difference_metrics(a,b,a,b)
    assert good['difference_nrmse'] == 0.
    assert np.isclose(good['pdacc_cos'],1.)
    wrong = pair_difference_metrics(b,a,a,b)
    assert np.isclose(wrong['difference_nrmse'],2.)
    assert np.isclose(wrong['pdacc_cos'],-1.)


def test_paper_tables():
    assert check_tables()['passed']


def test_all_comments_and_docstrings_english():
    han = re.compile('[\u4e00-\u9fff]')
    for folder in ('code','dataset_generation','scripts','tests','figures'):
        for path in (ROOT/folder).rglob('*.py'):
            source = path.read_text()
            for token in tokenize.generate_tokens(io.StringIO(source).readline):
                if token.type == tokenize.COMMENT:
                    assert not han.search(token.string), str(path)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node,(ast.Module,ast.ClassDef,ast.FunctionDef)):
                    assert not han.search(ast.get_docstring(node) or ''), str(path)


def test_self_contained_demo(tmp_path):
    from demo import run
    result = run(tmp_path/'demo')
    assert result['passed'] and result['checkpoint_replay_exact']
