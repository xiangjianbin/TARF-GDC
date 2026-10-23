"""Asset-free regression checks for current manuscript figure routing."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('figure_runner', ROOT / 'scripts/render_figures.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class FigureRoutes(unittest.TestCase):
    def run_route(self, group, expected_module, expected_function):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets, field, output = root / 'assets', root / 'field', root / 'output'
            assets.mkdir()
            field.mkdir()
            marker = field / 'fixture.txt'
            marker.write_bytes(b'immutable test input')
            original = hashlib.sha256(marker.read_bytes()).hexdigest()
            callbacks = {name: Mock() for name in ('main', 'vinton', 'matched')}
            argv = ['render_figures.py', '--assets', str(assets), '--group', group,
                    '--field-source', str(field), '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch.object(sys, 'path', list(sys.path)), \
                    patch.dict(os.environ, {}), \
                    patch.object(RUNNER.runpy, 'run_path', return_value=callbacks) as load:
                RUNNER.main()
                self.assertEqual(Path(load.call_args[0][0]).stem, expected_module)
                callbacks[expected_function].assert_called_once_with()
                for name, callback in callbacks.items():
                    if name != expected_function:
                        callback.assert_not_called()
                if group == 'field-diagnostics':
                    self.assertEqual(sys.argv[1:], ['--source-dir', str(output / 'source_data'),
                                                   '--output-dir', str(output)])
            self.assertEqual(hashlib.sha256(marker.read_bytes()).hexdigest(), original)
            self.assertEqual((output / 'source_data/fixture.txt').read_bytes(), marker.read_bytes())
            report = json.loads((output / 'replay_provenance.json').read_text())
            self.assertEqual(report['group'], group)
            self.assertTrue(report['immutable_inputs_unchanged'])

    def test_figure_13_dispatches_spatial_vinton_plot(self):
        self.run_route('13', 'field_and_classical', 'vinton')

    def test_diagnostics_are_a_separate_route(self):
        self.run_route('field-diagnostics', 'field_metrics', 'main')

    def test_figure_12_still_uses_selected_frozen_seed(self):
        self.run_route('12', 'best_field_seed', 'main')

    def test_obsolete_group_14_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ['render_figures.py', '--assets', tmp, '--group', '14', '--output', tmp + '/out']
            with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                RUNNER.main()
            self.assertEqual(error.exception.code, 2)

    def test_field_source_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            for group in ('12', '13', 'field-diagnostics'):
                with self.subTest(group=group):
                    argv = ['render_figures.py', '--assets', tmp, '--group', group,
                            '--output', tmp + '/' + group]
                    with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), \
                            self.assertRaises(SystemExit) as error:
                        RUNNER.main()
                    self.assertEqual(error.exception.code, 2)

    def test_spatial_audit_uses_single_seed_source(self):
        # Execute only the save helper, without importing plotting dependencies.
        source = ast.parse((ROOT / 'figures/field_and_classical.py').read_text(encoding='utf-8'))
        function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'save')
        module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
        with tempfile.TemporaryDirectory() as tmp:
            namespace = {'OUT': Path(tmp), 'PANELS': [], 'MATCHED_SOURCE': Path('matched.npz'), 'plt': Mock()}
            exec(compile(module, '<save helper>', 'exec'), namespace)
            axis = Mock()
            axis.get_title.return_value = 'density'
            axis.get_xlim.return_value = (0, 4)
            axis.get_ylim.return_value = (0, 4)
            figure = Mock()
            figure.axes = [axis]
            with contextlib.redirect_stdout(io.StringIO()):
                namespace['save'](figure, 13)
            entry = namespace['PANELS'][0]
            self.assertEqual(entry['source'], 'vinton_Full_s1107.npz')
            self.assertNotIn('sample SD', entry['interval'])
            self.assertEqual([call.args[0].name for call in figure.savefig.call_args_list],
                             ['Figure13.svg', 'Figure13.pdf', 'Figure13.png'])


if __name__ == '__main__':
    unittest.main()
