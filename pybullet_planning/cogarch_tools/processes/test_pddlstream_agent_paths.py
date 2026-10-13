from __future__ import print_function

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cogarch_tools.processes.pddlstream_agent import correct_home_path, PDDLStreamAgent


class TestCorrectHomePath(unittest.TestCase):

    def test_uses_current_exp_dir_when_legacy_marker_is_absent(self):
        loaded_exp_dir = '/tmp/workspace/experiments/old_run'
        current_exp_dir = '/tmp/workspace/experiments/new_run'

        self.assertEqual(current_exp_dir, correct_home_path(loaded_exp_dir, current_exp_dir))

    def test_archives_separate_stream_logs_for_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'visualizations'
            source.mkdir()
            agent = SimpleNamespace(pddlstream_kwargs={'log_failures': True},
                                    exp_dir=str(root / 'run'), problem_count=2,
                                    problem_count_suffix='b')
            with patch('pddlstream.algorithms.visualization.VISUALIZATIONS_DIR', str(source)):
                for marker in [1, 2]:
                    (source / 'log.json').write_text(json.dumps({'attempt': marker}))
                    PDDLStreamAgent._archive_stream_failures(agent)
            logs = list((root / 'run' / 'log' / 'stream_failures').glob('*.json'))
            self.assertEqual(sorted(json.loads(p.read_text())['attempt'] for p in logs), [1, 2])

    def test_logging_can_be_reenabled_after_a_disabled_solve(self):
        from pddlstream.algorithms import visualization
        with patch.object(visualization, 'VISUALIZE', False), \
                patch.object(visualization, 'ensure_dir'), \
                patch('os.path.isfile', return_value=False):
            visualization.reset_visualizations()
            self.assertTrue(visualization.VISUALIZE)


if __name__ == '__main__':
    unittest.main()
