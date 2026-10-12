import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from pddlstream import tamp_profile as p


class ProfileTests(unittest.TestCase):
    def test_tree_keeps_parent_paths_and_first_entry_order(self):
        record = p.Recorder({})
        with record.span('root'):
            with record.span('grounding'):
                with record.span('solver'):
                    pass
            with record.span('planning'):
                for _ in range(2):
                    with record.span('solver'):
                        with record.span('collision'):
                            pass
        data = json.loads(json.dumps(record.export()))
        root = data['call_tree'][0]
        self.assertEqual(root['name'], 'root')
        self.assertEqual([n['name'] for n in root['children']], ['grounding', 'planning'])
        self.assertEqual(root['children'][0]['children'][0]['calls'], 1)
        self.assertEqual(root['children'][1]['children'][0]['calls'], 2)
        self.assertEqual(data['spans']['solver']['calls'], 3)
        def check(node):
            self.assertAlmostEqual(node['inclusive_s'], node['exclusive_s'] +
                                   sum(c['inclusive_s'] for c in node['children']), places=8)
            for child in node['children']:
                check(child)
        check(root)

    def test_nested_accounting_and_stream_attribution(self):
        record = p.Recorder({})
        token = p._ACTIVE.set(record)
        try:
            with patch.object(p.time, 'perf_counter', side_effect=[0., 1., 3., 5.]):
                with p.span('stream.ik'):
                    with p.span('kernel.collision'):
                        pass
            self.assertEqual(record.stats['stream.ik']['inclusive_s'], 5.)
            self.assertEqual(record.stats['stream.ik']['exclusive_s'], 3.)
            self.assertEqual(record.kernels_by_stream['stream.ik']['kernel.collision']['inclusive_s'], 2.)
        finally:
            p._ACTIVE.reset(token)

    def test_generator_suspension_send_and_return(self):
        record = p.Recorder({})
        token = p._ACTIVE.set(record)
        @p.timed('generator')
        def gen():
            v = yield 3
            return v
        try:
            with patch.object(p.time, 'perf_counter', side_effect=[0., 1., 100., 102.]):
                g = gen()
                self.assertEqual(next(g), 3)
                self.assertEqual(record.stack, [])
                with self.assertRaises(StopIteration) as stop:
                    g.send(7)
                self.assertEqual(stop.exception.value, 7)
            self.assertEqual(record.stats['generator']['inclusive_s'], 3.)
            self.assertEqual(record.stats['generator']['exceptions'], 0)
        finally:
            p._ACTIVE.reset(token)

    def test_disabled_is_transparent(self):
        @p.timed('unused')
        def f(x): return x
        with patch.object(p.time, 'perf_counter', side_effect=AssertionError('timer called')):
            self.assertEqual(f(42), 42)

    def test_stream_success_and_exception(self):
        record = p.Recorder({})
        token = p._ACTIVE.set(record)
        @p.stream_call
        def call(obj, fail=False):
            if fail: raise TimeoutError('simulated')
            return [SimpleNamespace(is_successful=lambda: True)], []
        try:
            obj = SimpleNamespace(external=SimpleNamespace(name='ik'))
            call(obj)
            with self.assertRaises(TimeoutError): call(obj, True)
            data = record.export()
            stat = data['spans']['stream.ik']
            self.assertEqual(stat['successful_calls'], 1)
            self.assertEqual(stat['exceptions'], 1)
            self.assertEqual(stat['calls'], 2)
            self.assertIn('p95_s', stat)
        finally:
            p._ACTIVE.reset(token)

    def test_attempt_persists_exception_and_nested_attempt_once(self):
        class Agent:
            @p.attempt
            def outer(self): return self.inner()
            @p.attempt
            def inner(self): raise TimeoutError('simulated')
        with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'TAMP_PROFILE':'1','TAMP_PROFILE_DIR':tmp,'TAMP_CPROFILE':'1'}):
            agent=Agent(); agent.exp_dir=tmp
            with self.assertRaises(TimeoutError): agent.outer()
            files=list(Path(tmp).glob('tamp_profiles/*.json'))
            self.assertEqual(len(files), 1)
            self.assertEqual(json.loads(files[0].read_text())['metadata']['exception'], 'TimeoutError')
            self.assertEqual(len(list(Path(tmp).glob('tamp_profiles/*.pstats'))), 1)
            self.assertIsNone(p._ACTIVE.get())


if __name__ == '__main__': unittest.main()
