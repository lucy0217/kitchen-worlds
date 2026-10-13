import unittest

from cogarch_tools.processes.pddlstream_agent import _get_derived_goal, _contains_goal_literals


class FakePddlObject(object):

    def __init__(self, pddl):
        self.pddl = pddl

    def __str__(self):
        return self.pddl


class TestDerivedGoal(unittest.TestCase):

    def test_picked_goal_checks_graspable_semantics(self):
        self.assertEqual(
            _get_derived_goal(['picked', '1|chicken-leg'], []),
            [['graspable', '1|chicken-leg']],
        )

    def test_sprinkledto_goal_checks_sprinkler_and_region_semantics(self):
        self.assertEqual(
            _get_derived_goal(['sprinkledto', '12|salt-shaker', '5|braiserbody#1'], []),
            [
                ['sprinkler', '12|salt-shaker'],
                ['region', '5|braiserbody#1'],
            ],
        )

    def test_contains_goal_literals_matches_lists_and_tuples(self):
        init = [
            ['graspable', '1|chicken-leg'],
            ('sprinkler', '12|salt-shaker'),
            ['region', '5|braiserbody#1'],
        ]

        self.assertTrue(_contains_goal_literals(
            [['graspable', '1|chicken-leg']],
            init,
        ))

    def test_contains_goal_literals_matches_object_values_by_name(self):
        init = [('graspable', FakePddlObject('1|chicken-leg'))]

        self.assertTrue(_contains_goal_literals(
            [['graspable', '1|chicken-leg']],
            init,
        ))

    def test_contains_goal_literals_matches_debug_name_to_body_id(self):
        init = [('graspable', 1)]

        self.assertTrue(_contains_goal_literals(
            [['graspable', '1|chicken-leg']],
            init,
        ))


if __name__ == '__main__':
    unittest.main()
