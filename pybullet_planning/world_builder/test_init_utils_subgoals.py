from __future__ import print_function

import unittest

from world_builder.init_utils import check_subgoal_achieved, remove_unnecessary_movable_given_goal


class ObjectInfo(object):

    def __init__(self, categories):
        self.categories = categories


class FakeWorld(object):

    def __init__(self, categories_by_body):
        self.categories_by_body = categories_by_body

    def body_to_object(self, body):
        return ObjectInfo(self.categories_by_body[body])


class TestCheckSubgoalAchieved(unittest.TestCase):

    def test_picked_is_achieved_when_object_is_at_grasp(self):
        facts = [('atgrasp', 'left', 12, 'g0')]

        self.assertTrue(check_subgoal_achieved(facts, ['picked', 12], world=None))

    def test_picked_is_not_achieved_for_different_grasped_object(self):
        facts = [('atgrasp', 'left', 13, 'g0')]

        self.assertFalse(check_subgoal_achieved(facts, ['picked', 12], world=None))


class TestRemoveUnnecessaryMovableGivenGoal(unittest.TestCase):

    def test_sprinkledto_removes_target_region_grasping_facts(self):
        world = FakeWorld({5: ['movable']})
        init = [('graspable', 13), ('graspable', 5), ('stackable', 5), ('region', 5)]

        filtered = remove_unnecessary_movable_given_goal(init, [['sprinkledto', 13, 5]], world)

        self.assertIn(('graspable', 13), filtered)
        self.assertIn(('region', 5), filtered)
        self.assertNotIn(('graspable', 5), filtered)
        self.assertNotIn(('stackable', 5), filtered)


if __name__ == '__main__':
    unittest.main()
