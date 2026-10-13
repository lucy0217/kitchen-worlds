import unittest
from unittest.mock import patch

from pybullet_tools.pr2_streams import get_base_motion_gen


class FakeSaver:

    def __init__(self, *args, **kwargs):
        pass

    def restore(self):
        pass


class FakeRobot:
    body = 0
    self_collisions = False

    def __init__(self):
        self.world = FakeWorld()


class FakeWorld:
    ignored_pairs = []

    def body_to_object(self, body):
        return {4: FakeBodyInfo(['braiserlid']), 3: FakeBodyInfo(['counter']), 6: FakeBodyInfo(['oven'])}.get(body)

    def cat_to_bodies(self, category):
        return {'counter': [3], 'oven': [6]}.get(category, [])


class FakeBodyInfo:

    def __init__(self, categories):
        self.categories = categories
        self.supporting_surface = None


class FakeProblem:

    def __init__(self):
        self.robot = FakeRobot()
        self.fixed = [3, 6]
        self.world = self.robot.world


class FakeAttachment:

    def __init__(self, parent, child):
        self.parent = parent
        self.child = child


class FakeBConf:
    joints = [0, 1]

    def __init__(self, values):
        self.values = values

    def assign(self):
        pass


class BaseMotionTests(unittest.TestCase):

    def test_base_motion_ignores_counter_contact_for_held_braiser_lid(self):
        problem = FakeProblem()
        attachment = FakeAttachment(problem.robot, 4)

        with patch('pybullet_tools.pr2_streams.BodySaver', FakeSaver), \
             patch('pybullet_tools.pr2_streams.process_motion_fluents', return_value=[attachment]), \
             patch('pybullet_tools.pr2_streams.State', return_value='state'), \
             patch('pybullet_tools.pr2_streams.Trajectory', side_effect=lambda path: path), \
             patch('pybullet_tools.pr2_streams.Commands', side_effect=lambda *args, **kwargs: ('cmd', args, kwargs)), \
             patch('pybullet_tools.pr2_streams.plan_joint_motion', return_value=[(0, 0)]) as plan_motion:
            result = get_base_motion_gen(problem)(FakeBConf((1, 2)), FakeBConf((3, 4)), fluents=[('atgrasp', 'left', 4, 'g')])

        self.assertIsNotNone(result)
        ignored_pairs = plan_motion.call_args.kwargs['ignored_pairs']
        self.assertIn((3, 4), ignored_pairs)
        self.assertIn((4, 3), ignored_pairs)
        self.assertIn((6, 4), ignored_pairs)
        self.assertIn((4, 6), ignored_pairs)


if __name__ == '__main__':
    unittest.main()
