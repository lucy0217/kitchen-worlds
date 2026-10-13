from __future__ import print_function

import unittest

from world_builder.world import World


class BodyInfo(object):

    def __init__(self, grasp_parent=None):
        self.grasp_parent = grasp_parent


class TestRefineMarkerObstacles(unittest.TestCase):

    def test_non_marker_body_leaves_obstacles_unchanged(self):
        world = World.__new__(World)
        world.BODY_TO_OBJECT = {}
        obstacles = [10, 20]

        self.assertEqual(obstacles, world.refine_marker_obstacles(10, obstacles))

    def test_marker_removes_grasp_parent_from_obstacles(self):
        world = World.__new__(World)
        world.BODY_TO_OBJECT = {5: BodyInfo(grasp_parent=10)}
        obstacles = [10, 20]

        self.assertEqual([20], world.refine_marker_obstacles(5, obstacles))


if __name__ == '__main__':
    unittest.main()
