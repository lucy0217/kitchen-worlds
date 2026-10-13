import unittest

from pybullet_tools.pr2_primitives import APPROACH_DISTANCE
from robot_builder.robots import PR2Robot


class PR2ApproachTests(unittest.TestCase):

    def test_hand_approach_vector_moves_along_local_positive_z(self):
        robot = PR2Robot.__new__(PR2Robot)

        self.assertEqual(
            robot.get_approach_vector('left', 'hand'),
            (0.0, 0.0, APPROACH_DISTANCE / 2),
        )


if __name__ == '__main__':
    unittest.main()
