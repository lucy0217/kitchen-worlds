import unittest

from pybullet_tools.pose_utils import _lift_contained_placement_z
from pybullet_tools.utils import AABB


class ContainmentPlacementTests(unittest.TestCase):

    def test_lift_contained_placement_z_stays_inside_container(self):
        container_aabb = AABB(lower=(0, 0, 0), upper=(1, 1, 1))
        obj_aabb = AABB(lower=(0.4, 0.4, 0.1), upper=(0.6, 0.6, 0.3))

        self.assertEqual(
            _lift_contained_placement_z(0.2, obj_aabb, container_aabb, clearance=0.25),
            0.45,
        )

    def test_lift_contained_placement_z_clamps_to_maximum_inside_container(self):
        container_aabb = AABB(lower=(0, 0, 0), upper=(1, 1, 1))
        obj_aabb = AABB(lower=(0.4, 0.4, 0.1), upper=(0.6, 0.6, 0.3))

        self.assertEqual(
            _lift_contained_placement_z(0.85, obj_aabb, container_aabb, clearance=0.25),
            0.9,
        )


if __name__ == '__main__':
    unittest.main()
