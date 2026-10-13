import unittest
from unittest.mock import patch

from pybullet_tools.general_streams import get_above_pose_gen, get_contain_list_gen, get_handle_grasp_list_gen, \
    sample_joint_position_gen, prioritize_grasps_for_body, _prioritize_faucet_handle_grasps, \
    _faucet_handle_grasp_variants
from pybullet_tools.pr2_primitives import Pose
from pybullet_tools.utils import AABB


class SprinklePoseTests(unittest.TestCase):

    def test_above_pose_uses_minimum_sprinkle_clearance(self):
        region = 5
        body = 12
        region_pose = Pose(region, value=((0, 0, 1), (0, 0, 0, 1)))
        region_pose.assign = lambda: None
        region_aabb = AABB(lower=(-1, -1, 0), upper=(1, 1, 1))
        body_aabb = AABB(lower=(-0.05, -0.05, 0), upper=(0.05, 0.05, 0.1))

        def fake_get_aabb(entity, *args, **kwargs):
            return region_aabb if entity == region else body_aabb

        with patch('pybullet_tools.general_streams.get_aabb', side_effect=fake_get_aabb):
            pose = next(get_above_pose_gen(None, num_samples=1)(region, region_pose, body))[0]

        point, _ = pose.value
        self.assertGreaterEqual(point[2], region_aabb.upper[2] + 0.85)

    def test_above_pose_samples_high_sprinkle_clearance(self):
        region = 5
        body = 12
        region_pose = Pose(region, value=((0, 0, 1), (0, 0, 0, 1)))
        region_pose.assign = lambda: None
        region_aabb = AABB(lower=(-1, -1, 0), upper=(1, 1, 1))
        body_aabb = AABB(lower=(-0.05, -0.05, 0), upper=(0.05, 0.05, 0.1))

        def fake_get_aabb(entity, *args, **kwargs):
            return region_aabb if entity == region else body_aabb

        with patch('pybullet_tools.general_streams.get_aabb', side_effect=fake_get_aabb):
            gen = get_above_pose_gen(None, num_samples=1)(region, region_pose, body)
            poses = [entry[0] for entry in gen]

        self.assertGreaterEqual(max(pose.value[0][2] for pose in poses), region_aabb.upper[2] + 1.2)

    def test_above_pose_marks_target_region_as_support(self):
        class FakeSurface:
            pybullet_name = (3, None, 35)

        class FakeBodyInfo:
            supporting_surface = FakeSurface()

        class FakeWorld:
            def body_to_object(self, body):
                return FakeBodyInfo() if body == 5 else None

        class FakeProblem:
            world = FakeWorld()

        region = 5
        body = 13
        region_pose = Pose(region, value=((0, 0, 1), (0, 0, 0, 1)))
        region_pose.assign = lambda: None
        region_aabb = AABB(lower=(-1, -1, 0), upper=(1, 1, 1))
        body_aabb = AABB(lower=(-0.05, -0.05, 0), upper=(0.05, 0.05, 0.1))

        def fake_get_aabb(entity, *args, **kwargs):
            return region_aabb if entity == region else body_aabb

        with patch('pybullet_tools.general_streams.get_aabb', side_effect=fake_get_aabb):
            pose = next(get_above_pose_gen(FakeProblem(), num_samples=1)(region, region_pose, body))[0]

        self.assertEqual(pose.support, region)


class ContainPoseTests(unittest.TestCase):

    def test_lid_on_braiser_uses_adjusted_body_level_container_pose(self):
        class FakeProblem:
            fixed = []

            class FakeWorld:
                learned_pose_list_gen = None

                def get_name(self, body):
                    return {4: 'braiserlid#1', 5: 'braiserbody#1'}[body]

            world = FakeWorld()

        lid = 4
        pot = 5
        sampled_pose = ((0.1, 0.2, 0.3), (0, 0, 0, 1))
        adjusted_pose = ((0.5, 0.6, 0.7), (0, 0, 0, 1))

        with patch('pybullet_tools.general_streams.sample_obj_in_body_link_space') as sample_inside, \
             patch('pybullet_tools.general_streams.get_pose', return_value=sampled_pose), \
             patch('pybullet_tools.general_streams.get_aabb_center', return_value=(0.1, 0.2, 0.0)), \
             patch('pybullet_tools.general_streams.get_aabb', return_value=object()), \
             patch('pybullet_tools.general_streams.quat_from_euler', return_value=(0, 0, 0, 1)), \
             patch('pybullet_tools.general_streams.get_mod_pose', return_value=sampled_pose), \
             patch('pybullet_tools.general_streams.adjust_sampled_pose', return_value=adjusted_pose) as adjust, \
             patch('pybullet_tools.pr2_primitives.Pose.assign'), \
             patch('pybullet_tools.general_streams.collided', return_value=False):
            pose = get_contain_list_gen(FakeProblem(), num_samples=1)(lid, pot)[0][0]

        adjust.assert_called_once_with(FakeProblem.world, lid, pot, sampled_pose)
        sample_inside.assert_not_called()
        self.assertEqual(pose.value, adjusted_pose)
        self.assertEqual(pose.support, pot)


class CondimentGraspOrderingTests(unittest.TestCase):

    def test_prioritizes_reachable_center_grasp_for_braiser_lid(self):
        class FakeWorld:
            def cat_to_bodies(self, category):
                return []

            def get_name(self, body):
                return 'braiserlid#1'

        class FakeGrasp:
            def __init__(self, value):
                self.value = value

        left_offset = FakeGrasp(((-0.051, 0.232, 0.007), (0.0, -1.571, 1.571)))
        right_offset = FakeGrasp(((0.051, 0.232, 0.007), (0.0, 1.571, -1.571)))
        center_reachable = FakeGrasp(((-0.001, 0.232, 0.007), (0.0, 1.571, -1.571)))
        center_other_side = FakeGrasp(((-0.001, 0.232, 0.007), (0.0, -1.571, 1.571)))

        ordered = prioritize_grasps_for_body(
            FakeWorld(), 4, [left_offset, right_offset, center_other_side, center_reachable]
        )

        self.assertIs(ordered[0], center_reachable)

    def test_braiser_lid_grasp_order_accepts_pose_quaternion_values(self):
        class FakeWorld:
            def cat_to_bodies(self, category):
                return []

            def get_name(self, body):
                return 'braiserlid#1'

        class FakeGrasp:
            def __init__(self, value):
                self.value = value

        center = FakeGrasp(((-0.001, 0.232, 0.007), (0.5, 0.5, -0.5, 0.5)))
        offset = FakeGrasp(((0.051, 0.232, 0.007), (0.5, 0.5, -0.5, 0.5)))

        ordered = prioritize_grasps_for_body(FakeWorld(), 4, [offset, center])

        self.assertIs(ordered[0], center)

    def test_prioritizes_side_grasps_for_sprinklers(self):
        class FakeWorld:
            def cat_to_bodies(self, category):
                return [12, 13] if category == 'sprinkler' else []

            def get_name(self, body):
                return {12: 'salt-shaker', 13: 'pepper-shaker'}[body]

        class FakeGrasp:
            def __init__(self, value):
                self.value = value

        top = FakeGrasp(((0.0, 0.0, 0.134), (0, 0, 0, 1)))
        bottom = FakeGrasp(((0.0, 0.0, -0.132), (0, 0, 0, 1)))
        side = FakeGrasp(((0.0, 0.154, 0.0), (0, 0, 0, 1)))

        ordered = prioritize_grasps_for_body(FakeWorld(), 13, [top, bottom, side])

        self.assertIs(ordered[0], side)

    def test_prioritizes_top_grasps_for_salt(self):
        class FakeWorld:
            def cat_to_bodies(self, category):
                return [12] if category == 'sprinkler' else []

            def get_name(self, body):
                return 'salt-shaker'

        class FakeGrasp:
            def __init__(self, value):
                self.value = value

        top = FakeGrasp(((0.0, 0.0, 0.134), (0, 0, 0, 1)))
        side = FakeGrasp(((0.0, 0.154, 0.0), (0, 0, 0, 1)))

        ordered = prioritize_grasps_for_body(FakeWorld(), 12, [top, side])

        self.assertIs(ordered[0], top)


class HandleGraspTests(unittest.TestCase):

    def test_faucet_open_position_sampling_prefers_reversible_small_angles(self):
        class FakeWorld:
            learned_position_list_gen = None
            BODY_TO_OBJECT = {
                (9, 3): type('BodyInfo', (), {'categories': ['faucet', 'knob', 'joint']})(),
            }

            def get_name(self, body):
                return 'faucet#1'

        class FakeProblem:
            world = FakeWorld()

        with patch('pybullet_tools.general_streams.Position') as MockPosition:
            def fake_position(body, value):
                position = type('FakePosition', (), {})()
                position.body = body
                position.value = {'min': 0.0, 'max': 1.571}.get(value, value)
                position.is_prismatic = lambda: False
                return position

            MockPosition.side_effect = fake_position
            current = fake_position((9, 3), 0.0)
            sampled = list(sample_joint_position_gen(FakeProblem(), num_samples=4, verbose=False)((9, 3), current))

        values = [entry[0].value for entry in sampled]
        self.assertTrue(values)
        self.assertTrue(all(0.4 <= value <= 0.42 for value in values))

    def test_faucet_handle_grasps_include_local_variants_only_for_faucets(self):
        class FakeRobot:
            name = 'pr2'

            def get_approach_vector(self, *args, **kwargs):
                return (0.0, 0.0, -0.5)

            def get_approach_pose(self, app, grasp):
                return grasp

            def get_carry_conf(self, *args, **kwargs):
                return []

            def compute_grasp_width(self, *args, **kwargs):
                return 0.04

        class FakeWorld:
            BODY_TO_OBJECT = {
                (9, 3): type('BodyInfo', (), {'categories': ['faucet', 'knob', 'joint']})(),
                (6, 4): type('BodyInfo', (), {'categories': ['knob', 'joint']})(),
            }
            robot = FakeRobot()

            def cat_to_bodies(self, category):
                return [(9, 3), (6, 4)] if category == 'knob' else []

            def get_name(self, body):
                return ''

        class FakeProblem:
            fixed = []
            world = FakeWorld()
            robot = world.robot

        base_grasp = (-0.084, -0.04, 0.114, -3.142, 0.0, 1.571)

        with patch('pybullet_tools.general_streams.get_hand_grasps', return_value=[base_grasp]), \
             patch('pybullet_tools.general_streams.get_handle_link', return_value=3), \
             patch('pybullet_tools.general_streams.get_link_pose', return_value=((0, 0, 0), (0, 0, 0, 1))):
            faucet_grasps = get_handle_grasp_list_gen(FakeProblem(), num_samples=8)((9, 3))
            knob_grasps = get_handle_grasp_list_gen(FakeProblem(), num_samples=8)((6, 4))

        self.assertGreater(len(faucet_grasps), len(knob_grasps))
        self.assertEqual(len(knob_grasps), 1)
        self.assertEqual(knob_grasps[0][0].value, base_grasp)

    def test_faucet_handle_grasps_prioritize_reachable_local_orientation(self):
        grasps = [
            (-0.036, 0.034, 0.0, 0.0, 1.571, -1.571),
            (0.011, 0.074, 0.0, 3.142, 0.0, 1.571),
        ]

        ordered = _prioritize_faucet_handle_grasps(grasps)

        self.assertEqual(ordered[0], grasps[1])

    def test_faucet_handle_variant_order_starts_with_replay_verified_grasp(self):
        base_grasp = (0.011, 0.074, 0.0, 0.0, 0.0, 0.0)

        ordered = _prioritize_faucet_handle_grasps(_faucet_handle_grasp_variants([base_grasp]))

        for actual, expected in zip(ordered[0], (0.011, 0.074, 0.0, 3.142, 0.0, 1.571)):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_faucet_handle_variant_order_keeps_close_reverse_grasp_in_cap(self):
        base_grasp = (0.011, 0.074, 0.0, 0.0, 0.0, 0.0)

        ordered = _prioritize_faucet_handle_grasps(_faucet_handle_grasp_variants([base_grasp]))[:4]

        self.assertTrue(any(
            abs(grasp[0] - 0.011) < 0.01 and
            abs(grasp[1] - 0.074) < 0.01 and
            abs(grasp[2]) < 0.01 and
            abs(grasp[3]) < 0.01 and
            abs(grasp[4]) < 0.01 and
            abs(grasp[5] + 1.571) < 0.01
            for grasp in ordered
        ))

    def test_faucet_handle_grasp_priority_accepts_pose_tuple_values(self):
        target = ((0.011, 0.074, 0.0), (3.142, 0.0, 1.571))
        other = ((-0.036, 0.074, 0.0), (0.0, 1.571, -3.142))

        ordered = _prioritize_faucet_handle_grasps([other, target])

        self.assertIs(ordered[0], target)

    def test_faucet_handle_grasps_use_shorter_approach_than_other_knobs(self):
        class FakeRobot:
            name = 'pr2'

            def get_approach_vector(self, arm, grasp_type, scale=1):
                return (scale, 0.0, 0.0)

            def get_approach_pose(self, app, grasp):
                return app

            def get_carry_conf(self, *args, **kwargs):
                return []

            def compute_grasp_width(self, *args, **kwargs):
                return 0.04

        class FakeWorld:
            BODY_TO_OBJECT = {
                (9, 3): type('BodyInfo', (), {'categories': ['faucet', 'knob', 'joint']})(),
                (6, 4): type('BodyInfo', (), {'categories': ['knob', 'joint']})(),
            }
            robot = FakeRobot()

            def cat_to_bodies(self, category):
                return [(9, 3), (6, 4)] if category == 'knob' else []

            def get_name(self, body):
                return ''

        class FakeProblem:
            fixed = []
            world = FakeWorld()
            robot = world.robot

        base_grasp = (-0.084, -0.04, 0.114, -3.142, 0.0, 1.571)

        with patch('pybullet_tools.general_streams.get_hand_grasps', return_value=[base_grasp]), \
             patch('pybullet_tools.general_streams.get_handle_link', return_value=3), \
             patch('pybullet_tools.general_streams.get_link_pose', return_value=((0, 0, 0), (0, 0, 0, 1))):
            faucet_grasp = get_handle_grasp_list_gen(FakeProblem(), num_samples=1)((9, 3))[0][0]
            knob_grasp = get_handle_grasp_list_gen(FakeProblem(), num_samples=1)((6, 4))[0][0]

        self.assertEqual(faucet_grasp.approach, (0.2, 0.0, 0.0))
        self.assertEqual(knob_grasp.approach, (0.5, 0.0, 0.0))

    def test_faucet_handle_grasps_are_capped_to_targeted_variants(self):
        class FakeRobot:
            name = 'pr2'

            def get_approach_vector(self, *args, **kwargs):
                return (0.0, 0.0, -0.5)

            def get_approach_pose(self, app, grasp):
                return grasp

            def get_carry_conf(self, *args, **kwargs):
                return []

            def compute_grasp_width(self, *args, **kwargs):
                return 0.04

        class FakeWorld:
            BODY_TO_OBJECT = {
                (9, 3): type('BodyInfo', (), {'categories': ['faucet', 'knob', 'joint']})(),
            }
            robot = FakeRobot()

            def cat_to_bodies(self, category):
                return [(9, 3)] if category == 'knob' else []

            def get_name(self, body):
                return ''

        class FakeProblem:
            fixed = []
            world = FakeWorld()
            robot = world.robot

        base_grasp = (0.011, 0.074, 0.0, 0.0, 0.0, 0.0)

        with patch('pybullet_tools.general_streams.get_hand_grasps', return_value=[base_grasp]), \
             patch('pybullet_tools.general_streams.get_handle_link', return_value=3), \
             patch('pybullet_tools.general_streams.get_link_pose', return_value=((0, 0, 0), (0, 0, 0, 1))):
            faucet_grasps = get_handle_grasp_list_gen(FakeProblem(), num_samples=20)((9, 3))

        self.assertLessEqual(len(faucet_grasps), 4)


if __name__ == '__main__':
    unittest.main()
