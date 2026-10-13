import unittest
from unittest.mock import patch

from pybullet_tools.mobile_streams import get_ik_fn_old, get_ir_sampler, filter_grasp_obstacles_for_body, sample_bconf, \
    _priority_condiment_cabinet_bases, _priority_condiment_cabinet_pick_bases, _priority_pot_stove_bases, \
    _priority_pot_counter_pick_bases, _priority_lid_stove_pick_bases, _priority_stove_knob_bases, \
    _priority_faucet_knob_bases, _priority_lid_braiser_place_bases, get_handle_motion_contact_bodies, \
    get_target_support_bodies


class FakePose:
    value = ((0, 0, 0), (0, 0, 0, 1))
    support = None

    def __init__(self, value=None, support=None):
        if value is None:
            value = type(self).value
        self.value = value
        self.support = support

    def assign(self):
        pass


class FakePotPose(FakePose):

    def __init__(self):
        super().__init__(support=(3, None, 35))


class FakeSupportedPose(FakePose):

    def __init__(self):
        super().__init__(support=5)


class FakeGrasp:
    body = 1
    grasp_type = 'hand'
    value = ((0, 0, 0), (0, 0, 0, 1))

    def get_attachment(self, *args, **kwargs):
        class FakeAttachment:
            child = 1

        return FakeAttachment()


class FakeRobot:
    arms = ['left']
    body = 0
    use_torso = False

    def __init__(self):
        self.visualized = False
        self.restored = False
        self.saved_bodies = None

    def iterate_approach_path(self, arm, gripper, pose_value, grasp):
        yield None

    def visualize_grasp_approach(self, *args, **kwargs):
        self.visualized = True
        raise AssertionError('visualize_grasp_approach should not run without visualize=True')

    def open_arm(self, arm):
        pass

    def get_base_joints(self):
        return [0, 1, 2]

    def set_gripper_pose(self, *args, **kwargs):
        return 98

    def get_arm_joints(self, arm):
        return []

    def get_carry_conf(self, *args, **kwargs):
        return []

    def get_grasp_pose(self, *args, **kwargs):
        return ((0, 0, 0), (0, 0, 0, 1))


class FakeTorsoRobot(FakeRobot):
    use_torso = True

    def get_base_joints(self):
        return [0, 1, 17, 2]

    def get_base_positions(self):
        return (2, 6.25, 1.1, 3.142)

    def get_grasp_pose(self, *args, **kwargs):
        return ((0, 0, 1.0), (0, 0, 0, 1))


class FakeConf:

    def __init__(self, body, joints, values):
        self.body = body
        self.joints = joints
        self.values = values

    def assign(self):
        pass


class FakeProblem:
    floors = []

    def __init__(self):
        self.robot = FakeRobot()
        self.world = FakeWorld()
        self.fixed = [2, 5]

    def get_gripper(self, arm, visual=True):
        return 99


class FakeBodyInfo:

    def __init__(self, supporting_surface=None, grasp_parent=None, categories=None):
        self.supporting_surface = supporting_surface
        self.grasp_parent = grasp_parent
        self.categories = categories or []


class FakeSurface:

    def __init__(self, pybullet_name):
        self.pybullet_name = pybullet_name


class FakeWorld:

    def __init__(self):
        self.BODY_TO_OBJECT = {}
        self.attachments = {}
        self.ignored_pairs = []

    def body_to_object(self, body):
        return self.BODY_TO_OBJECT.get(body)

    def cat_to_bodies(self, category):
        return [
            body for body, body_info in self.BODY_TO_OBJECT.items()
            if category in (getattr(body_info, 'categories', []) or [])
        ]


class FakeWorldWithEmptyCategoryLookup(FakeWorld):

    def cat_to_bodies(self, category):
        return []


class FakeAttachment:

    def __init__(self, parent, child):
        self.parent = parent
        self.child = child


class FakeEntity:

    def __init__(self, body):
        self.body = body


class TestMobileStreams(unittest.TestCase):

    def test_verbose_collision_does_not_visualize_without_visualize_flag(self):
        problem = FakeProblem()
        gen_fn = get_ir_sampler(problem, verbose=True, visualize=False)

        with patch('pybullet_tools.mobile_streams.pairwise_collision', return_value=True), \
             patch('pybullet_tools.mobile_streams.draw_aabb'), \
             patch('pybullet_tools.mobile_streams.get_aabb'), \
             patch('pybullet_tools.mobile_streams.get_pose', return_value=((0, 0, 0), (0, 0, 0, 1))):
            self.assertEqual(list(gen_fn('left', 1, FakePose(), FakeGrasp())), [])

        self.assertFalse(problem.robot.visualized)

    def test_filter_grasp_obstacles_removes_support_parent_body(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[1] = FakeBodyInfo(
            supporting_surface=FakeSurface((10, None, 9)))

        self.assertEqual(
            filter_grasp_obstacles_for_body(world, 1, [3, 10, 11]),
            [3, 11],
        )

    def test_filter_grasp_obstacles_removes_contained_objects_when_grasping_container(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[1] = FakeBodyInfo(grasp_parent=5)

        self.assertEqual(
            filter_grasp_obstacles_for_body(world, 5, [1, 3, 4, 5]),
            [3, 4],
        )

    def test_filter_grasp_obstacles_removes_geometrically_contained_objects(self):
        world = FakeWorld()

        with patch('pybullet_tools.mobile_streams.is_contained', side_effect=lambda child, parent: child == 1 and parent == 5):
            self.assertEqual(
                filter_grasp_obstacles_for_body(world, 5, [1, 3, 4, 5]),
                [3, 4],
            )

    def test_filter_grasp_obstacles_removes_objects_attached_to_grasped_body(self):
        world = FakeWorld()
        world.attachments[1] = FakeAttachment(parent=5, child=1)

        self.assertEqual(
            filter_grasp_obstacles_for_body(world, 5, [1, 3, 4, 5]),
            [3, 4],
        )

    def test_filter_grasp_obstacles_removes_entity_attachments_to_grasped_body(self):
        world = FakeWorld()
        world.attachments[FakeEntity(1)] = FakeAttachment(parent=FakeEntity(5), child=FakeEntity(1))

        self.assertEqual(
            filter_grasp_obstacles_for_body(world, 5, [1, 3, 4, 5]),
            [3, 4],
        )

    def test_ir_sampler_ignores_objects_moving_with_grasped_container(self):
        problem = FakeProblem()
        problem.fixed = [1, 2]
        problem.world.BODY_TO_OBJECT[1] = FakeBodyInfo(grasp_parent=5)
        gen_fn = get_ir_sampler(problem, learned=False, max_attempts=1)

        def collides(body_a, body_b):
            return body_b == 1

        with patch('pybullet_tools.mobile_streams.uniform_pose_generator', return_value=iter([(0, 0, 0)])), \
             patch('pybullet_tools.mobile_streams.all_between', return_value=True), \
             patch('pybullet_tools.mobile_streams.set_joint_positions'), \
             patch('pybullet_tools.mobile_streams.get_joint_positions', return_value=[]), \
             patch('pybullet_tools.mobile_streams.get_custom_limits', return_value=([-1, -1, -1], [1, 1, 1])), \
             patch('pybullet_tools.pr2_primitives.set_joint_positions'), \
             patch('pybullet_tools.mobile_streams.pairwise_collision', side_effect=collides):
            self.assertIsNotNone(next(gen_fn('left', 5, FakePose(), FakeGrasp())))

    def test_ir_sampler_ignores_support_parent_for_target_region_approach(self):
        problem = FakeProblem()
        problem.fixed = [3]
        problem.world.BODY_TO_OBJECT[5] = FakeBodyInfo(
            supporting_surface=FakeSurface((3, None, 35)))
        gen_fn = get_ir_sampler(problem, learned=False, max_attempts=1)

        with patch('pybullet_tools.mobile_streams.uniform_pose_generator', return_value=iter([(0, 0, 0)])), \
             patch('pybullet_tools.mobile_streams.all_between', return_value=True), \
             patch('pybullet_tools.mobile_streams.set_joint_positions'), \
             patch('pybullet_tools.mobile_streams.get_joint_positions', return_value=[]), \
             patch('pybullet_tools.mobile_streams.get_custom_limits', return_value=([-1, -1, -1], [1, 1, 1])), \
             patch('pybullet_tools.pr2_primitives.set_joint_positions'), \
             patch('pybullet_tools.mobile_streams.pairwise_collision', return_value=False):
            self.assertIsNotNone(next(gen_fn('left', 13, FakePotPose(), FakeGrasp())))

    def test_target_support_bodies_include_oven_for_stove_surface(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[6] = FakeBodyInfo(categories=['oven'])

        self.assertEqual(
            get_target_support_bodies(world, (3, None, 4)),
            {3, 6},
        )

    def test_target_support_bodies_fall_back_to_body_object_categories_for_stove_oven(self):
        world = FakeWorldWithEmptyCategoryLookup()
        world.BODY_TO_OBJECT[6] = FakeBodyInfo(categories=['oven'])

        self.assertEqual(
            get_target_support_bodies(world, (3, None, 4)),
            {3, 6},
        )

    def test_ir_sampler_infers_missing_pose_support_for_lid_on_stove(self):
        problem = FakeProblem()
        problem.fixed = [6]
        problem.world.BODY_TO_OBJECT[4] = FakeBodyInfo(
            supporting_surface=FakeSurface((3, None, 4)),
            categories=['braiserlid', 'movable'])
        problem.world.BODY_TO_OBJECT[6] = FakeBodyInfo(categories=['oven'])
        gen_fn = get_ir_sampler(problem, learned=False, max_attempts=1)

        def collides(body_a, body_b):
            return body_b == 6

        with patch('pybullet_tools.mobile_streams.uniform_pose_generator', return_value=iter([(0, 0, 0)])), \
             patch('pybullet_tools.mobile_streams.all_between', return_value=True), \
             patch('pybullet_tools.mobile_streams.set_joint_positions'), \
             patch('pybullet_tools.mobile_streams.get_joint_positions', return_value=[]), \
             patch('pybullet_tools.mobile_streams.get_custom_limits', return_value=([-1, -1, -1], [1, 1, 1])), \
             patch('pybullet_tools.pr2_primitives.set_joint_positions'), \
             patch('pybullet_tools.mobile_streams.pairwise_collision', side_effect=collides), \
             patch('pybullet_tools.mobile_streams.Conf', FakeConf):
            result = next(gen_fn('left', 4, FakePose(), FakeGrasp()))

        self.assertIsNotNone(result)
        self.assertEqual(result[0].values, (1.25, 7.55, 0.3, 2.8))

    def test_target_support_bodies_include_contents_of_target_container(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[1] = FakeBodyInfo(grasp_parent=5)

        self.assertEqual(
            get_target_support_bodies(world, 5),
            {1, 5},
        )

    def test_torso_sampling_uses_gripper_height_not_current_high_torso(self):
        problem = FakeProblem()
        problem.robot = FakeTorsoRobot()
        gen_fn = get_ir_sampler(problem, learned=False, max_attempts=1)

        with patch('pybullet_tools.mobile_streams.uniform_pose_generator', return_value=iter([(1.4, 8.8, 0.0)])), \
             patch('pybullet_tools.mobile_streams.all_between', return_value=True), \
             patch('pybullet_tools.mobile_streams.set_joint_positions'), \
             patch('pybullet_tools.mobile_streams.get_joint_positions', return_value=[]), \
             patch('pybullet_tools.mobile_streams.get_custom_limits', return_value=([1, 3, 0, -3.2], [5, 10, 3, 3.2])), \
             patch('pybullet_tools.mobile_streams.pairwise_collision', return_value=False), \
             patch('pybullet_tools.mobile_streams.random.uniform', return_value=0.35) as uniform, \
             patch('pybullet_tools.mobile_streams.Conf', FakeConf):
            result = next(gen_fn('left', 5, FakePose(), FakeGrasp()))

        uniform.assert_called_once_with(0.30000000000000004, 0.75)
        self.assertEqual(result[0].values, (1.4, 8.8, 0.35, 0.0))

    def test_priority_condiment_cabinet_bases_only_for_cabinet_condiments(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[12] = FakeBodyInfo(categories=['condiment', 'sprinkler'])
        pose = FakePose(value=((0.771, 7.071, 1.152), (0, 0, 0, 1)), support=(3, None, 0))

        bases = _priority_condiment_cabinet_bases(world, 12, pose)
        self.assertEqual(len(bases), 2)
        self.assertEqual(bases[0], (1.508, 7.218, 0.968, 1.918))
        self.assertAlmostEqual(bases[1][0], 1.531)
        self.assertEqual(bases[1][1:], (7.116, 0.859, 1.972))
        self.assertEqual(_priority_condiment_cabinet_bases(world, 12, FakePose()), [])

    def test_priority_condiment_cabinet_pick_bases_for_supported_condiment(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[12] = FakeBodyInfo(
            supporting_surface=FakeSurface((3, None, 0)),
            categories=['condiment', 'sprinkler'])
        pose = FakePose(value=((0.771, 7.071, 1.152), (0, 0, 0, 1)))

        bases = _priority_condiment_cabinet_pick_bases(
            world, 12, pose, ((0, 0, 1.152), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2])

        self.assertEqual(bases[0], (1.508, 7.218, 0.702, 1.918))
        self.assertEqual(bases[1], (1.488, 7.04, 0.702, 1.933))
        self.assertEqual(_priority_condiment_cabinet_pick_bases(
            world, 99, pose, ((0, 0, 1.152), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2]), [])

    def test_priority_pot_stove_bases_only_for_braiserbody_on_stove(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[5] = FakeBodyInfo(categories=['braiserbody', 'movable'])
        pose = FakePose(value=((0.567, 8.18, 0.876), (0, 0, 0, 1)), support=(3, None, 5))

        bases = _priority_pot_stove_bases(
            world, 5, pose, ((0, 0, 0.958), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2])

        self.assertEqual(bases[0], (1.418, 8.438, 0.408, -3.0))
        self.assertEqual(len(bases), 4)
        self.assertEqual(_priority_pot_stove_bases(
            world, 5, FakePose(), ((0, 0, 0.958), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2]), [])

    def test_priority_pot_counter_pick_bases_for_braiserbody_on_counter_right(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[5] = FakeBodyInfo(
            supporting_surface=FakeSurface((3, None, 35)),
            categories=['braiserbody', 'movable'])
        pose = FakePose(value=((0.483, 9.029, 0.924), (0, 0, 0, 1)))

        bases = _priority_pot_counter_pick_bases(
            world, 5, pose, ((0, 0, 1.005), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2])

        self.assertEqual(bases[0], (1.022, 8.786, 0.455, 3.089))
        self.assertEqual(len(bases), 3)
        self.assertEqual(_priority_pot_counter_pick_bases(
            world, 99, pose, ((0, 0, 1.005), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2]), [])

    def test_priority_lid_stove_pick_bases_for_braiserlid_on_left_stove(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[4] = FakeBodyInfo(
            supporting_surface=FakeSurface((3, None, 4)),
            categories=['braiserlid', 'movable'])
        pose = FakePose(value=((0.567, 7.872, 0.712), (0, 0, 0, 1)))

        bases = _priority_lid_stove_pick_bases(
            world, 4, pose, ((0.561, 7.869, 0.994), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2])

        self.assertEqual(bases[0], (1.25, 7.55, 0.3, 2.8))
        self.assertIn((1.014, 7.723, 0.297, -2.795), bases)
        self.assertEqual(_priority_lid_stove_pick_bases(
            world, 5, pose, ((0.561, 7.869, 0.994), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2]), [])

    def test_priority_lid_braiser_place_bases_for_lid_on_braiserbody(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[4] = FakeBodyInfo(categories=['braiserlid', 'movable'])
        pose = FakePose(value=((0.567, 8.18, 0.876), (0, 0, 0, 1)), support=5)

        bases = _priority_lid_braiser_place_bases(
            world, 4, pose, ((0.567, 8.18, 1.064), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2])

        self.assertEqual(bases[0], (1.254, 8.047, 0.512, 2.994))
        self.assertIn((1.501, 8.48, 0.51, 2.14), bases)
        self.assertEqual(_priority_lid_braiser_place_bases(
            world, 4, FakePose(), ((0.567, 8.18, 1.064), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2]), [])

    def test_priority_stove_knob_bases_only_for_right_knob(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[(6, 4)] = FakeBodyInfo(categories=['knob', 'joint'])
        pose = FakePose(value=((0.293, 8.187, 0.993), (0, 0, 0, 1)))

        bases = _priority_stove_knob_bases(
            world, (6, 4), pose, ((0.398, 8.187, 0.978), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2])

        self.assertEqual(bases[0], (1.226, 8.208, 0.578, 1.385))
        self.assertIn((1.258, 8.461, 0.418, 0.134), bases)
        self.assertGreaterEqual(len(bases), 3)
        self.assertEqual(_priority_stove_knob_bases(
            world, (6, 5), pose, ((0.398, 8.187, 0.978), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2]), [])

    def test_priority_stove_knob_matches_name_when_joint_index_changes(self):
        world = FakeWorld()
        world.body_to_name = {'(6, 5)': 'oven#1::knob_joint_2',
                              '(6, 4)': 'oven#1::knob_joint_1'}
        world.BODY_TO_OBJECT[(6, 5)] = FakeBodyInfo(categories=['knob', 'joint'])
        pose = FakePose(value=((0.293, 8.187, 0.993), (0, 0, 0, 1)))
        args = (pose, ((0.398, 8.187, 0.978), (0, 0, 0, 1)),
                [1, 3, 0, -3.2], [5, 10, 3, 3.2])
        self.assertTrue(_priority_stove_knob_bases(world, (6, 5), *args))
        self.assertEqual(_priority_stove_knob_bases(world, (6, 4), *args), [])

    def test_priority_faucet_knob_bases_only_for_faucet_knob(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[(9, 3)] = FakeBodyInfo(categories=['knob', 'joint', 'faucet'])
        world.BODY_TO_OBJECT[(6, 4)] = FakeBodyInfo(categories=['knob', 'joint'])
        pose = FakePose(value=((0.265, 5.593, 1.012), (0, 0, 0, 1)))

        bases = _priority_faucet_knob_bases(
            world, (9, 3), pose, ((0.265, 5.45, 1.02), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2])

        self.assertGreaterEqual(len(bases), 3)
        self.assertEqual(bases[0], (1.08, 5.7, 0.7, 1.6))
        self.assertTrue(all(0 <= base[2] <= 3 for base in bases))
        self.assertEqual(_priority_faucet_knob_bases(
            world, (6, 4), pose, ((0.265, 5.45, 1.02), (0, 0, 0, 1)),
            [1, 3, 0, -3.2], [5, 10, 3, 3.2]), [])

    def test_handle_motion_contact_bodies_include_counter_for_knob(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[3] = FakeBodyInfo(categories=['counter'])
        world.BODY_TO_OBJECT[(6, 4)] = FakeBodyInfo(categories=['knob', 'joint'])

        self.assertEqual(get_handle_motion_contact_bodies(world, (6, 4)), {3, 6})
        self.assertEqual(get_handle_motion_contact_bodies(world, 6), set())

    def test_handle_motion_contact_bodies_include_stove_cookware_for_knob(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[3] = FakeBodyInfo(categories=['counter'])
        world.BODY_TO_OBJECT[5] = FakeBodyInfo(
            supporting_surface=FakeSurface((3, None, 5)))
        world.BODY_TO_OBJECT[4] = FakeBodyInfo(
            supporting_surface=FakeSurface(5))
        world.BODY_TO_OBJECT[1] = FakeBodyInfo(
            supporting_surface=FakeSurface(5))
        world.BODY_TO_OBJECT[12] = FakeBodyInfo(
            supporting_surface=FakeSurface((3, None, 0)))
        world.BODY_TO_OBJECT[(6, 4)] = FakeBodyInfo(categories=['knob', 'joint'])

        self.assertEqual(get_handle_motion_contact_bodies(world, (6, 4)), {1, 3, 4, 5, 6})

    def test_handle_motion_contact_bodies_include_faucet_support_chain(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[8] = FakeBodyInfo(categories=['basin'])
        world.BODY_TO_OBJECT[9] = FakeBodyInfo(categories=['faucet'])
        world.BODY_TO_OBJECT[(8, None, 2)] = FakeBodyInfo(categories=['surface'])
        world.BODY_TO_OBJECT[12] = FakeBodyInfo(supporting_surface=FakeSurface((3, None, 0)))
        world.BODY_TO_OBJECT[(9, 3)] = FakeBodyInfo(categories=['knob', 'joint', 'faucet'])

        contacts = get_handle_motion_contact_bodies(world, (9, 3))

        self.assertIn(8, contacts)
        self.assertIn(9, contacts)
        self.assertIn((8, None, 2), contacts)
        self.assertNotIn(12, contacts)

    def test_handle_motion_contact_bodies_include_faucet_contacts_for_hardcoded_faucet(self):
        world = FakeWorld()
        world.BODY_TO_OBJECT[8] = FakeBodyInfo(categories=['basin'])
        world.BODY_TO_OBJECT[9] = FakeBodyInfo(categories=['faucet'])
        world.BODY_TO_OBJECT[(8, None, 2)] = FakeBodyInfo(categories=['surface'])

        contacts = get_handle_motion_contact_bodies(world, (9, 3))

        self.assertIn(8, contacts)
        self.assertIn(9, contacts)
        self.assertIn((8, None, 2), contacts)

    def test_ik_excludes_target_support_from_placement_obstacles(self):
        problem = FakeProblem()
        ik_fn = get_ik_fn_old(problem)

        with patch('pybullet_tools.mobile_streams.solve_approach_ik', return_value=('cmd',)) as solve:
            result = ik_fn('left', 1, FakeSupportedPose(), FakeGrasp(), base_conf=object())

        self.assertEqual(result, ('cmd',))
        obstacles_here = solve.call_args.args[8]
        self.assertEqual(obstacles_here, [2])

    def test_ik_excludes_target_support_parent_from_placement_obstacles(self):
        problem = FakeProblem()
        problem.fixed = [2, 3, 5]
        problem.world.BODY_TO_OBJECT[5] = FakeBodyInfo(
            supporting_surface=FakeSurface((3, None, 5)))
        ik_fn = get_ik_fn_old(problem)

        with patch('pybullet_tools.mobile_streams.solve_approach_ik', return_value=('cmd',)) as solve:
            result = ik_fn('left', 1, FakeSupportedPose(), FakeGrasp(), base_conf=object())

        self.assertEqual(result, ('cmd',))
        obstacles_here = solve.call_args.args[8]
        self.assertEqual(obstacles_here, [2])

    def test_sample_bconf_restores_world_after_failed_ik(self):
        problem = FakeProblem()

        class FakeSaver:

            def __init__(self, bodies=None):
                problem.robot.saved_bodies = bodies
                pass

            def restore(self):
                problem.robot.restored = True

        def ir_sampler(*args):
            yield ('base-conf',)

        with patch('pybullet_tools.mobile_streams.WorldSaver', FakeSaver), \
             patch('pybullet_tools.mobile_streams.collided', return_value=False):
            result = list(sample_bconf(
                problem.world, problem.robot,
                ('left', 1, FakePose(), FakeGrasp()), FakePose.value, [], 'test',
                ir_sampler=ir_sampler, ik_fn=lambda *args: None,
                ir_max_attempts=1, learned=False))

        self.assertEqual(result, [])
        self.assertEqual(problem.robot.saved_bodies, [problem.robot.body, 1])
        self.assertTrue(problem.robot.restored)


if __name__ == '__main__':
    unittest.main()
