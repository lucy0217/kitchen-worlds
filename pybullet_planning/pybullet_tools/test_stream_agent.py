from __future__ import print_function

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pybullet_tools.stream_agent as stream_agent_module
from pybullet_tools.stream_agent import fix_init_given_goals, get_stream_map


class TestFixInitGivenGoals(unittest.TestCase):

    def test_in_goal_adds_containable_fact(self):
        init = []

        fixed = fix_init_given_goals([['in', 1, 5]], init)

        self.assertIn(('Containable', 1, 5), fixed)
        self.assertNotIn(('Containble', 1, 5), fixed)


class TestStreamMap(unittest.TestCase):

    def test_use_learned_ir_false_reaches_inverse_reachability_streams(self):
        def make_gen(*args, **kwargs):
            def gen(*_args, **_kwargs):
                if False:
                    yield None
            return gen

        ignored_factories = [
            'get_stable_list_gen', 'get_contain_list_gen', 'get_above_pose_gen',
            'sample_joint_position_gen', 'sample_joint_position_closed_gen',
            'get_grasp_list_gen', 'get_handle_grasp_gen', 'get_nudge_grasp_gen',
            'get_compute_pose_kin', 'get_compute_pose_rel_kin',
            'get_ik_fn_old', 'get_ik_rel_fn_old', 'get_ik_ungrasp_gen',
            'get_ik_pull_gen', 'get_ik_pull_with_link_gen',
            'get_pull_door_handle_motion_gen', 'get_pull_door_handle_with_link_motion_gen',
            'get_base_motion_gen', 'get_cfree_pose_pose_test', 'get_cfree_approach_pose_test',
            'get_cfree_rel_pose_pose_test', 'get_cfree_approach_rel_pose_test',
            'get_cfree_pose_between_test', 'get_cfree_traj_pose_test',
            'get_cfree_traj_pose_at_bconf_at_joint_position_test',
            'get_cfree_traj_pose_at_bconf_at_joint_position_at_link_pose_test',
            'get_cfree_btraj_pose_test', 'get_bconf_close_to_surface', 'get_reachable_test',
            'get_marker_grasp_gen', 'get_ik_ungrasp_mark_gen',
            'get_pull_marker_random_motion_gen', 'get_marker_pose_gen',
            'get_pull_marker_to_bconf_motion_gen', 'get_pull_marker_to_pose_motion_gen',
            'get_bconf_in_region_gen', 'get_pose_in_region_gen', 'get_bconf_in_region_test',
            'get_pose_in_region_test', 'get_pose_in_space_test',
        ]

        with ExitStack() as stack:
            for name in ignored_factories:
                stack.enter_context(patch.object(stream_agent_module, name, side_effect=make_gen))
            get_ik_gen_old = stack.enter_context(
                patch.object(stream_agent_module, 'get_ik_gen_old', side_effect=make_gen))
            get_ik_rel_gen_old = stack.enter_context(
                patch.object(stream_agent_module, 'get_ik_rel_gen_old', side_effect=make_gen))
            get_stream_map(
                p=SimpleNamespace(fixed=[], robot=object()), c=True, l={}, t=False,
                use_learned_ir=False, ir_max_attempts=10)

        ir_call = next(call for call in get_ik_gen_old.call_args_list
                       if call.kwargs.get('ir_only'))
        rel_ir_call = get_ik_rel_gen_old.call_args
        self.assertFalse(ir_call.kwargs['learned'])
        self.assertEqual(10, ir_call.kwargs['max_attempts'])
        self.assertFalse(rel_ir_call.kwargs['learned'])
        self.assertEqual(10, rel_ir_call.kwargs['max_attempts'])


if __name__ == '__main__':
    unittest.main()
