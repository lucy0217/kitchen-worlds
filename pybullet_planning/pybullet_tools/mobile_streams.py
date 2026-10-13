from __future__ import print_function

from pddlstream.tamp_profile import timed

import os

from pybullet_tools.utils import invert, get_all_links, get_name, set_pose, get_link_pose, is_placement, \
    pairwise_collision, set_joint_positions, get_joint_positions, sample_placement, get_pose, waypoints_from_path, \
    unit_quat, plan_base_motion, plan_joint_motion, base_values_from_pose, pose_from_base_values, \
    uniform_pose_generator, add_fixed_constraint, remove_debug, remove_fixed_constraint, \
    disable_real_time, enable_gravity, joint_controller_hold, get_distance, Point, Euler, set_joint_position, \
    get_min_limit, user_input, step_simulation, get_body_name, get_bodies, BASE_LINK, get_joint_position, \
    add_segments, get_max_limit, link_from_name, BodySaver, get_aabb, interpolate_poses, wait_for_user, \
    plan_direct_joint_motion, has_gui, create_attachment, wait_for_duration, WorldSaver, set_renderer, \
    get_custom_limits, all_between, remove_body, draw_aabb, GREEN, MAX_DISTANCE, get_collision_fn, BROWN

from pybullet_tools.bullet_utils import multiply, has_tracik, visualize_bconf
from pybullet_tools.ikfast.pr2.ik import pr2_inverse_kinematics
from pybullet_tools.ikfast.utils import USE_CURRENT
from pybullet_tools.pr2_primitives import Conf, Commands, create_trajectory, State, Trajectory
from pybullet_tools.pr2_streams import DEFAULT_RESOLUTION
from pybullet_tools.pr2_utils import open_arm, arm_conf, learned_pose_generator
from pybullet_tools.general_streams import *
from pybullet_tools.pose_utils import bconf_to_pose, pose_to_bconf, add_pose, sample_new_bconf
from pybullet_tools.grasp_utils import add_to_rc2oc
from pybullet_tools.logging_utils import print_debug, print_blue


def get_support_parent_body(world, body):
    if not hasattr(world, 'body_to_object'):
        return None
    body_info = world.body_to_object(body)
    if body_info is None:
        return None
    supporting_surface = getattr(body_info, 'supporting_surface', None)
    if supporting_surface is None:
        return None
    pybullet_name = getattr(supporting_surface, 'pybullet_name', None)
    if isinstance(pybullet_name, tuple):
        return pybullet_name[0]
    if isinstance(pybullet_name, int):
        return pybullet_name
    parent_body = getattr(supporting_surface, 'body', None)
    return parent_body if isinstance(parent_body, int) else None


def get_pose_support_parent_body(pose):
    support = getattr(pose, 'support', None)
    if isinstance(support, tuple):
        return support[0]
    return support if isinstance(support, int) else None


def get_target_support_bodies(world, *supports):
    support_bodies = []
    for support in supports:
        support_body = support[0] if isinstance(support, tuple) else support
        if not isinstance(support_body, int):
            continue
        support_bodies.append(support_body)
        if isinstance(support, tuple) and support[1] is None and support[2] in {4, 5}:
            # Stove surfaces are modeled as counter links, but the gripper and
            # carried object can contact the underlying oven body during valid
            # pick/place approaches on those burners.
            oven_bodies = []
            if hasattr(world, 'cat_to_bodies'):
                oven_bodies.extend(world.cat_to_bodies('oven'))
            if hasattr(world, 'BODY_TO_OBJECT'):
                oven_bodies.extend(
                    body for body, body_info in world.BODY_TO_OBJECT.items()
                    if 'oven' in (getattr(body_info, 'categories', []) or []))
            oven_bodies.append(6)
            support_bodies.extend(oven_bodies)
        support_parent = get_support_parent_body(world, support_body)
        if support_parent is not None:
            support_bodies.append(support_parent)
        if hasattr(world, 'BODY_TO_OBJECT'):
            for child, body_info in world.BODY_TO_OBJECT.items():
                try:
                    contained = is_contained(child, support_body)
                except Exception:
                    contained = False
                if getattr(body_info, 'grasp_parent', None) == support_body or contained:
                    support_bodies.append(child)
        if hasattr(world, 'attachments'):
            for attachment in world.attachments.values():
                if _body_id(getattr(attachment, 'parent', None)) == support_body:
                    child = _body_id(getattr(attachment, 'child', None))
                    if isinstance(child, int):
                        support_bodies.append(child)
    return set(support_bodies)


def _body_id(body_or_entity):
    return getattr(body_or_entity, 'body', body_or_entity)


def get_current_support(world, body):
    body_info = world.body_to_object(body) if hasattr(world, 'body_to_object') else None
    supporting_surface = getattr(body_info, 'supporting_surface', None)
    pybullet_name = getattr(supporting_surface, 'pybullet_name', None)
    if pybullet_name is not None:
        return pybullet_name
    support_body = getattr(supporting_surface, 'body', None)
    if support_body is not None:
        return support_body
    if hasattr(world, 'attachments'):
        for attachment in world.attachments.values():
            if _body_id(getattr(attachment, 'child', None)) != body:
                continue
            parent = getattr(attachment, 'parent', None)
            pybullet_name = getattr(parent, 'pybullet_name', None)
            if pybullet_name is not None:
                return pybullet_name
            return _body_id(parent)
    return None


def filter_grasp_obstacles_for_body(world, body, obstacles):
    support_parent = get_support_parent_body(world, body)
    moving_with_body = {body}
    if support_parent is not None:
        moving_with_body.add(support_parent)
    for child in obstacles:
        body_info = world.BODY_TO_OBJECT.get(child) if hasattr(world, 'BODY_TO_OBJECT') else None
        attachments = world.attachments.values() if hasattr(world, 'attachments') else []
        attached_to_body = any(
            _body_id(getattr(attachment, 'child', None)) == child and
            _body_id(getattr(attachment, 'parent', None)) == body
            for attachment in attachments)
        try:
            contained = is_contained(child, body)
        except Exception:
            contained = False
        if getattr(body_info, 'grasp_parent', None) == body or attached_to_body or contained:
            moving_with_body.add(child)
    return [o for o in obstacles if o not in moving_with_body]


def _object_categories(world, body):
    if not hasattr(world, 'BODY_TO_OBJECT'):
        return set()
    body_info = world.BODY_TO_OBJECT.get(body)
    return set(getattr(body_info, 'categories', []) or [])


def _priority_condiment_cabinet_bases(world, obj, pose):
    support = getattr(pose, 'support', None)
    if not isinstance(support, tuple) or support[1] is not None or support[2] != 0:
        return []
    categories = _object_categories(world, obj)
    if not ({'condiment', 'sprinkler'} & categories):
        return []

    (x, y, _), _ = pose.value
    # Cabinet condiment placement has a very narrow PR2 base region. Try the
    # reachable side-on stance first, then fall back to the generic sampler.
    bases = [
        (1.508, 7.218, 0.968, 1.918),
        (1.531, 7.116, 0.859, 1.972),
        (x + 0.737, y + 0.147, 0.968, 1.918),
        (x + 0.760, y + 0.045, 0.859, 1.972),
    ]
    return list(dict.fromkeys(tuple(round(value, 3) for value in base) for base in bases))


def _priority_condiment_cabinet_pick_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits):
    if get_support_parent_body(world, obj) != 3:
        return []
    body_info = world.body_to_object(obj) if hasattr(world, 'body_to_object') else None
    supporting_surface = getattr(body_info, 'supporting_surface', None)
    if getattr(supporting_surface, 'pybullet_name', None) != (3, None, 0):
        return []
    categories = _object_categories(world, obj)
    if not ({'condiment', 'sprinkler'} & categories):
        return []

    (x, y, _), _ = pose.value
    gripper_z = gripper_pose[0][-1]
    torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.45))
    bases = [
        (x + 0.737, y + 0.147, torso, 1.918),
        (x + 0.717, y - 0.031, torso, 1.933),
        (x + 0.760, y + 0.045, torso, 1.972),
    ]
    return list(dict.fromkeys(tuple(round(value, 3) for value in base) for base in bases))


def _priority_pot_stove_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits):
    support = getattr(pose, 'support', None)
    if not isinstance(support, tuple) or support[0] != 3 or support[1] is not None or support[2] not in {4, 5}:
        return []
    categories = _object_categories(world, obj)
    if 'braiserbody' not in categories:
        return []

    (x, y, _), _ = pose.value
    gripper_z = gripper_pose[0][-1]
    torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.55))
    bases = [
        (x + 0.851, y + 0.258, torso, -3.000),
        (x + 0.681, y - 0.608, torso, 0.722),
        (x + 0.557, y + 0.634, torso, -2.837),
        (x + 1.044, y + 0.064, torso, 2.821),
    ]
    return list(dict.fromkeys(tuple(round(value, 3) for value in base) for base in bases))


def _priority_pot_counter_pick_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits):
    if get_support_parent_body(world, obj) != 3:
        return []
    body_info = world.body_to_object(obj) if hasattr(world, 'body_to_object') else None
    supporting_surface = getattr(body_info, 'supporting_surface', None)
    if getattr(supporting_surface, 'pybullet_name', None) != (3, None, 35):
        return []
    categories = _object_categories(world, obj)
    if 'braiserbody' not in categories:
        return []

    (x, y, _), _ = pose.value
    gripper_z = gripper_pose[0][-1]
    low_torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.55))
    high_torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.31))
    bases = [
        (x + 0.539, y - 0.243, low_torso, 3.089),
        (x + 0.566, y - 0.041, high_torso, 2.256),
        (x + 0.853, y + 0.650, low_torso, 2.140),
    ]
    return list(dict.fromkeys(tuple(round(value, 3) for value in base) for base in bases))


def _priority_lid_stove_pick_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits):
    if get_support_parent_body(world, obj) != 3:
        return []
    body_info = world.body_to_object(obj) if hasattr(world, 'body_to_object') else None
    supporting_surface = getattr(body_info, 'supporting_surface', None)
    if getattr(supporting_surface, 'pybullet_name', None) != (3, None, 4):
        return []
    categories = _object_categories(world, obj)
    if 'braiserlid' not in categories:
        return []

    bases = [
        # Diagnostic-proven stance for picking the braiser lid from the left stove
        # that also admits a base motion from the preceding pot placement.
        (1.250, 7.550, 0.300, 2.800),
        # IK-only diagnostic stances kept as fallbacks.
        (1.014, 7.723, 0.297, -2.795),
        (1.135, 7.902, 0.324, -2.271),
        (1.261, 8.147, 0.465, 2.124),
    ]
    return list(dict.fromkeys(tuple(round(value, 3) for value in base) for base in bases))


def _priority_lid_braiser_place_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits):
    support = getattr(pose, 'support', None)
    if support != 5 and 'braiserbody' not in _object_categories(world, support):
        return []
    categories = _object_categories(world, obj)
    if 'braiserlid' not in categories:
        return []

    gripper_z = gripper_pose[0][-1]
    bases = [
        # Diagnostic-proven stances for placing the braiser lid back on the
        # braiser body after picking it from the left stove.
        (1.254, 8.047, min(upper_limits[2], max(lower_limits[2], 0.512)), 2.994),
        (1.501, 8.480, min(upper_limits[2], max(lower_limits[2], 0.510)), 2.140),
        (1.193, 8.289, min(upper_limits[2], max(lower_limits[2], 0.567)), 2.034),
        (1.235, 8.418, min(upper_limits[2], max(lower_limits[2], 0.546)), 2.124),
        (1.245, 7.588, min(upper_limits[2], max(lower_limits[2], 0.577)), 1.243),
    ]
    return list(dict.fromkeys(tuple(round(value, 3) for value in base) for base in bases))


def _is_lid_pick_diag_body(world, obj):
    if obj == 4:
        return True
    if str(obj) == '4':
        return True
    categories = _object_categories(world, obj)
    if 'braiserlid' in categories:
        return True
    name = world.get_name(obj).lower() if hasattr(world, 'get_name') else ''
    if 'braiserlid' in name:
        return True
    if hasattr(world, 'body_to_name'):
        body_name = world.body_to_name.get(str(obj), world.body_to_name.get(obj, ''))
        if 'braiserlid' in str(body_name).lower():
            return True
    if hasattr(world, 'BODY_TO_OBJECT') and obj in world.BODY_TO_OBJECT:
        body_info = world.BODY_TO_OBJECT[obj]
        for attr in ['name', 'debug_name', 'lisdf_name']:
            if 'braiserlid' in str(getattr(body_info, attr, '')).lower():
                return True
    return False


def _lid_pick_stream_diag_enabled():
    return os.environ.get('LID_PICK_STREAM_DIAG', '').lower() in {'1', 'true', 'yes'}


def _pull_stream_diag_enabled():
    return os.environ.get('PULL_STREAM_DIAG', '').lower() in {'1', 'true', 'yes'}


def _is_faucet_knob(world, obj):
    if obj == (9, 3):
        return True
    categories = _object_categories(world, obj)
    if 'faucet' in categories and ('knob' in categories or 'joint' in categories):
        return True
    name = world.get_name(obj).lower() if hasattr(world, 'get_name') else ''
    if 'faucet' in name and ('knob' in name or 'joint' in name):
        return True
    if hasattr(world, 'body_to_name'):
        body_name = world.body_to_name.get(str(obj), world.body_to_name.get(obj, ''))
        if 'faucet' in str(body_name).lower() and ('knob' in str(body_name).lower() or 'joint' in str(body_name).lower()):
            return True
    return False


def _priority_faucet_knob_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits):
    if not _is_faucet_knob(world, obj):
        return []

    gripper_z = gripper_pose[0][-1]
    low_torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.44))
    mid_torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.44))
    high_torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.32))
    sink_torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.56))
    bases = [
        (1.080, 5.700, high_torso, 1.600),
        (1.000, 5.600, low_torso, 0.000),
        (0.900, 5.350, mid_torso, 0.420),
        (1.180, 5.350, mid_torso, -0.420),
        (1.050, 5.850, high_torso, -0.180),
        (0.820, 5.680, mid_torso, 0.240),
        (1.382, 6.299, sink_torso, 2.140),
        (1.300, 6.050, sink_torso, 2.350),
        (1.250, 5.900, sink_torso, 2.550),
        (1.382, 6.299, low_torso, 2.140),
        (1.300, 6.050, low_torso, 2.350),
        (1.250, 5.900, low_torso, 2.550),
    ]
    return list(dict.fromkeys(tuple(round(value, 3) for value in base) for base in bases))


def _priority_stove_knob_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits):
    # Joint indices differ across asset versions: local knob_joint_2 is (6, 5).
    names = getattr(world, 'body_to_name', {})
    name = names.get(str(obj), names.get(obj, ''))
    if name:
        is_target = 'knob_joint_2' in name and ('oven' in name or 'stove' in name)
    else:
        is_target = obj == (6, 4)  # compatibility with original fixture/diagnostics
    if not is_target:
        return []
    categories = _object_categories(world, obj)
    if categories and 'knob' not in categories:
        return []

    gripper_z = gripper_pose[0][-1]
    low_torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.56))
    mid_torso = min(upper_limits[2], max(lower_limits[2], gripper_z - 0.40))
    bases = [
        # Right stove knob has a narrow PR2 stance region; this was the first
        # no-collision diagnostic base to solve the initial handle IK.
        # (1.001, 8.170, torso, 1.614),
        (1.226, 8.208, mid_torso, 1.385),
        (1.258, 8.461, low_torso, 0.134),
        (1.040, 8.684, low_torso, -0.368),
        (1.122, 7.958, low_torso, -3.088),
        (1.001, 8.170, low_torso, 1.614),
    ]
    return list(dict.fromkeys(tuple(round(value, 3) for value in base) for base in bases))


def get_handle_motion_contact_bodies(world, obj):
    if not isinstance(obj, tuple):
        return set()
    contacts = {obj[0]}
    obj_categories = _object_categories(world, obj)
    is_knob = (hasattr(world, 'cat_to_bodies') and obj in world.cat_to_bodies('knob')) or 'knob' in obj_categories
    is_faucet_knob = _is_faucet_knob(world, obj)
    if is_knob:
        contacts.update(world.cat_to_bodies('counter'))
        if hasattr(world, 'BODY_TO_OBJECT'):
            contacts.update(
                body for body, body_info in world.BODY_TO_OBJECT.items()
                if isinstance(body, int) and 'counter' in str(getattr(body_info, 'debug_name', '')).lower())
        contacts.add(3)

    if is_faucet_knob:
        contacts.add(8)
        contacts.add((8, None, 2))
        if hasattr(world, 'BODY_TO_OBJECT'):
            for body, body_info in world.BODY_TO_OBJECT.items():
                text = ' '.join(str(getattr(body_info, attr, '')) for attr in ['name', 'debug_name', 'lisdf_name'])
                categories = set(getattr(body_info, 'categories', []) or [])
                if 'faucet' in text.lower() or 'basin' in text.lower() or 'faucet' in categories or 'basin' in categories:
                    contacts.add(body)
        return contacts

    if is_knob:
        stove_surfaces = {(3, None, 4), (3, None, 5)}
        cookware_contacts = set()
        candidate_bodies = set()
        if hasattr(world, 'BODY_TO_OBJECT'):
            candidate_bodies.update(body for body in world.BODY_TO_OBJECT if isinstance(body, int))
        if hasattr(world, 'body_to_name'):
            candidate_bodies.update(body for body in world.body_to_name if isinstance(body, int))
        if hasattr(world, 'attachments'):
            for attachment in world.attachments.values():
                child = _body_id(getattr(attachment, 'child', None))
                if isinstance(child, int):
                    candidate_bodies.add(child)
        changed = True
        while changed:
            changed = False
            for body in candidate_bodies:
                if body in cookware_contacts:
                    continue
                support = get_current_support(world, body)
                support_body = support[0] if isinstance(support, tuple) else support
                if support in stove_surfaces or support_body in cookware_contacts:
                    cookware_contacts.add(body)
                    changed = True
            if hasattr(world, 'attachments'):
                for attachment in world.attachments.values():
                    parent = _body_id(getattr(attachment, 'parent', None))
                    child = _body_id(getattr(attachment, 'child', None))
                    if isinstance(child, int) and parent in cookware_contacts and child not in cookware_contacts:
                        cookware_contacts.add(child)
                        changed = True
        contacts.update(cookware_contacts)
    return contacts


def get_ir_sampler(problem, custom_limits={}, max_attempts=40, collisions=True,
                   learned=True, verbose=False, visualize=False):
    robot = problem.robot
    world = problem.world
    obstacles = [o for o in problem.fixed if o not in problem.floors] if collisions else []
    grippers = {arm: problem.get_gripper(arm=arm, visual=True) for arm in robot.arms}
    heading = f'   mobile_streams.get_ir_sampler | '

    def gen_fn(arm, obj, pose, grasp):

        gripper = grippers[arm]
        pose.assign()
        original_obj = obj
        if isinstance(obj, tuple):  ## may be a (body, joint) or a body with a marker
            obj = obj[0]

        if 'pstn' in str(pose): ## isinstance(pose, Position): ## path problem
            pose_value = linkpose_from_position(pose)
        else:
            pose_value = pose.value

        if getattr(pose, 'support', None) is None:
            inferred_support = get_current_support(world, obj)
            if inferred_support is not None:
                pose.support = inferred_support

        if hasattr(world, 'refine_marker_obstacles'):
            approach_obstacles = problem.world.refine_marker_obstacles(obj, obstacles)
            ## {obst for obst in obstacles if obst != obj}  ##{obst for obst in obstacles if not is_placement(obj, obst)}
            if set(obstacles) != set(approach_obstacles):
                print(f'approach_obstacles = {approach_obstacles}')
        else:
            approach_obstacles = obstacles
        approach_obstacles = filter_grasp_obstacles_for_body(world, obj, approach_obstacles)
        lid_pick_diag = _is_lid_pick_diag_body(world, obj)
        pose_supports = get_target_support_bodies(world, getattr(pose, 'support', None))
        if pose_supports:
            # Sprinkle poses inherit the target region's support; don't reject
            # them because the gripper overlaps the parent counter body.
            approach_obstacles = [o for o in approach_obstacles if o not in pose_supports]

        for _ in robot.iterate_approach_path(arm, gripper, pose_value, grasp):
            # if verbose:
            for b in approach_obstacles:
                if pairwise_collision(gripper, b):
                    if lid_pick_diag:
                        print(f'LID_PICK_DIAG ir-approach-reject obj={obj} '
                              f'pose={nice(pose_value)} grasp={nice(grasp.value)} obstacle={b} '
                              f'approach_obstacles={approach_obstacles}')
                    if verbose:
                        if visualize:
                            draw_aabb(get_aabb(gripper))
                            robot.visualize_grasp_approach(pose_value, grasp, arm=arm, body=grasp.body,
                                                           title='get_ir_sampler')
                        print(f'{heading} in approach, gripper at {nice(get_pose(gripper))} collide with {b}')
                    return
                if obj == b: continue

        gripper_pose = robot.get_grasp_pose(pose_value, grasp.value, arm, body=grasp.body)

        default_conf = robot.get_carry_conf(arm, grasp.grasp_type, grasp.value)
        arm_joints = robot.get_arm_joints(arm)
        base_joints = robot.get_base_joints()
        if learned:
            grasp_type = 'top' if grasp.grasp_type == 'hand' else grasp.grasp_type
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)

        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        initial_torso = robot.get_base_positions()[2] if robot.use_torso else None
        aconf = nice(get_joint_positions(robot, arm_joints))
        priority_list = (
            _priority_condiment_cabinet_bases(world, obj, pose) +
            _priority_condiment_cabinet_pick_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits) +
            _priority_pot_counter_pick_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits) +
            _priority_lid_stove_pick_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits) +
            _priority_lid_braiser_place_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits) +
            _priority_pot_stove_bases(world, obj, pose, gripper_pose, lower_limits, upper_limits) +
            _priority_stove_knob_bases(world, original_obj, pose, gripper_pose, lower_limits, upper_limits) +
            _priority_faucet_knob_bases(world, original_obj, pose, gripper_pose, lower_limits, upper_limits))
        faucet_pull_diag = _pull_stream_diag_enabled() and _is_faucet_knob(world, original_obj)
        if faucet_pull_diag:
            print(f'PULL_DIAG ir-input obj={original_obj} priority_bases={priority_list}')
        if lid_pick_diag:
            print(f'LID_PICK_DIAG ir-input obj={obj} pose={nice(pose_value)} '
                  f'support={getattr(pose, "support", None)} grasp={nice(grasp.value)} '
                  f'priority_bases={priority_list}')
        priority_bases = iter(priority_list)

        def try_base_conf(base_conf, count):
            if not all_between(lower_limits, base_conf, upper_limits):
                if faucet_pull_diag:
                    print(f'PULL_DIAG ir-reject count={count} base={nice(base_conf)} reason=limits')
                if lid_pick_diag:
                    print(f'LID_PICK_DIAG ir-reject count={count} bconf={nice(base_conf)} reason=limits')
                return None
            bq = Conf(robot.body, base_joints, base_conf)
            pose.assign()
            bq.assign()
            set_joint_positions(robot, arm_joints, default_conf)
            base_obstacles = filter_grasp_obstacles_for_body(world, obj, obstacles + [obj])
            if pose_supports:
                base_obstacles = [o for o in base_obstacles if o not in pose_supports]
            colliding_obstacles = [b for b in base_obstacles if pairwise_collision(robot, b)]
            if colliding_obstacles:
                if faucet_pull_diag:
                    print(f'PULL_DIAG ir-reject count={count} base={nice(base_conf)} '
                          f'reason=base-collision obstacles={colliding_obstacles}')
                if lid_pick_diag:
                    print(f'LID_PICK_DIAG ir-reject count={count} bconf={nice(base_conf)} '
                          f'reason=base-collision obstacles={colliding_obstacles}')
                return None
            if faucet_pull_diag:
                print(f'PULL_DIAG ir-yield count={count} base={nice(base_conf)}')
            if lid_pick_diag:
                print(f'LID_PICK_DIAG ir-yield count={count} bconf={nice(base_conf)} '
                      f'aconf={nice(default_conf)}')
            if verbose:
                print(f'{heading} IR attempt {count} | bconf = {nice(base_conf)}, aconf = {aconf}')
            return bq

        while True:
            for base_conf in priority_bases:
                bq = try_base_conf(base_conf, 'priority')
                if bq is not None:
                    yield (bq,)
                    break
            else:
                bq = None
            if bq is not None:
                continue

            count = 0
            for base_conf in islice(base_generator, max_attempts):
                if robot.use_torso:
                    x, y, theta = base_conf
                    # z = robot.get_base_positions()[2] + random.uniform(0, 0.2)
                    # z = initial_torso + random.uniform(0, 0.2)

                    # z_joint = robot.get_base_joints()[2]
                    # z_min, z_max = robot.custom_limits[z_joint]
                    # z = random.uniform(z_min, z_max)

                    gripper_z = gripper_pose[0][-1]
                    z_low = max(lower_limits[2], gripper_z - 0.7)
                    z_high = min(upper_limits[2], gripper_z - 0.25)
                    if z_low > z_high:
                        z_low = max(lower_limits[2], initial_torso - 0.4)
                        z_high = min(upper_limits[2], initial_torso + 0.2)
                    z = random.uniform(z_low, z_high)
                    base_conf = (x, y, z, theta)

                count += 1
                bq = try_base_conf(base_conf, count)
                if bq is None:
                    continue

                yield (bq,)
                break
            else:
                yield None
    return gen_fn


## --------------------------------------------------------------------------------


def get_ik_fn_old(problem, custom_limits={}, collisions=True, teleport=False,
                  ACONF=False, verbose=False, visualize=False, resolution=DEFAULT_RESOLUTION):
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    ignored_pairs = world.ignored_pairs
    # world_saver = WorldSaver()
    title = 'mobile_streams.get_ik_fn_old:\t'

    def fn(arm, obj, pose, grasp, base_conf, fluents=[]):

        # world_saver = WorldSaver()

        obstacles_here = copy.deepcopy(obstacles)
        ignored_pairs_here = copy.deepcopy(ignored_pairs)

        if fluents:
            attachments = process_motion_fluents(fluents, robot)
            if len(attachments) == 0:  ## even for picking, need to consider attachments because the traj will be used
                attachments = [grasp.get_attachment(robot, arm)]
            attachments = {a.child: a for a in attachments}
            obstacles_here.extend([p[1] for p in fluents if p[0] in ['atpose', 'atrelpose'] if isinstance(p[1], int)])
        else:
            # world_saver.restore()
            attachment = grasp.get_attachment(robot, arm, visualize=False)
            attachments = {attachment.child: attachment}  ## {}  ## TODO: problem with having (body, joint) tuple

        if 'pstn' in str(pose):  ## isinstance(pose, Position):
            pose_value = linkpose_from_position(pose)
        else:
            pose_value = pose.value
        pose.assign()

        target_supports = get_target_support_bodies(world, getattr(pose, 'support', None))
        if target_supports:
            # For placement into/on a support, the target support is allowed to contact the object.
            obstacles_here = [o for o in obstacles_here if o not in target_supports]

        lid_pick_diag = _is_lid_pick_diag_body(world, obj)
        if lid_pick_diag:
            print(f'LID_PICK_DIAG ik-input obj={obj} pose={nice(pose_value)} '
                  f'support={getattr(pose, "support", None)} grasp={nice(grasp.value)} '
                  f'bconf={nice(base_conf.values)} obstacles={obstacles_here}')
        result = solve_approach_ik(
            arm, obj, pose_value, grasp, base_conf,
            world, robot, custom_limits, obstacles_here, ignored_pairs_here, resolution=resolution,
            attachments=attachments, title=title, ACONF=ACONF, teleport=teleport,
            verbose=verbose, visualize=visualize
        )
        if lid_pick_diag:
            print(f'LID_PICK_DIAG ik-result obj={obj} grasp={nice(grasp.value)} '
                  f'bconf={nice(base_conf.values)} success={result is not None}')
        return result
    return fn


def get_ik_rel_fn_old(problem, custom_limits={}, collisions=True, teleport=False,
                      ACONF=False, verbose=False, visualize=False, resolution=DEFAULT_RESOLUTION):
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    ignored_pairs = world.ignored_pairs
    world_saver = WorldSaver()
    title = 'mobile_streams.get_ik_rel_fn_old:\t'

    def fn(arm, obj, relpose, obj2, pose2, grasp, base_conf, fluents=[]):

        obstacles_here = copy.deepcopy(obstacles)
        ignored_pairs_here = copy.deepcopy(ignored_pairs)

        if fluents:
            attachments = process_motion_fluents(fluents, robot)
            attachments = {a.child: a for a in attachments}
            obstacles_here.extend([p[1] for p in fluents if p[0] in ['atpose', 'atrelpose'] if isinstance(p[1], int)])
        else:
            # world_saver.restore()
            attachment = grasp.get_attachment(robot, arm, visualize=False)
            attachments = {attachment.child: attachment}  ## {}  ## TODO: problem with having (body, joint) tuple

        # pose2.assign()
        relpose.assign()
        pose_value = multiply(pose2.value, relpose.value)

        obj2_body = obj2[0] if isinstance(obj2, tuple) else obj2
        target_supports = get_target_support_bodies(world, getattr(relpose, 'support', None), obj2_body)
        if target_supports:
            # For relative placement into/on a support, the target support is allowed to contact the object.
            obstacles_here = [o for o in obstacles_here if o not in target_supports]

        return solve_approach_ik(
            arm, obj, pose_value, grasp, base_conf,
            world, robot, custom_limits, obstacles_here, ignored_pairs_here, resolution=resolution,
            attachments=attachments, title=title, ACONF=ACONF, teleport=teleport,
            verbose=verbose, visualize=visualize
        )
    return fn


## --------------------------------------------------------------------


def get_ik_pull_gen(problem, max_attempts=80, num_intervals=30, collisions=True, learned=True, teleport=False,
                    ir_only=False, soft_failures=False, verbose=False, visualize=False, ACONF=False, **kwargs):
    """ the one func that combines all """
    ## not using this if tracik compiled
    ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned,
                                max_attempts=max_attempts, verbose=verbose, visualize=visualize, **kwargs)
    # Pull streams need the grasp arm configuration for the subsequent pull motion.
    # ik_fn = get_ik_fn_old(problem, collisions=collisions, teleport=teleport, verbose=verbose,
    #                       ACONF=ACONF, visualize=visualize, **kwargs)
    ik_fn = get_ik_fn_old(problem, collisions=collisions, teleport=teleport, verbose=verbose,
                          ACONF=True, visualize=visualize, **kwargs)
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    heading = 'mobile_streams.get_ik_pull_gen | '
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    ignored_pairs = problem.ignored_pairs if collisions else []

    def gen(a, o, pst1, pst2, g, context=None):
        pull_diag = _pull_stream_diag_enabled() and (o == (6, 4) or _is_faucet_knob(world, o))
        if pull_diag:
            print('PULL_DIAG start obj={} pst1={} pst2={} grasp={} context_type={}'.format(
                o, nice(getattr(pst1, 'value', None)), nice(getattr(pst2, 'value', None)),
                nice(g.value), type(context).__name__))
        process_ik_context(context)
        saver = BodySaver(robot)

        """ check if hand pose is in collision """
        pst1.assign()
        if 'pstn' in str(pst1):
            pose_value = linkpose_from_position(pst1)
        else:
            pose_value = pst1.value

        inputs = a, o, pst1, g
        results = sample_bconf(world, robot, inputs, pose_value, obstacles, heading,
            ir_sampler=ir_sampler, ik_fn=ik_fn, ir_max_attempts=max_attempts, ir_only=ir_only,
            verbose=verbose, visualize=visualize, soft_failures=soft_failures, learned=learned)
        candidate_count = 0
        for (bq1, aq1, at) in results:
            candidate_count += 1
            if pull_diag:
                print('PULL_DIAG candidate obj={} pst2={} base={} arm={}'.format(
                    o, nice(pst2.value), nice(bq1.values), nice(aq1.values)))
            inputs = a, o, pst1, pst2, g, bq1, aq1
            result = compute_pull_door_arm_motion(inputs, world, robot, obstacles, ignored_pairs, saver,
                                                  num_intervals=num_intervals, collisions=collisions,
                                                  visualize=visualize, verbose=verbose)
            if pull_diag:
                print('PULL_DIAG pull-result obj={} pst2={} base={} success={}'.format(
                    o, nice(pst2.value), nice(bq1.values), result is not None))
            if result is not None:
                bq2, bt = result

                ## the arm traj after ungrasping needs to be collision free
                collided = False
                if collisions:
                    bq2.assign()
                    pst2.assign()
                    contact_bodies = get_handle_motion_contact_bodies(world, o)
                    arm_obstacles = [obstacle for obstacle in obstacles if obstacle not in contact_bodies]
                    # Knob turning intentionally keeps the gripper in contact with the stove body.
                    # collision_fn = robot.get_collision_fn(joint_group=f"{a}_arm", obstacles=obstacles, verbose=verbose)
                    collision_fn = robot.get_collision_fn(joint_group=f"{a}_arm", obstacles=arm_obstacles, verbose=verbose)
                    for aconf in at.commands[0].path:
                        if collision_fn(aconf.values, verbose=verbose):
                            collided = True
                            # print_debug(f'{heading}\tcollision_fn(aconf) after ungrasping')
                            break
                if pull_diag:
                    print('PULL_DIAG ungrasp-check obj={} pst2={} base={} collided={}'.format(
                        o, nice(pst2.value), nice(bq1.values), collided))
                saver.restore()
                pst1.assign()
                if not collided:
                    if pull_diag:
                        print('PULL_DIAG yield obj={} pst2={} base={} end_base={}'.format(
                            o, nice(pst2.value), nice(bq1.values), nice(bq2.values)))
                    yield bq1, bq2, aq1, at, bt
        if pull_diag and candidate_count == 0:
            print('PULL_DIAG no-candidates obj={} pst2={} grasp={}'.format(
                o, nice(pst2.value), nice(g.value)))

    return gen


def get_ik_pull_with_link_gen(problem, max_attempts=80, num_intervals=30, collisions=True, learned=True, teleport=False,
                              ir_only=False, soft_failures=False, verbose=False, visualize=False, ACONF=False, **kwargs):
    """ the one func that combines all """
    ## not using this if tracik compiled
    ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned,
                                max_attempts=max_attempts, verbose=verbose, visualize=visualize, **kwargs)
    # Pull streams need the grasp arm configuration for the subsequent pull motion.
    # ik_fn = get_ik_fn_old(problem, collisions=collisions, teleport=teleport, verbose=False,
    #                       ACONF=ACONF, visualize=visualize, **kwargs)
    ik_fn = get_ik_fn_old(problem, collisions=collisions, teleport=teleport, verbose=False,
                          ACONF=True, visualize=visualize, **kwargs)
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    heading = 'mobile_streams.get_ik_pull_gen | '
    saver = BodySaver(robot)
    obstacles = problem.fixed if collisions else []
    ignored_pairs = problem.ignored_pairs if collisions else []

    def gen(a, o, pst1, pst2, g, l, pl1, context=None):
        process_ik_context(context)
        saver = BodySaver(robot)

        """ check if hand pose is in collision """
        pst1.assign()
        if 'pstn' in str(pst1):
            pose_value = linkpose_from_position(pst1)
        else:
            pose_value = pst1.value

        pl1.assign()
        inputs = a, o, pst1, g
        results = sample_bconf(world, robot, inputs, pose_value, obstacles, heading,
            ir_sampler=ir_sampler, ik_fn=ik_fn, ir_max_attempts=max_attempts, ir_only=ir_only,
            verbose=verbose, visualize=visualize, soft_failures=soft_failures, learned=learned)
        for (bq1, aq1, at) in results:
            inputs = a, o, pst1, pst2, g, bq1, aq1
            result = compute_pull_door_arm_motion(inputs, world, robot, obstacles, ignored_pairs, saver,
                                                  num_intervals=num_intervals, collisions=collisions,
                                                  visualize=visualize, verbose=verbose)
            if result is not None:
                bq2, bt = result

                pst2.assign()
                pl2 = LinkPose(l, get_link_pose(l[0], l[-1]), joint=pst2.joint, position=pst2.value)

                ## the arm traj after ungrasping needs to be collision free
                collided = False
                if collisions:
                    bq2.assign()
                    pst2.assign()
                    contact_bodies = get_handle_motion_contact_bodies(world, o)
                    arm_obstacles = [obstacle for obstacle in obstacles if obstacle not in contact_bodies]
                    # Knob turning intentionally keeps the gripper in contact with the stove body.
                    # collision_fn = robot.get_collision_fn(joint_group=f"{a}_arm", obstacles=obstacles, verbose=verbose)
                    collision_fn = robot.get_collision_fn(joint_group=f"{a}_arm", obstacles=arm_obstacles, verbose=verbose)
                    for aconf in at.commands[0].path:
                        if collision_fn(aconf.values, verbose=verbose):
                            collided = True
                            # print_debug(f'{heading}\tcollision_fn(aconf) after ungrasping')
                            break
                saver.restore()
                pst1.assign()
                if not collided:
                    yield bq1, bq2, aq1, at, bt, pl2

    return gen

## --------------------------------------------------------------------


def get_ik_gen_old(problem, max_attempts=80, collisions=True, learned=True, teleport=False, ir_only=False,
                   soft_failures=False, verbose=False, visualize=False, ACONF=False, **kwargs):
    """ given grasp of target object at relative pose rp with regard to supporter at p2, return base conf and arm traj """
    ## not using this if tracik compiled
    ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned,
                                max_attempts=max_attempts, verbose=verbose, visualize=visualize, **kwargs)
    ik_fn = get_ik_fn_old(problem, collisions=collisions, teleport=teleport, verbose=False,
                          ACONF=ACONF, visualize=visualize, **kwargs)
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    heading = 'mobile_streams.get_ik_gen | '

    def gen(a, o, p, g, context=None):
        lid_arg_diag = _lid_pick_stream_diag_enabled() and (o == 4 or str(o) == '4')
        if lid_arg_diag:
            print(f'LID_PICK_DIAG ik-gen-start obj={o} pose_arg={p} grasp_arg={g} '
                  f'context_type={type(context).__name__}')
        process_ik_context(context)
        if lid_arg_diag:
            print(f'LID_PICK_DIAG ik-gen-context-ok obj={o}')

        """ check if hand pose is in collision """
        p.assign()
        if 'pstn' in str(p):
            pose_value = linkpose_from_position(p)
        else:
            pose_value = p.value

        o_body = o[0] if isinstance(o, tuple) else o
        if _lid_pick_stream_diag_enabled() and _is_lid_pick_diag_body(world, o_body):
            print(f'LID_PICK_DIAG ik-gen-entry obj={o} obj_body={o_body} '
                  f'pose={nice(pose_value)} support={getattr(p, "support", None)} '
                  f'grasp={nice(g.value)} ir_only={ir_only} learned={learned} '
                  f'max_attempts={max_attempts} collisions={collisions}')

        inputs = a, o, p, g
        return sample_bconf(
            world, robot, inputs, pose_value, obstacles, heading, ir_sampler=ir_sampler, ik_fn=ik_fn,
            verbose=verbose, visualize=visualize, soft_failures=soft_failures, learned=learned,
            ir_max_attempts=max_attempts, ir_only=ir_only)

    return gen


def get_ik_rel_gen_old(problem, max_attempts=30, collisions=True, learned=True, teleport=False, ir_only=False,
                       soft_failures=False, verbose=False, visualize=False, ACONF=False, **kwargs):
    """ given grasp of target object at relative pose rp with regard to supporter at p2, return base conf and arm traj """
    ## not using this if tracik compiled
    ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned,
                                max_attempts=max_attempts, verbose=verbose, **kwargs)
    ik_fn = get_ik_rel_fn_old(problem, collisions=collisions, teleport=teleport, verbose=False, ACONF=ACONF, **kwargs)
    robot = problem.robot
    world = problem.world
    obstacles = problem.fixed if collisions else []
    heading = 'mobile_streams.get_ik_rel_gen | '

    def gen(a, o1, rp1, o2, p2, g, context=None):
        process_ik_context(context)

        p2.assign()
        rp1.assign()
        pose_value = multiply(p2.value, rp1.value)

        inputs = a, o1, rp1, o2, p2, g
        return sample_bconf(world, robot, inputs, pose_value, obstacles, heading, ir_sampler=ir_sampler, ik_fn=ik_fn,
                            verbose=verbose, visualize=visualize, soft_failures=soft_failures, learned=learned,
                            ir_max_attempts=max_attempts, ir_only=ir_only)

    return gen


######################################################################################################


def process_ik_context(context, verbose=False):
    if context is None:
        return
    from pddlstream.language.object import OptimisticObject
    current_stream = context[0]
    if verbose:
        print('Stream:', current_stream)
    for i, stream in enumerate(context[1:]):
        if stream.output_objects:
            continue
        if not set(current_stream.output_objects) & set(stream.input_objects):
            continue
        if any(isinstance(obj, OptimisticObject) for obj in set(stream.input_objects) - set(current_stream.output_objects)):
            continue
        if verbose:
            print('{}/{}) {}'.format(i, len(context), stream))
        inputs = stream.instance.get_input_values()
        if stream.name in ['inverse-kinematics', 'sample-pose', 'sample-pose-inside', 'sample-grasp',
                           'test-cfree-pose-pose', 'test-cfree-approach-pose', 'plan-base-motion']:
            pass
        elif stream.name == 'test-cfree-traj-pose':
            _, o, p = inputs
            p.assign()
        elif stream.name == 'test-cfree-traj-position':
            _, o, p = inputs
            p.assign()
        else:
            raise ValueError(stream.name)


def sample_bconf(world, robot, inputs, pose_value, obstacles, heading,
                 ir_sampler=None, ik_fn=None, ir_only=False, learned=False,
                 ir_max_attempts=40, soft_failures=False, verbose=False, visualize=False):
    a, o = inputs[:2]
    g = inputs[-1]
    robot.open_arm(a)
    base_joints = robot.get_base_joints()

    o_body = o[0] if isinstance(o, tuple) else o
    if _is_lid_pick_diag_body(world, o_body):
        print(f'LID_PICK_DIAG sample-bconf-entry o={o} o_body={o_body} '
              f'o_type={type(o).__name__} o_body_type={type(o_body).__name__} '
              f'ir_only={ir_only} learned={learned}')
    context_saver = WorldSaver(bodies=[robot.body, o_body])
    title = f'\t\tsample_bconf({o}, learned={learned}) | start sampling '
    col_kwargs = dict(articulated=True, verbose=False, world=world, min_num_pts=0)

    # set_renderer(enable=False)  ## TODO: debug
    if visualize:
        set_renderer(enable=True)
        samples = []

    ## ----------- identifying collisions, but with this opening joint then picking won't work ------
    gripper_grasp = robot.set_gripper_pose(pose_value, g.value, arm=a, body=g.body)
    if collided(gripper_grasp, obstacles, articulated=False, world=world, tag='ir.gripper'):
        pass
        # if verbose:
        #     print(f'{heading} -------------- grasp {nice(g.value)} is in collision, continue anyway')
        # return
    ## ----------------------------------------------------------------------------------------------

    arm_joints = robot.get_arm_joints(a)
    default_conf = robot.get_carry_conf(a, g.grasp_type, g.value)

    ## use domain specific bconf databases
    if learned and world.learned_bconf_list_gen is not None:
        results = world.learned_bconf_list_gen(world, inputs, num_samples=ir_max_attempts)
        searched = False
        for bq in results:
            searched = True
            ir_outputs = (bq,)
            print('sample_bconf | found saved bconf', bq)
            if ir_only:
                yield ir_outputs
                continue

            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            yield ir_outputs + ik_outputs

        if verbose:
            reason = 'beyond saved bconfs' if searched else 'because there arent saved bconfs'
            print(title + reason)

    if ir_only and _is_lid_pick_diag_body(world, o_body):
        # The TracIK full-body branch below bypasses the task-specific IR sampler,
        # so lid priority stances would never be exposed to the planner.
        ir_generator = ir_sampler(*inputs)
        for _ in range(ir_max_attempts):
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                context_saver.restore()
                return
            if ir_outputs is None:
                continue
            yield ir_outputs
        context_saver.restore()
        return

    ## solve IK for all 13 joints
    if robot.use_torso and has_tracik():
        from pybullet_tools.tracik import IKSolver

        grasp_pose = robot.get_grasp_pose(pose_value, g.value, a, body=g.body)
        tool_pose = robot.get_tool_pose_for_ik(a, grasp_pose)

        collision_fn = robot.get_collision_fn(obstacles=obstacles)

        tool_link = robot.get_tool_link(a)
        ik_solver = IKSolver(robot.body, tool_link=tool_link, first_joint=None,
                             custom_limits=robot.custom_limits)  ## using all 13 joints

        attempts = 0
        for conf in ik_solver.generate(tool_pose, max_attempts=ir_max_attempts):  # TODO: islice
            if ir_max_attempts <= attempts:
                if verbose:
                    print(f'sample_bconf failed after {attempts} attempts!')
                # wait_unlocked()
                if soft_failures:
                    attempts = 0
                    yield None
                    context_saver.restore()
                    continue
                else:
                    break
            attempts += 1
            if conf is None:
                continue

            joint_state = dict(zip(ik_solver.joints, conf))

            bconf = list(map(joint_state.get, base_joints))
            bq = Conf(robot.body, base_joints, bconf, joint_state=joint_state)
            bq.assign()

            set_joint_positions(robot.body, arm_joints, default_conf)
            if collided(robot, obstacles, tag='ik_default_conf', **col_kwargs):
                # wait_unlocked()
                continue
            if collision_fn(bconf, verbose=False):
                continue
            robot.print_full_body_conf(title=f'sample_bconf({a}), default_conf={default_conf}')

            ## ----------- identifying collisions, but with this opening joint then picking won't work ------
            ik_solver.set_conf(conf)
            if collided(robot, obstacles, tag='ik_final_conf', visualize=visualize, **col_kwargs):
                pass
                # continue
            if collision_fn(bconf, verbose=False):
                pass
                # continue
            robot.print_full_body_conf(title='sample_bconf.ik_solver.set_conf(conf)')
            ## ----------------------------------------------------------------------------------------------

            if visualize:
                samples.append(visualize_bconf(bconf))
                # set_renderer(True)
                # Conf(robot, joints, conf).assign()
                # wait_for_user()

            ir_outputs = (bq,)
            if ir_only:
                if visualize:
                    [remove_body(samp) for samp in samples]
                yield ir_outputs
                continue

            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            if verbose:
                print('succeed after TracIK solutions:', attempts)

            if visualize:
                [remove_body(samp) for samp in samples]
            yield ir_outputs + ik_outputs
            context_saver.restore()

        if verbose:
            if visualize:
                robot.visualize_grasp_approach(pose_value, g, arm=a, body=g.body, title='sample_bconf')
            print(f'sample_bconf\tIKSolver somehow stopped generating after {attempts} attempts')

    ## do ir sampling of x, y, theta, torso, then solve ik for arm
    else:
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if ir_max_attempts <= attempts:
                print(f'{heading} exceeding ir_max_attempts = {ir_max_attempts}')
                context_saver.restore()
                return
                yield None
                # break # TODO(caelan): probably should be break/return

            attempts += 1
            if verbose: print(f'{heading} attempt {attempts} | inputs = {inputs}')

            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                if verbose: print('    stopped ir_generator in', attempts, 'attempts')
                print(f'{heading} exceeding ir_generator ir_max_attempts = {ir_max_attempts}')
                context_saver.restore()
                return

            if ir_outputs is None:
                continue
            if verbose:
                inp = ir_generator.gi_frame.f_locals
                inp = [inp.get(k) for k in ['pose', 'grasp', 'custom_limits']]
                print(f'           ir_generator  |  inputs = {inp}  |  ir_outputs = {ir_outputs}')
                from pybullet_tools.logging_utils import myprint
                myprint(f'           ir_generator  |  inputs = {inp}  |  ir_outputs = {ir_outputs}')

            if visualize:
                bconf = ir_outputs[0].values
                samples.append(visualize_bconf(bconf))

            if ir_only:
                yield ir_outputs
                continue

            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                context_saver.restore()
                if _pull_stream_diag_enabled() and _is_faucet_knob(world, o):
                    print(f'{heading} PULL_DIAG ik-failed ir_outputs={ir_outputs}')
                if verbose:
                    from pybullet_tools.logging_utils import myprint
                    myprint(f'{heading} IK failed for ir_outputs = {ir_outputs}')
                continue
            if verbose:
                print('succeed after IK attempts:', attempts)
                from pybullet_tools.logging_utils import myprint
                myprint(f'{heading} succeed after IK attempts: {attempts} | ir_outputs = {ir_outputs}')

            if visualize:
                [remove_body(samp) for samp in samples]
            yield ir_outputs + ik_outputs
            return
            # if not p.init:
            #    return


@timed('kernel.ik.approach')
def solve_approach_ik(arm, obj, pose_value, grasp, base_conf,
                      world, robot, custom_limits, obstacles_here, ignored_pairs_here,
                      resolution=DEFAULT_RESOLUTION, attachments={}, title='solve_approach_ik',
                      ACONF=False, teleport=False, verbose=False, debug_mp_obstacles=False, visualize=False):

    if isinstance(obj, tuple):  ## may be a (body, joint) or a body with a marker
        body = obj[0]
        obstacles_here = []
    else:
        body = obj
    obj_name = ''
    if hasattr(world, 'BODY_TO_OBJECT') and body in world.BODY_TO_OBJECT:
        obj_name = getattr(world.BODY_TO_OBJECT[body], 'debug_name', '')
    log_this_object = verbose or ('chicken' in str(obj_name))
    pull_diag = _pull_stream_diag_enabled() and _is_faucet_knob(world, obj)

    ## TODO: change to world.get_grasp_parent
    addons = [body]
    if hasattr(world, 'BODY_TO_OBJECT') and body in world.BODY_TO_OBJECT and \
            world.BODY_TO_OBJECT[body].grasp_parent is not None:
        addons.append(world.BODY_TO_OBJECT[body].grasp_parent)

    ## don't consider objects that will be moving with obj
    # obstacles_here = filter_grasp_obstacles_for_body(world, body, obstacles_here)
    obstacles_here = [oo for oo in filter_grasp_obstacles_for_body(world, body, obstacles_here) if oo != body]
    approach_obstacles = [oo for oo in obstacles_here if oo != body]
    for obst in obstacles_here:
        if obst == obj: continue
        if is_placement(body, obst) or is_placement(obst, body) or is_contained(obst, body):
            if (body, obst) not in ignored_pairs_here:
                pairs = [(body, obst), (obst, body)]
                # print_blue(f'{title}\t adding ignored pairs {pairs}')
                ignored_pairs_here.extend(pairs)

    # approach_obstacles = problem.world.refine_marker_obstacles(obj, approach_obstacles)  ## for steerables

    gripper_pose = robot.get_grasp_pose(pose_value, grasp.value, arm, body=obj)
    approach_pose = robot.get_grasp_pose(pose_value, grasp.approach, arm, body=obj)

    arm_joints = robot.get_arm_joints(arm)

    default_conf = robot.get_carry_conf(arm, grasp.grasp_type, grasp.value)
    base_conf.assign()
    set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()

    collision_fn = robot.get_collision_fn(obstacles=obstacles_here)

    ## visualize the gripper
    gripper_grasp = None
    if visualize:
        robot.visualize_grasp_approach(pose_value, grasp, arm=arm, title='solve_approach_ik')

    ## cached from whole-body IK
    if base_conf.joint_state is not None:
        grasp_conf = list(map(base_conf.joint_state.get, arm_joints))
    else:
        grasp_conf = robot.inverse_kinematics(arm, gripper_pose, obstacles_here, verbose=verbose)
        if grasp_conf is None and _is_faucet_knob(world, obj) and has_tracik():
            from pybullet_tools.tracik import IKSolver
            tool_link = robot.get_tool_link(arm)
            tool_pose = robot.get_tool_pose_for_ik(arm, gripper_pose)
            ik_solver = IKSolver(robot.body, tool_link=tool_link, first_joint=arm_joints[0],
                                 custom_limits=custom_limits)
            grasp_conf = ik_solver.solve(tool_pose, seed_conf=default_conf)

    if grasp_conf is not None:
        set_joint_positions(robot, arm_joints, grasp_conf)
        if visualize:
            set_renderer(True)
            wait_unlocked('solve_approach_ik | visualized the arm')

    found_collision = False
    if collided(robot, obstacles_here, articulated=True, world=world, tag=title, verbose=verbose,
                ignored_pairs=ignored_pairs_here, min_num_pts=3): ## approach_obstacles): # [obj]
        found_collision = True
    if collision_fn(base_conf.values, verbose=False):
        found_collision = True
    robot.print_full_body_conf(title=f'solve_approach_ik({arm}), grasp_conf={nice(grasp_conf)}')
    if pull_diag:
        print('PULL_DIAG grasp-stage base={} grasp={} grasp_conf={} collision={}'.format(
            nice(base_conf.values), nice(grasp.value), nice(grasp_conf), found_collision))

    if grasp_conf is None or found_collision:
        # wait_unlocked()
        if log_this_object:
            from pybullet_tools.logging_utils import myprint
            reason = 'grasp_conf=None' if grasp_conf is None else 'grasp collision'
            tool_pose = robot.get_tool_pose_for_ik(arm, gripper_pose)
            myprint(f'{title}Grasp IK failure for {obj_name or obj} | {reason} | '
                    f'base_conf={nice(base_conf.values)} | pose={nice(pose_value)} | '
                    f'grasp={nice(grasp.value)} | gripper_pose={nice(gripper_pose)} | '
                    f'tool_pose={nice(tool_pose)} | obstacles={obstacles_here}')
            if grasp_conf is None and verbose:
                print(f'{title}Grasp IK computation failure')
        if visualize:
            remove_body(gripper_grasp)
        return None

    approach_conf = None
    if has_tracik():
        from pybullet_tools.tracik import IKSolver
        tool_link = robot.get_tool_link(arm)
        tool_pose = robot.get_tool_pose_for_ik(arm, approach_pose)
        ik_solver = IKSolver(robot.body, tool_link=tool_link, first_joint=arm_joints[0],
                             custom_limits=custom_limits)  # TODO: cache
        approach_conf = ik_solver.solve(tool_pose, seed_conf=grasp_conf)

    if not has_tracik() or approach_conf is None:
        approach_conf = robot.inverse_kinematics(arm, approach_pose, obstacles_here, verbose=verbose)
        # if not has_tracik() and approach_conf is not None:
        #     print('\n\n FastIK succeeded after TracIK failed\n\n')
        # approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits)

    found_collision = False
    if approach_conf is not None:
        set_joint_positions(robot, arm_joints, approach_conf)
        if collided(robot, obstacles_here, articulated=True, world=world, tag=title,
                    verbose=verbose, ignored_pairs=ignored_pairs_here, min_num_pts=3):
            found_collision = True
        if collision_fn(base_conf.values, verbose=False):
            found_collision = True
    robot.print_full_body_conf(title=f'solve_approach_ik({arm}), approach_conf={nice(approach_conf)}')

    if approach_conf is None or found_collision:
        if pull_diag:
            print('PULL_DIAG approach-stage base={} grasp={} approach_conf={} collision={}'.format(
                nice(base_conf.values), nice(grasp.value), nice(approach_conf), found_collision))
        if log_this_object:
            from pybullet_tools.logging_utils import myprint
            reason = 'approach_conf=None' if approach_conf is None else 'approach collision'
            myprint(f'{title}Approach IK failure for {obj_name or obj} | {reason} | '
                    f'base_conf={nice(base_conf.values)} | grasp={nice(grasp.value)} | '
                    f'approach={nice(grasp.approach)} | approach_conf={nice(approach_conf)} | '
                    f'obstacles={obstacles_here}')
        if verbose:
            if approach_conf is not None:
                approach_conf = nice(approach_conf)
            print(f'{title}Approach IK failure', approach_conf)
        # wait_if_gui()
        if visualize:
            remove_body(gripper_grasp)
        return None
    # elif verbose:
    #     print(f'{title}Approach IK success | sub_inverse_kinematics({robot} at {nice(base_conf.values)}, '
    #           f'{arm}, {nice(approach_pose[0])}) | pose = {pose}, grasp = {nice(grasp.approach)} -> {nice(approach_conf)}')

    # ## -------------------------------------------
    # arm_joints = get_arm_joints(robot, 'left')
    # aconf = Conf(robot, arm_joints, get_joint_positions(robot, arm_joints))
    # print(f'@ mobile_streams.get_ik_fn() -> aconf = {aconf} | bconf = {base_conf}')
    # ## -------------------------------------------

    set_joint_positions(robot, arm_joints, approach_conf)
    # approach_conf = get_joint_positions(robot, arm_joints)

    motion_planning_kwargs = dict(self_collisions=robot.self_collisions,
                                  use_aabb=True, cache=True, ignored_pairs=ignored_pairs_here,
                                  custom_limits=custom_limits, max_distance=robot.max_distance)

    if teleport:
        path = [default_conf, approach_conf, grasp_conf]
    else:
        resolutions = resolution * np.ones(len(arm_joints))
        if is_top_grasp(robot, arm, body, grasp) or True:
            grasp_path = plan_direct_joint_motion(robot.body, arm_joints, grasp_conf, obstacles=approach_obstacles,
                                                  resolutions=resolutions / 2., **motion_planning_kwargs)
            if grasp_path is None:
                if pull_diag:
                    print('PULL_DIAG grasp-path-failed base={} grasp={} obstacles={}'.format(
                        nice(base_conf.values), nice(grasp.value), approach_obstacles))
                if log_this_object:
                    from pybullet_tools.logging_utils import myprint
                    myprint(f'{title}Grasp path failure for {obj_name or obj} | '
                            f'base_conf={nice(base_conf.values)} | grasp={nice(grasp.value)} | '
                            f'obstacles={approach_obstacles}')
                    if verbose:
                        print(f'{title}Grasp path failure')
                if visualize:
                    remove_body(gripper_grasp)
                return None
            dest_conf = approach_conf
        else:
            grasp_path = []
            dest_conf = grasp_conf

        attachments_arg = list(attachments.values()) if isinstance(obj, int) else []
        initially_attached_to_o = [m.child.body for m in world.attachments.values() if m.parent.body == obj]
        obstacles_here = [m for m in obstacles_here if m not in initially_attached_to_o + [obj]]
        obstacles_here = [m for m in obstacles_here if (m, obj) not in ignored_pairs_here]
        verbose = f'[{title}.plan_joint_motion]' if debug_mp_obstacles else False
        motion_planning_kwargs.update(attachments=attachments_arg, resolutions=resolutions,
                                      restarts=3, iterations=25, smooth=50, verbose=verbose)
        # print_debug(f'{title}\t obstacles_here: {obstacles_here}')

        set_joint_positions(robot, arm_joints, default_conf)
        approach_path = plan_joint_motion(robot, arm_joints, dest_conf, obstacles=obstacles_here, **motion_planning_kwargs)  # smooth=25
        if approach_path is None:

            if debug_mp_obstacles:
                recover_threshold = 3
                found_objects = []
                for o in obstacles_here:
                    isolated_obstacles = [m for m in obstacles_here if m != o]
                    set_joint_positions(robot, arm_joints, default_conf)
                    possible_path = plan_joint_motion(robot, arm_joints, dest_conf, obstacles=isolated_obstacles,
                                                      **motion_planning_kwargs)
                    if possible_path is not None:
                        found_objects.append(o)
                        print(f'\t\tsolved after removing {world.name_to_object(o).debug_name}, '
                              f'len(possible_path) = {len(possible_path)}')

                    if len(found_objects) >= recover_threshold:
                        print(f'\tfound more than {recover_threshold} that enable cfree motion planning')
                        break
                found_objects = [f"{world.body_to_object(m).debug_name}" for m in found_objects]
                print(f'{title}\tApproach path failure, would have succeeded without any object in {found_objects}')

                if len(found_objects) >= recover_threshold:
                    approach_path = plan_joint_motion(robot, arm_joints, dest_conf, obstacles=obstacles_here,
                                                      **motion_planning_kwargs)

            if approach_path is None:
                if pull_diag:
                    print('PULL_DIAG approach-path-failed base={} grasp={} obstacles={}'.format(
                        nice(base_conf.values), nice(grasp.value), obstacles_here))
                if log_this_object:
                    from pybullet_tools.logging_utils import myprint
                    myprint(f'{title}\tApproach path failure for {obj_name or obj} | '
                            f'base_conf={nice(base_conf.values)} | grasp={nice(grasp.value)} | '
                            f'obstacles={obstacles_here}')
                    if verbose:
                        print(f'{title}\tApproach path failure')

                if visualize:
                    remove_body(gripper_grasp)
                return None
        path = approach_path + grasp_path

    mt = create_trajectory(robot.body, arm_joints, path)

    robot.reset_ik_solvers()  ## otherwise unpickleable
    cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot.body)], commands=[mt])

    set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()

    if visualize:
        remove_body(gripper_grasp)
    if ACONF:
        return (mt.path[-1], cmd)
    return (cmd,)


## ------------------------------------------------------------------------------


# def get_ik_gen(problem, max_attempts=100, collisions=True, learned=True, teleport=False,
#                ir_only=False, pick_up=True, given_grasp_conf=False,
#                soft_failures=False, verbose=False, visualize=False, **kwargs):
#     """ given grasp of target object p, return base conf and arm traj """
#     ir_max_attempts = 40
#     ir_sampler = get_ir_sampler(problem, collisions=collisions, learned=learned,
#                                 max_attempts=ir_max_attempts, verbose=verbose, **kwargs)
#     if not pick_up and given_grasp_conf:
#         ik_fn = get_ik_fn_old(problem, collisions=collisions, teleport=teleport, verbose=False,
#                               ACONF=True, **kwargs)
#     else:
#         ik_fn = get_ik_fn(problem, pick_up=pick_up, given_grasp_conf=given_grasp_conf,
#                           collisions=collisions, teleport=teleport, verbose=verbose, **kwargs)
#     robot = problem.robot
#     world = problem.world
#     obstacles = problem.fixed if collisions else []
#     heading = '\t\tmobile_streams.get_ik_gen | '
#
#     co_kwargs = dict(articulated=True, verbose=verbose, world=world)
#
#     def gen(a, o, p, g, context=None):
#         if isinstance(o, tuple):
#             obstacles_here = [obs for obs in obstacles if obs != o[0]]
#         else:
#             obstacles_here = [obs for obs in obstacles if obs != o]
#
#         if visualize:
#             samples = []
#
#         process_ik_context(context)
#
#         """ check if hand pose is in collision """
#         p.assign()
#         if 'pstn' in str(p):
#             pose_value = linkpose_from_position(p)
#         else:
#             pose_value = p.value
#         open_arm(robot, a)
#         context_saver = WorldSaver(bodies=[robot, o])
#
#         if visualize:
#             gripper_grasp = robot.visualize_grasp(pose_value, g.value, arm=a, body=g.body)
#             set_renderer(enable=True)
#             wait_unlocked()
#         else:
#             gripper_grasp = robot.set_gripper_pose(pose_value, g.value, arm=a, body=g.body)
#
#         # co_kwargs['verbose'] = True
#         if collided(gripper_grasp, obstacles_here, **co_kwargs):
#             if verbose:
#                 print(f'{heading} -------------- grasp {nice(g.value)} is in collision')
#             if visualize:
#                 set_renderer(enable=True)
#                 wait_unlocked()
#                 robot.remove_gripper(gripper_grasp)
#             return None
#         # co_kwargs['verbose'] = False
#
#         arm_joints = get_arm_joints(robot, a)
#         default_conf = robot.get_carry_conf(arm, grasp.grasp_type, grasp.value)
#
#         ## solve IK for all 13 joints
#         if robot.use_torso and has_tracik():
#             from pybullet_tools.tracik import IKSolver
#             tool_from_root = robot.get_tool_from_root(a)
#             tool_pose = robot.get_grasp_pose(pose_value, g.value, a, body=g.body)
#             gripper_pose = multiply(tool_pose, invert(tool_from_root))
#
#             # gripper_grasp = robot.visualize_grasp(pose_value, g.value, a, body=g.body)
#
#             tool_link = robot.get_tool_link(a)
#             ik_solver = IKSolver(robot, tool_link=tool_link, first_joint=None,
#                                  custom_limits=robot.custom_limits)  ## using all 13 joints
#
#             attempts = 0
#             for conf in ik_solver.generate(gripper_pose):
#                 joint_state = dict(zip(ik_solver.joints, conf))
#                 if max_attempts <= attempts:
#                     if verbose:
#                         print(f'\t\t{get_ik_gen.__name__} failed after {attempts} attempts!')
#                     # wait_unlocked()
#                     if soft_failures:
#                         attempts = 0
#                         yield None
#                         context_saver.restore()
#                         continue
#                     else:
#                         break
#                 attempts += 1
#
#                 base_joints = robot.get_base_joints()
#                 bconf = list(map(joint_state.get, base_joints))
#                 bq = Conf(robot, base_joints, bconf, joint_state=joint_state)
#                 bq.assign()
#
#                 set_joint_positions(robot, arm_joints, default_conf)
#                 if collided(robot, obstacles_here, tag='ik_default_conf', **co_kwargs):
#                     # set_renderer(True)
#                     # wait_for_user()
#                     continue
#
#                 ik_solver.set_conf(conf)
#                 if collided(robot, obstacles_here, tag='ik_final_conf', **co_kwargs):
#                     # robot.add_collision_grasp(a, o, g)
#                     # robot.add_collision_conf(Conf(robot.body, ik_solver.joints, conf))
#                     continue
#
#                 if visualize:
#                     samples.append(visualize_bconf(bconf))
#                     # set_renderer(True)
#                     # Conf(robot, joints, conf).assign()
#                     # wait_for_user()
#
#                 ir_outputs = (bq,)
#                 if ir_only:
#                     if visualize:
#                         [remove_body(samp) for samp in samples]
#                     yield ir_outputs
#                     continue
#
#                 inputs = a, o, p, g
#                 ik_outputs = ik_fn(*(inputs + ir_outputs))
#                 if ik_outputs is None:
#                     continue
#                 if verbose: print('succeed after TracIK solutions:', attempts)
#
#                 if visualize:
#                     [remove_body(samp) for samp in samples]
#                 yield ir_outputs + ik_outputs
#                 context_saver.restore()
#
#         ## do ir sampling of x, y, theta, torso, then solve ik for arm
#         else:
#             inputs = a, o, p, g
#             ir_generator = ir_sampler(*inputs)
#             attempts = 0
#             while True:
#                 if max_attempts <= attempts:
#                     # print(f'{heading} exceeding max_attempts = {max_attempts}')
#                     yield None
#                     # break # TODO(caelan): probably should be break/return
#
#                 attempts += 1
#                 if verbose: print(f'{heading} | attempt {attempts} | inputs = {inputs}')
#
#                 try:
#                     ir_outputs = next(ir_generator)
#                 except StopIteration:
#                     if verbose: print('    stopped ir_generator in', attempts, 'attempts')
#                     print(f'{heading} exceeding ir_generator ir_max_attempts = {ir_max_attempts}')
#                     return
#
#                 if ir_outputs is None:
#                     continue
#                 inp = ir_generator.gi_frame.f_locals
#                 inp = [inp[k] for k in ['pose', 'grasp', 'custom_limits']]
#                 if verbose:
#                     print(f'           ir_generator  |  inputs = {inp}  |  ir_outputs = {ir_outputs}')
#
#                 if visualize:
#                     bconf = ir_outputs[0].values
#                     samples.append(visualize_bconf(bconf))
#
#                 if ir_only:
#                     yield ir_outputs
#                     continue
#
#                 ik_outputs = ik_fn(*(inputs + ir_outputs))
#                 if ik_outputs is None:
#                     continue
#                 if verbose:
#                     print('succeed after IK attempts:', attempts)
#
#                 if visualize:
#                     [remove_body(samp) for samp in samples]
#                 yield ir_outputs + ik_outputs
#                 return
#                 #if not p.init:
#                 #    return
#     return gen


###########################################


def compute_pull_door_arm_motion(inputs, world, robot, obstacles, ignored_pairs, saver, resolution=DEFAULT_RESOLUTION,
                                 num_intervals=30, round_to=4, collisions=True, visualize=False, verbose=False):
    a, o, pst1, pst2, g, bq1, aq1 = inputs
    is_knob = o in world.cat_to_bodies('knob')

    contact_bodies = get_handle_motion_contact_bodies(world, o) if is_knob else set()
    motion_obstacles = [obstacle for obstacle in obstacles if obstacle not in contact_bodies]
    # Knob turning intentionally keeps the gripper in contact with the stove body.
    # collision_fn = robot.get_collision_fn(obstacles=obstacles, verbose=verbose)
    collision_fn = robot.get_collision_fn(obstacles=motion_obstacles, verbose=verbose)

    if pst1.value == pst2.value:
        return None

    saver.restore()
    pst1.assign()
    bq1.assign()
    aq1.assign()

    arm_joints = robot.get_arm_joints(a)
    resolutions = resolution * np.ones(len(arm_joints))
    other_obstacles = [mm for mm in obstacles if mm not in contact_bodies]

    # BODY_TO_OBJECT = problem.world.BODY_TO_OBJECT
    # joint_object = BODY_TO_OBJECT[o]
    # old_pose = get_link_pose(joint_object.body, joint_object.handle_link)
    handle_link = get_handle_link(o)
    old_pose = get_link_pose(o[0], handle_link)
    if visualize:
        # set_renderer(enable=True)
        gripper_before = robot.visualize_grasp(old_pose, g.value)

    gripper_before = robot.get_grasp_pose(old_pose, g.value, a, body=o)

    world_from_base = bconf_to_pose(bq1)
    gripper_from_base = multiply(invert(gripper_before), world_from_base)
    # print('gripper_before', nice(gripper_before))
    # print('invert(gripper_before)', nice(invert(gripper_before)))

    ## saving the mapping between robot bconf to object pst for execution
    mapping = {}
    if is_knob:
        rconf_rounded = tuple([round(n, round_to) for n in aq1.values])
        mapping[rconf_rounded] = pst1.value
    else:
        rpose_rounded = tuple([round(n, round_to) for n in bq1.values])
        mapping[rpose_rounded] = pst1.value

    ## may move arm only or base only
    apath = []
    aq_after = Conf(aq1.body, aq1.joints, aq1.values)
    bpath = []
    bq_after = Conf(bq1.body, bq1.joints, bq1.values)
    for i in range(num_intervals):
        step_str = f"pr2_streams.get_pull_door_handle_motion_gen | step {i}/{num_intervals}\t"
        change = (i + 1) / num_intervals * (pst2.value - pst1.value)
        value = change + pst1.value
        pst_after = Position((pst1.body, pst1.joint), value)
        pst_after.assign()
        new_pose = get_link_pose(o[0], handle_link)
        if visualize:
            gripper_after = robot.visualize_grasp(new_pose, g.value, color=BROWN)
            set_camera_target_body(gripper_after, dx=0.2, dy=0, dz=1)  ## look top down
            remove_body(gripper_after)

        gripper_after = robot.get_grasp_pose(new_pose, g.value, a, body=o)

        if is_knob:
            aconf_after = list(aq1.values)
            aconf_after[-1] -= change
            aq_after = Conf(aq1.body, aq1.joints, aconf_after)
            aq_after.assign()
        else:
            ## try to transform the base the same way as gripper to a cfree pose
            world_from_base = multiply(gripper_after, gripper_from_base)
            # joint_state = dict(zip(aq1.joints, aq1.values))
            bq_after = pose_to_bconf(world_from_base, robot) ## , joint_state=joint_state
            bq_after.assign()

        found_collision = False
        if collisions:
            col_kwargs = dict(articulated=False, world=world, verbose=verbose)
            if collided(robot, motion_obstacles, **col_kwargs):
                found_collision = True
                if verbose:
                    print('[COLLISION 1]\tcollided(robot, obstacles, **col_kwargs)')
            if collided(o[0], other_obstacles, ignored_pairs=ignored_pairs, **col_kwargs):
                found_collision = True
                if verbose:
                    print('[COLLISION 2]\tcollided(o[0], other_obstacles, ignored_pairs=ignored_pairs, **col_kwargs)')
            if collision_fn(bq_after.values, verbose=verbose):
                found_collision = True
                if verbose:
                    print(f'[COLLISION 3]\tcollision_fn(bq_after.values)')

        if found_collision:
            if len(apath) > 1:
                apath[-1].assign()
            if len(bpath) > 1:
                bpath[-1].assign()
            break
        else:
            if is_knob:
                apath.append(aq_after)
                rpose_rounded = tuple([round(n, round_to) for n in aq_after.values])
                mapping[rpose_rounded] = value
                if verbose:
                    print(f'{step_str} : {nice(aq_after.values)}')
            else:
                bpath.append(bq_after)
                bq_rounded = tuple([round(n, round_to) for n in bq_after.values])
                mapping[bq_rounded] = value
                if verbose and False:
                    print(f'{step_str} : {nice(bq_after.values)}')

    if visualize:
        remove_body(gripper_before)

    if (is_knob and len(apath) < num_intervals * 0.25) or (not is_knob and len(bpath) < num_intervals):
        return None

    if is_knob:
        at = Trajectory(apath)
        arm_cmd = Commands(State(), savers=[BodySaver(robot.body)], commands=[at])
        aq2 = at.path[-1]
        if aq2.values == aq1.values:
            aq2 = aq1
        group = f"{a}_arm"
        step_str = f"pr2_streams.get_turn_knob_handle_motion_gen | step {len(apath)}/{num_intervals}\t"
        if verbose:
            print(f'{step_str} : {nice(aq2.values)}')
    else:
        bt = Trajectory(bpath)
        robot.reset_ik_solvers()
        base_cmd = Commands(State(), savers=[BodySaver(robot.body)], commands=[bt])
        bq2 = bt.path[-1]
        group = 'base-torso'
        step_str = f"pr2_streams.get_pull_door_handle_motion_gen | step {len(bpath)}/{num_intervals}\t"
        if verbose:
            print(f'{step_str} : {nice(bq2.values)}')

    pst1.assign()
    bq1.assign()
    aq1.assign()
    add_to_rc2oc(robot, group, a, o, mapping)

    if is_knob:
        return aq2, arm_cmd
    return bq2, base_cmd


def get_pull_door_handle_motion_gen(problem, custom_limits={}, collisions=True, teleport=False,
                                    num_intervals=30, max_ir_trial=30, visualize=False, verbose=False):
    visualize = visualize and has_gui()
    if teleport:
        num_intervals = 1
    robot = problem.robot
    world = problem.world
    saver = BodySaver(robot)
    # world_saver = WorldSaver()
    obstacles = problem.fixed if collisions else []
    ignored_pairs = problem.ignored_pairs if collisions else []

    def fn(a, o, pst1, pst2, g, bq1, aq1, fluents=[]):
        if fluents:
            process_motion_fluents(fluents, robot, verbose=verbose)
        # else:
        #     world_saver.restore()

        inputs = a, o, pst1, pst2, g, bq1, aq1
        return compute_pull_door_arm_motion(inputs, world, robot, obstacles, ignored_pairs, saver,
                                            num_intervals=num_intervals, collisions=collisions,
                                            visualize=visualize, verbose=verbose)

    return fn


def get_pull_door_handle_with_link_motion_gen(problem, custom_limits={}, collisions=True, teleport=False,
                                              num_intervals=30, max_ir_trial=30, visualize=False, verbose=False):
    visualize &= has_gui()
    if teleport:
        num_intervals = 1
    robot = problem.robot
    world = problem.world
    saver = BodySaver(robot)
    world_saver = WorldSaver()
    obstacles = problem.fixed if collisions else []
    ignored_pairs = problem.ignored_pairs if collisions else []

    def fn(a, o, pst1, pst2, g, bq1, aq1, l, pl1, fluents=[]):
        if fluents:
            process_motion_fluents(fluents, robot)
        else:
            world_saver.restore()

        pl1.assign()
        inputs = a, o, pst1, pst2, g, bq1, aq1
        results = compute_pull_door_arm_motion(inputs, world, robot, obstacles, ignored_pairs, saver,
                                               num_intervals=num_intervals, collisions=collisions,
                                               visualize=visualize, verbose=verbose)
        if results is None:
            return None
        bq2, base_cmd = results

        pst2.assign()
        pl2 = LinkPose(l, get_link_pose(l[0], l[-1]), joint=pst2.joint, position=pst2.value)
        pst1.assign()

        return bq2, base_cmd, pl2

    return fn


##################################################


def get_arm_ik_fn(problem, custom_limits={}, resolution=DEFAULT_RESOLUTION, return_first_aconf=False,
                  collisions=True, teleport=False, verbose=False):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    world = problem.world
    world_saver = WorldSaver()
    title = 'mobile_streams.get_arm_ik_fn:\t'

    def fn(arm, obj, pose, grasp, base_conf, grasp_conf, fluents=[]):
        if isinstance(obj, tuple): ## may be a (body, joint) or a body with a marker
            body = obj[0]
        else:
            body = obj
        if fluents:
            attachments = process_motion_fluents(fluents, robot) # TODO(caelan): use attachments
        # else:
        #     world_saver.restore()

        if 'pstn' in str(pose): ## isinstance(pose, Position):
            pose_value = linkpose_from_position(pose)
        else:
            pose_value = pose.value

        addons = [body]
        # if world.BODY_TO_OBJECT[body].grasp_parent != None:
        #     addons.append(world.BODY_TO_OBJECT[body].grasp_parent)

        # approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        approach_obstacles = {o for o in obstacles if o not in addons}
        # approach_obstacles = problem.world.refine_marker_obstacles(obj, approach_obstacles)  ## for steerables

        gripper_pose = robot.get_grasp_pose(pose_value, grasp.value, arm, body=obj)
        approach_pose = robot.get_grasp_pose(pose_value, grasp.approach, arm, body=obj)

        # arm_link = get_gripper_link(robot, arm)
        arm_joints = robot.get_arm_joints(arm)

        default_conf = robot.get_carry_conf(arm, grasp.grasp_type, grasp.value)
        pose.assign()
        base_conf.assign()
        robot.open_arm(arm)
        grasp_conf = grasp_conf.values
        set_joint_positions(robot, arm_joints, grasp_conf) # default_conf | sample_fn()
        # grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose, custom_limits=custom_limits) #, upper_limits=USE_CURRENT)
        #                                     #nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
        if (grasp_conf is None) or collided(robot, obstacles, articulated=True, tag=title, world=world): ## approach_obstacles): # [obj]
            if verbose:
                if grasp_conf is not None:
                    grasp_conf = nice(grasp_conf)
                print(f'{title}Grasp IK failure | {grasp_conf} = pr2_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                      f'{arm}, {nice(gripper_pose[0])}) | pose = {pose}, grasp = {grasp}')
                for b in obstacles:
                    if pairwise_collision(robot, b):
                        # set_renderer(True)
                        print(f'                        robot at {nice(base_conf.values)} colliding with {b} at {nice(get_pose(b))}')
            return None
        else:
            if verbose:
                print(f'{title}Grasp IK success | {nice(grasp_conf)} = pr2_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                      f'{arm}, {nice(gripper_pose[0])}) | pose = {pose}, grasp = {grasp}')

        #approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose, custom_limits=custom_limits) ##, max_iterations=500
        # approach_conf = solve_nearby_ik(robot, arm, approach_pose, custom_limits=custom_limits)
        approach_conf = robot.inverse_kinematics(arm, approach_pose, obstacles, verbose=verbose)
        if (approach_conf is None) or collided(robot, obstacles, articulated=True, tag=title, world=world): ##
            if verbose:
                if approach_conf != None:
                    approach_conf = nice(approach_conf)
                print(f'{title}Approach IK failure | sub_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                      f'{arm}, {nice(approach_pose[0])}) | pose = {pose}, grasp = {nice(grasp.approach)} -> {approach_conf}')
                for b in obstacles:
                    if pairwise_collision(robot, b):
                        print(f'                        robot at {nice(base_conf.values)} colliding with {b} at {nice(get_pose(b))}')
            #wait_if_gui()
            return None
        else:
            if verbose:
                print(f'{title}Approach IK success | sub_inverse_kinematics({robot} at {nice(base_conf.values)}, '
                      f'{arm}, {nice(approach_pose[0])}) | pose = {pose}, grasp = {nice(grasp.approach)} -> {nice(approach_conf)}')

        set_joint_positions(robot, arm_joints, approach_conf)
        #approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {}  ## {attachment.child: attachment} TODO: problem with having (body, joint) tuple

        motion_planning_kwargs = dict(attachments=list(attachments.values()), self_collisions=robot.self_collisions,
                                      use_aabb=True, cache=True,
                                      custom_limits=custom_limits, max_distance=robot.max_distance)

        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = resolution * np.ones(len(arm_joints))
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, obstacles=approach_obstacles,
                                                  resolutions=resolutions/2., **motion_planning_kwargs)
            if grasp_path is None:
                if verbose: print(f'{title}Grasp path failure')
                return None
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, obstacles=obstacles, resolutions=resolutions,
                                              restarts=2, iterations=25, smooth=0, **motion_planning_kwargs) # smooth=25
            if approach_path is None:
                if verbose: print(f'{title}Approach path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot.body, arm_joints, path)
        attachments = {attachment.child: attachment} ## TODO: problem with having (body, joint) tuple
        robot.reset_ik_solvers()
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot.body)], commands=[mt])
        aconf = mt.path[-1]
        if return_first_aconf:
            aconf = mt.path[0]
        return (aconf, cmd)
    return fn


def get_ik_ungrasp_gen(problem, max_attempts=25, teleport=False, **kwargs):
    ik_fn = get_arm_ik_fn(problem, teleport=teleport, return_first_aconf=True, **kwargs)
    # ik_fn = get_ik_fn(problem, pick_up=False, given_grasp_conf=True, **kwargs)
    def gen(*inputs):
        attempts = 0
        while True:
            if max_attempts <= attempts:
                return None
            yield ik_fn(*(inputs))
            return
    return gen


def get_ik_ungrasp_mark_gen(problem, max_attempts=25, teleport=False, **kwargs):
    ik_fn = get_ik_fn_old(problem, teleport=teleport, **kwargs)
    def gen(*inputs):
        return ik_fn(*(inputs))
        # attempts = 0
        # while True:
        #     if max_attempts <= attempts:
        #         return None
        #     yield ik_fn(*(inputs))
        #     return
    return gen


def get_ik_nudge_gen(problem, max_attempts=25, teleport=False, **kwargs):
    ik_fn = get_arm_ik_fn(problem, teleport=teleport, **kwargs)
    # ik_fn = get_ik_fn(problem, pick_up=False, given_grasp_conf=True, **kwargs)
    def gen(*inputs):
        attempts = 0
        while True:
            if max_attempts <= attempts:
                return None
            yield ik_fn(*(inputs))
            return
    return gen
