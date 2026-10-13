import unittest

from leap_tools.object_reducers import reduce_facts_given_objects


class TestObjectReducers(unittest.TestCase):

    def test_reduce_facts_keeps_tuple_goal_joint_facts(self):
        target_joint = (6, 4)
        other_joint = (6, 5)
        facts = [
            ['joint', target_joint],
            ['unattachedjoint', target_joint],
            ['isjointto', target_joint, 6],
            ['isclosedposition', target_joint, 'closed'],
            ['position', target_joint, 'closed'],
            ['atposition', target_joint, 'closed'],
            ['joint', other_joint],
            ['position', other_joint, 'closed'],
            ['=', ('pickcost',), 1],
        ]

        reduced = reduce_facts_given_objects(
            facts,
            objects=[6, 'left'],
            goals=[['openedjoint', target_joint]],
        )

        self.assertIn(['joint', target_joint], reduced)
        self.assertIn(['unattachedjoint', target_joint], reduced)
        self.assertIn(['isjointto', target_joint, 6], reduced)
        self.assertIn(['isclosedposition', target_joint, 'closed'], reduced)
        self.assertIn(['position', target_joint, 'closed'], reduced)
        self.assertIn(['atposition', target_joint, 'closed'], reduced)
        self.assertNotIn(['joint', other_joint], reduced)

    def test_reduce_facts_drops_irrelevant_dynamic_pose_facts(self):
        target_joint = (6, 4)
        pot = 5
        lid = 4
        facts = [
            ['joint', target_joint],
            ['position', target_joint, 'closed'],
            ['atposition', target_joint, 'closed'],
            ['graspable', pot],
            ['pose', pot, 'pot-pose'],
            ['atpose', pot, 'pot-pose'],
            ['graspable', lid],
            ['pose', lid, 'lid-pose'],
            ['atpose', lid, 'lid-pose'],
        ]

        reduced = reduce_facts_given_objects(
            facts,
            objects=[6, 'left'],
            goals=[['openedjoint', target_joint]],
        )

        self.assertIn(['position', target_joint, 'closed'], reduced)
        self.assertIn(['atposition', target_joint, 'closed'], reduced)
        self.assertNotIn(['pose', pot, 'pot-pose'], reduced)
        self.assertNotIn(['atpose', pot, 'pot-pose'], reduced)
        self.assertNotIn(['pose', lid, 'lid-pose'], reduced)
        self.assertNotIn(['atpose', lid, 'lid-pose'], reduced)


if __name__ == '__main__':
    unittest.main()
