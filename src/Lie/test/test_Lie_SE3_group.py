import unittest
import numpy as np
from Lie.Lie_SE3_group import LieSE3Group

class LieSE3GroupTest(unittest.TestCase):

    @unittest.skip('Ignored')
    def test_build_from_numpy(self):
        rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        trans_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 2.0]])
        lie_se3_group = LieSE3Group.build_from_numpy(rot_matrix, trans_matrix)
        print(lie_se3_group)
        self.assertTrue(lie_se3_group.se3_element[0, 0] == 1.0)
        self.assertTrue(lie_se3_group.se3_element[1, 2] == -2.0)

    @unittest.skip('Ignored')
    def test_build_from_vec(self):
        rot_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        trans_vector = [1.0, 3.0, 2.0]
        print(f'\nRotation matrix:\n{np.reshape(rot_matrix, (3, 3))}')
        print(f'Translation vector: {trans_vector}')
        lie_se3_group = LieSE3Group.build_from_vec(rot_matrix, trans_vector)
        print(lie_se3_group)
        self.assertTrue(lie_se3_group.se3_element[0, 0] == 1.0)
        self.assertTrue(lie_se3_group.se3_element[2, 1] == 3.0)


    def test_inverse(self):
        rot_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        trans_vector = [0.01, 0.00, 0.02]
        print(f'\nRotation matrix:\n{np.reshape(rot_matrix, (3, 3))}')
        print(f'Translation vector: {trans_vector}')
        lie_se3_group = LieSE3Group.build_from_vec(rot_matrix, trans_vector)
        inv_lie_se3_group = lie_se3_group.inverse()
        print(f'\nSE3 element\n{lie_se3_group}\nInverse\n{inv_lie_se3_group}')
        lie_se3_group.visualize(inv_lie_se3_group.group_element, f'Inverse')


    @unittest.skip('Ignored')
    def test_visualize(self):
        rot_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        trans_vector = [0.5, 0.3, 0.4]
        print(f'\nRotation matrix:\n{np.reshape(rot_matrix, (3, 3))}')
        print(f'Translation vector: {trans_vector}')
        lie_se3_group = LieSE3Group.build_from_vec(rot_matrix, trans_vector)
        print(lie_se3_group)
        lie_se3_group.visualize_all(rot_matrix, trans_vector)




