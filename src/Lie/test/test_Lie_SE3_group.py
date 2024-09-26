import unittest
import numpy as np
from Lie.Lie_SE3_group import LieSE3Group

class LieSE3GroupTest(unittest.TestCase):

    def test_init(self):
        rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        tran_vector = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]) # np.array([1.0, 0.5, 1.5])
        lie_se3_group = LieSE3Group(rot_matrix, tran_vector)
        print(lie_se3_group)

    def test_build(self):
        rot_matrix = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0]
        tran_vector = [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0] # np.array([1.0, 0.5, 1.5])
        lie_se3_group = LieSE3Group.build(rot_matrix, tran_vector)
        print(lie_se3_group)