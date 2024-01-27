import unittest
import path
import sys

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
print(sys.path)
from vectoralgebra3D import Vector3D

class test_vectoralgebra3D(unittest.TestCase):
    def test_vector_3D(self):
        v = Vector3D(4, 7, 1)
        print(str(v))
        print(v('A'))


if __name__ == '__main__':
    unittest.main()
