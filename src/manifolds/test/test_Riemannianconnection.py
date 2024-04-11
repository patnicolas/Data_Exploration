import unittest
import path
import sys
import os
directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)
sys.path.append(directory.parent)

from manifoldpoint import ManifoldPoint
from functionspace import FunctionSpace
import numpy as np


class TestRiemannianConnection(TestCase):

    def test_init(self):