import unittest

import numpy as np
from manifolds.spddatasetlimits import SPDDatasetLimits
class SPDDdatasetLimitsTest(unittest.TestCase):

    def test_init(self):
        input = np.random.random(90).reshape(3, 5, -1)
        print(f'\n{str(input)}')
        spd_dataset_limits = SPDDatasetLimits(input)
        self.assertTrue( spd_dataset_limits.in_x_min < 0.5)
        self.assertTrue( spd_dataset_limits.in_y_max > 0.5)
        print(str(spd_dataset_limits))

    def test_plot_axis_values(self):
        input = np.random.random(90).reshape(3, 5, -1)
        spd_dataset_limits = SPDDatasetLimits(input)
        axis_x, axis_y, axis_z = spd_dataset_limits.create_axis_values()
        print(f'\n{len(axis_x)} Axis values\n{axis_x}')
        assert(len(axis_x) == int(1.0/SPDDatasetLimits.scale_factor))


if __name__ == '__main__':
    unittest.main()