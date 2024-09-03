import unittest
from fractal.fractaldimobject import FractalDimObject


class FractalDimObjectTest(unittest.TestCase):

    @unittest.skip('ignore')
    def test_init(self):
        import matplotlib.pyplot as plt

        sample_size = 256
        threshold = 0.85
        fractal_dim_object = FractalDimObject.build(sample_size, threshold, True)
        xyz = fractal_dim_object.xyz
        print(f'len(xyz[1]) = {len(xyz[1])}')

        fig = plt.figure()

        # Add a 3D subplot
        ax = fig.add_subplot(111, projection='3d')

        # Create the scatter plot
        ax.scatter(xyz[0], xyz[1], xyz[2], c='r', marker='o')

        # Add labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Add title and labels
        plt.title('Input data pattern')
        plt.show()

        self.assertTrue(fractal_dim_object.xyz.shape[0] == sample_size)
        self.assertTrue(fractal_dim_object.xyz.shape[1] == sample_size)
        self.assertTrue(fractal_dim_object.xyz.shape[2] == sample_size)
        print(str(fractal_dim_object))

    def test_call(self):
        import math
        sample_size = 256
        threshold = 0.92
        fractal_dim_object = FractalDimObject.build(sample_size, threshold, True)
        coefficient, counts, sizes = fractal_dim_object()
        estimated_num_counts = math.log2(sample_size)
        self.assertTrue(len(counts) == estimated_num_counts - 1)
        print(coefficient)
        self.assertTrue( 2 < math.fabs(coefficient) < 3)

    @unittest.skip('ignore')
    def test_plot(self):
        import matplotlib.pyplot as plt
        import math

        sample_size = 1024
        threshold = 0.88
        fractal_dim_object = FractalDimObject.build(sample_size, threshold)
        _, counts, sizes = fractal_dim_object()
        estimated_num_counts = math.log2(sample_size)
        self.assertTrue(len(counts) == estimated_num_counts-1)

        # Create a scatter plot
        plt.scatter(counts, sizes)

        # Add title and labels
        plt.title('Count vs. sizes counting boxes')
        plt.xlabel('Box counts')
        plt.ylabel('Box sizes')
        plt.legend()
        plt.show()
