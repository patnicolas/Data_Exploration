import unittest
from fractal.fractaldimimage import FractalDimImage


class FractalDimImageTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        image_path_name = '../../../images/fractal_test_image.jpg'
        fractal_dim_image = FractalDimImage(image_path_name)
        self.assertTrue(fractal_dim_image.image is not None)
        if fractal_dim_image.image is not None:
            print(fractal_dim_image.image.shape)

    @unittest.skip('Ignore')
    def test_call(self):
        import numpy as np
        image_path_name = '../../../images/fractal_test_image.jpg'

        fractal_dim_image = FractalDimImage(image_path_name)
        self.assertTrue(fractal_dim_image.image is not None)
        fractal_dim, trace = fractal_dim_image()
        trace_str = '/n'.join([str(box_param) for box_param in trace])
        print(f'Fractal dimension: {float(fractal_dim)}\nTrace {trace_str}')

    @unittest.skip('Ignore')
    def test_timeit(self):
        import timeit
        image_path_name = '../../../images/fractal_test_image.jpg'
        timeit.timeit(FractalDimImage(image_path_name))
        image_path_name = '../../../images/fractal_test_image_large.jpg'
        timeit.timeit(FractalDimImage(image_path_name))


    def test_plots(self):
        import numpy as np
        image_path_name = '../../../images/fractal_test_image.jpg'
        fractal_dim_image = FractalDimImage(image_path_name)
        self.assertTrue(fractal_dim_image.image is not None)
        fractal_dim, trace = fractal_dim_image()
        trace_str = '/n'.join([str(box_param) for box_param in trace])
        print(f'Fractal dimension: {float(fractal_dim)}\nTrace {trace_str}')

        box_params = np.array([[param.eps, param.measurements] for param in trace])
        y = box_params[:, 1]
        x = np.linspace(1, len(y), len(y))
        import matplotlib.pyplot as plt

        # Create a scatter plot
        plt.scatter(x, y)

        # Add title and labels
        plt.title('Trace box measurement distribution - image')
        plt.xlabel('Iterations')
        plt.ylabel('Box measurement units')

        plt.show()

