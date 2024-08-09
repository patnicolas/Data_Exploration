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

    def test__call(self):
        image_path_name = '../../../images/fractal_test_image.jpg'
        fractal_dim_image = FractalDimImage(image_path_name)
        self.assertTrue(fractal_dim_image.image is not None)
        fractal_dim, history = fractal_dim_image()
        print(f'Fractal dimension: {float(fractal_dim)}\nHistory {history}')