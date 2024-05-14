import unittest

from manifolds.binarylrmanifold import BinaryLRManifold

class BinaryLRManifoldTest(unittest.TestCase):
    def test_generate(self):
        n_samples = 2000
        n_features = 16
        binary_lr_manifold = BinaryLRManifold(n_features, n_samples)
        train_data = binary_lr_manifold.generate()
        n_train_samples = train_data.n_train_samples()
        self.assertTrue(n_train_samples == int(n_samples*binary_lr_manifold.train_eval_split))
        print(str(train_data))

    def test_train_euclidean(self):
        n_samples = 2000
        n_features = 16
        binary_lr_manifold = BinaryLRManifold(n_features, n_samples)
        train_data = binary_lr_manifold.generate()
        lr_model = binary_lr_manifold.train_euclidean(train_data)
        print(str(lr_model))

    def test_eval_euclidean(self):
        n_samples = 2000
        n_features = 16
        binary_lr_manifold = BinaryLRManifold(n_features, n_samples)
        train_data = binary_lr_manifold.generate()
        lr_model = binary_lr_manifold.train_euclidean(train_data)
        accuracy = binary_lr_manifold.eval_euclidean(train_data, lr_model)
        self.assertTrue(accuracy > 0.7)
        print(f'\nAccuracy: {str(accuracy)}')


if __name__ == '__main__':
    unittest.main()
