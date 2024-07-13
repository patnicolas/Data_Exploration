import unittest

from manifolds.spdmatricesdataset import SPDMatricesDataset


class SPDMatricesDatasetTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        n_spd_matrices = 48
        n_channels = 4
        spd_matrices_generator = SPDMatricesDataset(n_spd_matrices, n_channels)
        size_target = len(spd_matrices_generator.target)
        print(size_target)
        self.assertEqual(size_target, n_spd_matrices*2)
        print(spd_matrices_generator.target[0:n_spd_matrices])
        print(spd_matrices_generator.target[n_spd_matrices:])
        self.assertTrue(all(spd_matrices_generator.target[0:n_spd_matrices]) == 0)
        self.assertTrue(all(spd_matrices_generator.target[n_spd_matrices:0]) == 1)

    def test_call(self):
        n_spd_matrices = 48
        n_channels = 4
        spd_matrices_generator = SPDMatricesDataset(n_spd_matrices, n_channels)
        evals_lows_1 = 13
        evals_lows_2 = 11
        class_sep_ratio_1 = 1.0
        class_sep_ratio_2 = 0.5

        datasets = spd_matrices_generator(evals_lows_1, evals_lows_2, class_sep_ratio_1, class_sep_ratio_2)
        features, target = datasets[0]
        self.assertEqual(len(target), n_spd_matrices * 2)
        print(len([y for y in target if y == 1.0]))
        self.assertEqual(len([y for y in target if y == 1.0]), len(target) / 2)

        features, target = datasets[2]
        self.assertEqual(len(target), n_spd_matrices * 4)
        print(len([y for y in target if y == 1.0]))
        self.assertEqual(len([y for y in target if y == 1.0]), len(target) / 2)

if __name__ == '__main__':
    unittest.main()