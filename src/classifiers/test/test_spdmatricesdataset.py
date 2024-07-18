import unittest

from classifiers.spdmatricesdataset import SPDMatricesDataset
from classifiers.spdmatricesconfig import SPDMatricesConfig


class SPDMatricesDatasetTest(unittest.TestCase):

    @unittest.skip('Ignore')
    def test_init(self):
        spd_matrices_config = SPDMatricesDatasetTest.__instance()
        spd_matrices_generator = SPDMatricesDataset(spd_matrices_config)
        size_target = len(spd_matrices_generator.target)
        print(size_target)
        self.assertEqual(size_target, spd_matrices_config.n_spd_matrices*2)
        print(spd_matrices_generator.target[0:spd_matrices_config.n_spd_matrices])
        print(spd_matrices_generator.target[spd_matrices_config.n_spd_matrices:])
        self.assertTrue(all(spd_matrices_generator.target[0:spd_matrices_config.n_spd_matrices]) == 0)
        self.assertTrue(all(spd_matrices_generator.target[spd_matrices_config.n_spd_matrices:0]) == 1)

    @unittest.skip('Ignore')
    def test_create(self):
        spd_matrices_config = SPDMatricesDatasetTest.__instance()

        spd_matrices_dataset = SPDMatricesDataset(spd_matrices_config)
        datasets = spd_matrices_dataset.create()
        features, target = datasets[0]
        self.assertEqual(len(target), spd_matrices_config.n_spd_matrices * 2)
        print(len([y for y in target if y == 1.0]))
        self.assertEqual(len([y for y in target if y == 1.0]), len(target) / 2)

        features, target = datasets[2]
        self.assertEqual(len(target), spd_matrices_config.n_spd_matrices * 4)
        print(len([y for y in target if y == 1.0]))
        self.assertEqual(len([y for y in target if y == 1.0]), len(target) / 2)


    def test_plot_datasets(self):
        import matplotlib.pyplot as plt

        spd_matrices_config = SPDMatricesDatasetTest.__instance()
        spd_matrices_dataset = SPDMatricesDataset(spd_matrices_config)
        datasets = spd_matrices_dataset.create()
        features, target = datasets[2]
        sp_training_data = SPDMatricesDataset.train_test_data(features, target)
        SPDMatricesDataset.plot(sp_training_data, features)
        plt.show()



    @unittest.skip('Ignore')
    def test_scatter_3d(self):
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.array([13.03935402, 15.24222798, 10.23830705, 15.24938927, 16.20484899, 13.65910484, 12.84819379, 11.4862075,  11.18008915,
         15.11120485, 12.74482328, 16.15514782, 15.93203208, 11.82921933, 12.14658923, 14.14045029, 11.57721228, 13.32029981,
         11.78302606, 13.22087892, 12.17659684])
        y = 1.2*x
        z = 0.3*x

        # Creating figure
        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection="3d")

        # Add x, y gridlines
        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.3,
                alpha=0.2)

        # Creating color map
        my_cmap = plt.get_cmap('hsv')

        # Creating plot
        sctt = ax.scatter3D(x, y, z,
                            alpha=0.8,
                            c=(x + y + z),
                            cmap=my_cmap,
                            marker='^')

        plt.title("simple 3D scatter plot")
        ax.set_xlabel('X-axis', fontweight='bold')
        ax.set_ylabel('Y-axis', fontweight='bold')
        ax.set_zlabel('Z-axis', fontweight='bold')
        fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)

        # show plot
        plt.show()

    @unittest.skip('Ignore')
    def test_scatter_3d_2(self):
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.array(
            [13.03935402, 15.24222798, 10.23830705, 15.24938927, 16.20484899, 13.65910484, 12.84819379, 11.4862075,
             11.18008915,
             15.11120485, 12.74482328, 16.15514782, 15.93203208, 11.82921933, 12.14658923, 14.14045029, 11.57721228,
             13.32029981,
             11.78302606, 13.22087892, 12.17659684])
        y = 1.2 * x
        z = 0.3 * x

        fig = plt.figure(figsize=(16, 9))
        ax = plt.axes(projection="3d")
        my_cmap = plt.get_cmap('hsv')

        ax.grid(b=True, color='grey',
                linestyle='-.', linewidth=0.3)
        sc = ax.scatter3D(
            x,
            y,
            z,
            c=(x + y + z),
            cmap=my_cmap,
            # alpha=0.2,
        )
        """
        plt.title("simple 3D scatter plot")
        ax.set_xlabel('X-axis', fontweight='bold')
        ax.set_ylabel('Y-axis', fontweight='bold')
        ax.set_zlabel('Z-axis', fontweight='bold')
        """
        fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)


        # ax.set_xticklabels(())
        # ax.set_yticklabels(())
        # ax.set_zticklabels(())

        #spd_dataset_limits = SPDDatasetLimits(features)
        # spd_dataset_limits.set_limits(ax)
        plt.show()

    @staticmethod
    def __instance() -> SPDMatricesConfig:
        n_spd_matrices = 48
        n_channels = 4
        evals_lows_1 = 13
        evals_lows_2 = 11
        class_sep_ratio_1 = 1.0
        class_sep_ratio_2 = 0.5

        return SPDMatricesConfig(
            n_spd_matrices,
            n_channels,
            evals_lows_1,
            evals_lows_2,
            class_sep_ratio_1,
            class_sep_ratio_2
        )


if __name__ == '__main__':
    unittest.main()