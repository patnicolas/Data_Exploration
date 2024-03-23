import geomstats.backend as gs


gs.random.seed(42)
print('Seed back-end')


class GeometricException(Exception):
    def __init__(self, *args, **kwargs):  # real signature unknown
        pass
