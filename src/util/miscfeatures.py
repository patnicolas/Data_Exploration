__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 2024  All rights reserved."

from typing import TypeVar, AnyStr

class BaseA(object):
    T = TypeVar('T')

    def __init__(self, a: T) -> None:
        self.a = a


class FeaturesEvaluation(object):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def eval_type(n: type[BaseA]):
        return n*n

    @staticmethod
    def walrus_operator() -> None:
        import math
        from pprint import pprint

        n = 0
        while (n := math.sqrt(n)) < 1.615:
            n += 1
        print(n)
        assert n > 1.615

        values = [(x, math.exp(x)) for n in range(1, 24) if (x := math.log(n)) < 3]
        pprint(values)
        assert len(values) < 23


MagicItems = TypeVar('MagicItems')

class MagicItems(object):
    def __init__(self, a: int) -> None:
        self.a = a

    def __add__(self, other: MagicItems) -> MagicItems:
        return MagicItems(int(0.5*(self.a + other.a)))

    def __mul__(self, other: MagicItems) -> MagicItems:
        return MagicItems(int(0.5*(self.a * other.a)))

    def __invert__(self) -> MagicItems:
        return MagicItems(int(100.0/self.a))

    def __str__(self) -> AnyStr:
        return str(self.a)



if __name__ == '__main__':
    FeaturesEvaluation.walrus_operator()

    magic_item1 = MagicItems(4)
    magic_item2 = MagicItems(22)
    added = magic_item1 + magic_item2
    print(f'Added {added}')
    assert added.a == 13
    assert (magic_item1 * magic_item2).a == 44

    print(f'Inverted {~magic_item1}')
    assert (~magic_item1).a == 25


