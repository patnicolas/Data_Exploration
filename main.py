# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch

def print_hi():
    from datetime import datetime

    m: int = 1000000000000
    s: str = "hello"
    now: datetime = datetime.now()
    x: float = 341.998123
    a: int = 67
    b: int = 33

    print(f'{m}:_ => [{m:_}]')
    print(f'{m}:_ => [{m:,}]')
    print(f'{s}:>10 => [{s:>10}]')
    print(f'{s}:a>10 => [{s:a>10}]')
    print(f'{s}^10 => [{s:^10}]')
    print(f'{now}:%%m:%d:%y => [{now:%m:%d:%y}]')
    print(f'{x}:.3f => [{x:.3f}]')
    print(f'{x}:,.2f => [{x:,.2f}]')
    print(f'{a} + {b} => [{a + b = }]')
    print(f'bool{a} => [{(bool(a)) = }]')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi()
    """
    pt = torch.tensor([2.0, 4.0, 5.0, 6.0])
    print(pt.is_contiguous())   # True
    print(pt.storage())         # 2.0, 4.0, 5.0, 6.0
    pt.contiguous()
    """

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
