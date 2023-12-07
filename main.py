# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pt = torch.tensor([2.0, 4.0, 5.0, 6.0])
    print(pt.is_contiguous())   # True
    print(pt.storage())         # 2.0, 4.0, 5.0, 6.0
    pt.contiguous()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
