__author__ = "Patrick Nicolas"
__copyright__ = "Copyright 2020, 202  All rights reserved."

import torch
import os
import pathlib

if torch.cuda.is_available():
    torch_device = 'cuda:0'
    torch_FloatTensor = torch.cuda
else:
    torch_device = 'cpu'
    torch_FloatTensor = torch.FloatTensor


# Set up folders from either unit-test1 or production
# It is assumed that test1 folder is a sub-directory of the given Python package
relative_path = pathlib.PurePath(os.getcwd()).name
print(f'Relative path: {relative_path}')

# If the directory is a test1
if relative_path == 'test':
    vocab_path = '../../conf/codes/terms.txt'
    models_path = '../../models'
    input_path = '../../input_tensor'
    images_folder = '../../images'
    config_path = '../../conf/config.json'
# or it is the root directory
elif relative_path == 'python':
    vocab_path = 'conf/codes/terms.txt'
    models_path = 'models'
    input_path = 'input'
    images_folder = 'images'
    config_path = 'conf/config.json'
# or any directory between the root and test1
else:
    vocab_path = '../../../conf/codes/terms.txt'
    models_path = '../../../models'
    input_path = '../../../input_tensor'
    images_folder = '../../../images'
    config_path = '../../../conf/config.json'


# Deep learning related labels
model_parameters_label = "model_parameters"
optimizer_parameters_label = "optimizer_parameters"
train_eval_ratio = 0.90
optim_sgd_label = 'sgd'
optim_adam_label = 'adam'
optim_rms_label = 'rms'
optim_nesterov_label = 'nesterov'
conv_net_label = 'convnet'
de_conv_net_label = 'deconvnet'
dff_net_label = 'dffnet'
params_label = 'params'

from config import Config

try:
    config = Config(config_path)
except FileNotFoundError as e:
    if config_path[0] == '.':
        config = Config('conf/config.json')
    else:
        config = Config('../conf/config.json')


assert config is not None, 'Configuration undefined'

s3_config = config('s3_config')
assert s3_config is not None, 'S3 Configuration undefined'

s3_sources = config('s3_sources')

logger = None


def get_logger():
    from util.logger import Logger
    if logger is None:
        return Logger()
    else:
        return logger


def log_info(msg: str):
    get_logger().log(msg, 'INFO')

def log_warn(msg: str):
    get_logger().log(msg, 'WARN')

def log_error(msg: str):
    get_logger().log(msg, 'ERROR')

def log_debug(msg: str):
    get_logger().log(msg, 'DEBUG')

def log_profile(msg: str):
    get_logger().log(msg, 'PROFILE')


# ------------  Debugging statements for size of PyTorch tensor -----------
def log_size(x: torch.Tensor, msg: str):
    get_logger().size(x, msg)


def is_log_debug() -> bool:
    return config['log_level'] == 'DEBUG'

