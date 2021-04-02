import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import os
import warnings
import argparse
from isonet.utils.misc import tprint, pprint
from isonet.utils.config import C
from isonet.models import *
from torchvision import datasets
from torchvision import transforms


def arg_parse():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--cfg', required=True,
                        help='path to config file', type=str)
    parser.add_argument('--output', default='default', type=str)
    parser.add_argument('--gpus', type=str)
    parser.add_argument('--resume', default='', type=str)
    args = parser.parse_args()
    return args

with torch.cuda.device(0):
    args = arg_parse()
    # disable imagenet dataset jpeg warnings
    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
    # ---- setup GPUs ----
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    assert torch.cuda.is_available()
    num_gpus = torch.cuda.device_count()
    # cudnn.benchmark = True
    # ---- setup configs ----
    C.merge_from_file(args.cfg)
    C.SOLVER.TRAIN_BATCH_SIZE *= num_gpus
    C.SOLVER.TEST_BATCH_SIZE *= num_gpus
    C.SOLVER.BASE_LR *= num_gpus
    C.freeze()

    net = ISONet()
    # net = models.densenet161()
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
