from __future__ import print_function

import accelerate
import matplotlib.pyplot as plt
# matplotlib inline
import os
import os.path
import numpy as np
import time
import scipy.io
import h5py
from pathlib import Path
import copy
import logging
import math
import torch
import torch.optim
import torch.nn as nn
from PIL import Image
import yaml
from collections import defaultdict
from accelerate import Accelerator
from accelerate.logging import get_logger
from rich.logging import RichHandler
from rich.progress import Progress
from omegaconf import DictConfig
from typing import Optional, Tuple
from torch.utils.data import DataLoader
import torch.distributions as D
import torch.optim as optim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from anomaly_detector import run_rx
from utils_bs import normalize, false_alarm_rate, channel_align


def calculate_AUC(data,mask, detection_map):
    DataTest = np.array(data) #.transpose(2, 1, 0)
    mask = np.array(mask) #.transpose(1, 0)
    H, W, Dim = DataTest.shape
    num = H * W

    # rx_result_bs = run_rx(difference.cpu().detach().numpy())
    # auc, fpr = false_alarm_rate(mask.reshape([-1]), rx_result_bs.reshape([-1]))
    # print('RX_detector | with Bayesian | ' + file_name + ' | AUC: {}, FPR: {}'.format(auc, fpr))

    R0 = detection_map
    mask_reshape = np.reshape(mask, (1, num))
    anomaly_map = mask_reshape > 0
    normal_map = mask_reshape == 0
    # rows, cols, bands = data.shape
    R0value = np.reshape(R0, (1, num))
    r_max = np.max(R0)
    taus = np.linspace(0, r_max, 5000)

    PF0 = np.zeros_like(taus)
    PD0 = np.zeros_like(taus)

    for index2, tau in enumerate(taus):
        anomaly_map_rx = R0value > tau
        PF0[index2] = np.sum(anomaly_map_rx & normal_map) / np.sum(normal_map)
        PD0[index2] = np.sum(anomaly_map_rx & anomaly_map) / np.sum(anomaly_map)

    f_show = R0
    f_show = (f_show - np.min(f_show)) / (np.max(f_show) - np.min(f_show))
    plt.figure('test')
    plt.title('test')
    plt.imshow(f_show)
    plt.show()

    # plt.figure('R01')
    # plt.imshow(R01)
    # plt.show()

    AUC0 = np.sum((PF0[:-1] - PF0[1:]) * (PD0[1:] + PD0[:-1]) / 2)
    print("AUC0:", AUC0)
    return AUC0