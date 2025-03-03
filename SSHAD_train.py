
from __future__ import print_function
import matplotlib.pyplot as plt
from timm.layers import drop_path
from pytorch_msssim import ssim
import os
import numpy as np
import time
import scipy.io as sio
import h5py
from net.memnet import MemNet3
import torch
import random
import torch.optim
from PIL import Image
from scipy.io import loadmat
from sklearn.decomposition import PCA
from torch.optim.lr_scheduler import ReduceLROnPlateau
import calculate_AUC_show
from thop import profile
from fvcore.nn import FlopCountAnalysis
import cmocean
from anomaly_detector import run_rx
from focal_frequency_loss import FocalFrequencyLoss as FFL
from utils.inpainting_utils import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_hsi(img_np):
    max_val = np.max(img_np, axis=(0, 1))
    min_val = np.min(img_np, axis=(0, 1))
    normalize_hsi = (img_np - min_val) / (max_val - min_val)
    return normalize_hsi


def main():
    """read data"""
    seed = 100  
    set_seed(seed)

    torch.cuda.empty_cache()
    root_path = "/data/lst/SSHAD/data/abu-urban-2.mat"
    mat = loadmat(root_path)
    img_h5 = mat["data"]
    data = np.array(img_h5)
    groundtruth = np.array(mat["map"])



    """data process"""
    img_norm = normalize_hsi(data)  
    img_false = img_norm 
    img_size = img_norm.shape
    band = img_size[2]
    row = img_size[0]
    col = img_size[1]
    print("band,row,col:",band,row,col)


    img_tensor = torch.from_numpy(img_false).type(dtype)  

    img_tensor_exp = img_tensor[None, :].cuda()
    img_tensor_exp = img_tensor_exp.permute(3, 0, 1, 2)
    mask_var = torch.ones(img_tensor.size(2), 1, row, col).cuda()  
    residual_varr = torch.ones(row, col).cuda()

    """parameter setting"""


    OPT_OVER = 'net'
    method = '2D'
    input_depth = img_tensor.shape[2]
    LR = 1e-4

    num_iter = 201 #1001
    lamba1 = 0.0
    lamba2 = 0.1

    thres = 0.000015
    param_noise = False
    reg_noise_std = 0.1 
    set_seed(seed)
    net = MemNet3(img_tensor.size(2), 16, 3, 3,drop_path=0.3).type(dtype) 
    net_input = get_noise(input_depth, method, img_tensor.shape[:-1]).type(dtype)
    mse = torch.nn.MSELoss().type(dtype)


    def closure(iter_num, mask_varr, residual_varr):

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        
        print("net_input:",net_input.size()) # ([1, 205, 100, 100])
        # net_input = net_input.permute(0, 3, 1, 2)


        out = net(net_input)  
        out_np = out.detach().cpu().squeeze().numpy()


        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()

        if iter_num % 50==0 and iter_num!=0:
            img_var_clone = img_tensor_exp.detach().clone()
            net_output_clone = out.detach().clone()
            temp = (net_output_clone[0, :] - img_var_clone[0, :]) * (net_output_clone[0, :] - img_var_clone[0, :])
            residual_img = temp.sum(0)

            residual_var_clone = residual_img
            r_max = residual_img.max()
            # residuals to weights
            residual_img = r_max - residual_img
            r_min, r_max = residual_img.min(), residual_img.max()
            residual_img = (residual_img - r_min) / (r_max - r_min)

            mask_size = mask_var_clone.size()
            for i in range(mask_size[1]):
                mask_var_clone[0, i, :] = residual_img[:]

        out = out.permute(1,0,2,3)

        ffl = FFL(loss_weight=1.0, alpha=1.0)
        loss_ffl = ffl(out* mask_var_clone, img_tensor_exp * mask_var_clone)

        mse_loss = mse(out* mask_var_clone, img_tensor_exp * mask_var_clone)
        ssim_loss_value = 1 - ssim(out * mask_var_clone, img_tensor_exp * mask_var_clone)


        total_loss = (1-lamba1-lamba2) * mse_loss + lamba1 * ssim_loss_value + lamba2 * loss_ffl 


        total_loss.backward()

        print("iteration: %d; MSE loss: %f, FFL loss: %f,Total loss: %f"
              % (iter_num+1, mse_loss, loss_ffl, total_loss))

        return mask_var_clone, residual_var_clone, out, total_loss

    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()
    loss_np = np.zeros((1, 50), dtype=np.float32)
    loss_last = 0
    end_iter = False
    p = get_params(OPT_OVER, net, net_input)
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(p, lr=LR)

    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, background_img, loss = closure(j, mask_var, residual_varr)
        optimizer.step()

        if j >= 1:
            index = j-int(j/50)*50
            loss_np[0][index-1] = abs(loss-loss_last)
            if j % 50 == 0:
                mean_loss = np.mean(loss_np)
                if mean_loss < thres:
                    end_iter = True

        loss_last = loss

        if j == num_iter-1 or end_iter == True:
            residual_np = residual_varr.detach().cpu().squeeze().numpy()

            AUC0 = calculate_AUC_show.calculate_AUC(data, groundtruth, residual_np)
            # AUC_RX = run_rx(residual_np)

            # abu
            total_pixels = row*band

            from sklearn.metrics import roc_curve, auc
            fpr, tpr, thresholds = roc_curve(np.squeeze(np.resize(groundtruth, [total_pixels, 1])),
                                             np.squeeze(np.resize(residual_np, [total_pixels, 1])))
            roc_auc = auc(fpr, tpr)

            
            return




if __name__ == "__main__":
    start = time.process_time()
    main()
    end = time.process_time()
    print("runtime：", end-start)





