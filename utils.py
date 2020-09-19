import os
import pickle
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def cal_psnr(img_true, img_test, data_range=None):
    return peak_signal_noise_ratio(img_true, img_test, data_range=data_range)

def cal_ssim(im1, im2, data_range=None, multichannel=False):
    return structural_similarity(im1, im2, data_range=data_range, multichannel=multichannel)

def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_obj(obj, pkl_file):
    with open(pkl_file, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(pkl_file):
    with open(pkl_file, 'rb') as f:
        return pickle.load(f)
    
def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 0.5 every 6 epochs"""
    new_lr = init_lr * (0.5 ** (epoch // 6))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr