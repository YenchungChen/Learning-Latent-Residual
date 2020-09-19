import gc
import time
import torch
import configargparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans

import utils
import models
from dataset import Kinetics

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--ori_root', required=True, help='Path to directory with uncompressed data.')
p.add_argument('--cpr_root', required=True, help='Path to directory with compressed data.')
p.add_argument('--checkpoint_path', type=str, default='./logs',
               required=False, help='Path to directory where checkpoints & tensorboard events will be saved.')

# testing setting
p.add_argument('--begin_epoch', type=int, default=5, help='Begin epoch during testing.')
p.add_argument('--end_epoch', type=int, default=50, help='End epoch during testing.')

opt = p.parse_args()

testset = Kinetics(ori_dir= opt.ori_root, cpr_dir= opt.cpr_root)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

number = range(opt.begin_epoch, opt.end_epoch+1)

# Create Models
featExNets = models.featExtractionNets()
upSamplingNets = models.upSamplingNets()
refineNets = models.refineNets()

if torch.cuda.is_available():
    featExNets = featExNets.cuda()
    upSamplingNets = upSamplingNets.cuda()
    refineNets = refineNets.cuda()

featExNets.eval()
upSamplingNets.eval()
refineNets.eval()

# Start Evaluation
with torch.no_grad():
    for i in number:
        feat_dict = torch.load(opt.checkpoint_path + '/models/featExNets_%d.pth' %i)
        featExNets.load_state_dict(feat_dict)

        upSampling_dict = torch.load(opt.checkpoint_path + '/models/upSamplingNets_%d.pth' %i)
        upSamplingNets.load_state_dict(upSampling_dict)

        refine_dict = torch.load(opt.checkpoint_path + '/models/refineNets_%d.pth' %i)
        refineNets.load_state_dict(refine_dict)
        
        pre_kmmodel = utils.load_obj(opt.checkpoint_path + '/kmeans/kmmodel_%d' %i)
        pre_centerPatch = utils.load_obj(opt.checkpoint_path + '/kmeans/centerPatch_%d' %i)
        
        ori_psnr, ori_ssim = 0, 0
        avg_err, avg_psnr, avg_ssim = 0, 0, 0
        avg_f_diff = 0
        start_time = time.time()

        for _, data in enumerate(testloader):
            ori_v = torch.autograd.Variable(data['ori'], requires_grad=False).cuda()
            de_v = torch.autograd.Variable(data['de'], requires_grad=False).cuda()
            residual = ori_v - de_v

            ori_psnr += utils.cal_psnr(ori_v.cpu().data.numpy(), de_v.cpu().data.numpy(), data_range= 1.0).item() / len(testset)
            ori_ssim += utils.cal_ssim(ori_v.squeeze().cpu().data.numpy().transpose(1,2,0), de_v.squeeze().cpu().data.numpy().transpose(1,2,0),data_range=1.0,multichannel=True).item() / len(testset)

            _, features = featExNets(residual)

            pick = []
            patchResFeat = features.squeeze().permute(1,2,0).contiguous().view(-1,features.size()[1]).cpu().detach().data.numpy()
            prediction = pre_kmmodel.predict(patchResFeat.astype(np.float64))
            
            for x in prediction:
                pick.append(pre_centerPatch[x])
            pick = torch.from_numpy(np.array(pick)).permute(1,0).contiguous().view(-1,features.size()[2],features.size()[3]).unsqueeze(0)
            
            
            pre_features = torch.autograd.Variable(pick.float(), requires_grad=False).cuda()
            f_diff = torch.abs(features - pre_features).mean()
            avg_f_diff += f_diff.cpu().item() / len(testset)

            upsampled_representation = upSamplingNets(pre_features)
            rec = refineNets(de_v, upsampled_representation)

            err = torch.abs(rec - ori_v).mean()
            avg_err += err.item() / len(testset)
            post_psnr = utils.cal_psnr(ori_v.cpu().data.numpy(), rec.cpu().data.numpy()).item() / len(testset)
            post_ssim = utils.cal_ssim(ori_v.squeeze().cpu().data.numpy().transpose(1,2,0), rec.squeeze().cpu().data.numpy().transpose(1,2,0),data_range=1.0,multichannel=True).item() / len(testset)
            avg_psnr += post_psnr
            avg_ssim += post_ssim

        with open(opt.checkpoint_path + '/eval.txt', "a") as out_file:
            out_file.write('\n=========================')
            out_file.write('\n[%d/%d] L1: %.6f, ori_PSNR: %.6f, PSNR: %.6f, SSIM: %.6f, Time: %d'%(i, opt.end_epoch, avg_err, ori_psnr, avg_psnr, avg_ssim, time.time()-start_time))
            out_file.write('\nfilter difference = %.6f'% (avg_f_diff))