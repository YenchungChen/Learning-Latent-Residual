import gc
import time
import random
import torch
import configargparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm

import utils
import models
from dataset import Kinetics

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--ori_root', required=True, help='Path to directory with uncompressed data.')
p.add_argument('--cpr_root', required=True, help='Path to directory with compressed data.')
p.add_argument('--logging_root', type=str, default='./logs',
               required=False, help='Path to directory where checkpoints & tensorboard events will be saved.')

# training_settings
p.add_argument('--max_epochs', type=int, default=500, help='Max epochs during training.')
p.add_argument('--batch_size', type=int, default=8, help='Batch size.')
p.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
p.add_argument('--k', type=int, default=1024, help='Number of clusters in Residual Pattern Discovery.')

opt = p.parse_args()

k = opt.k
lr = opt.learning_rate
max_epochs = opt.max_epochs

# Create Dataloader
trainset = Kinetics(ori_dir= opt.ori_root, cpr_dir= opt.cpr_root)
trainloader = DataLoader(trainset, batch_size=opt.batch_size, shuffle=True)

# Create Models
featExNets = models.featExtractionNets()
upSamplingNets = models.upSamplingNets()
refineNets = models.refineNets()

if torch.cuda.is_available():
    featExNets = featExNets.cuda()
    upSamplingNets = upSamplingNets.cuda()
    refineNets = refineNets.cuda()

# Create Optimizer
opt_feature = torch.optim.Adam(featExNets.parameters(),lr=lr)
opt_upSampling = torch.optim.Adam(upSamplingNets.parameters(),lr=lr)
opt_refine = torch.optim.Adam(refineNets.parameters(),lr=lr)

# Create Logging dir
utils.cond_mkdir(opt.logging_root + '/kmeans')
utils.cond_mkdir(opt.logging_root + '/models')
# Save command-line parameters to log directory.
with open(opt.logging_root + '/params.txt', "w") as out_file:
    out_file.write('\n'.join(["%s: %s" % (key, value) for key, value in vars(opt).items()]))

# Start Training
ori_psnr = 0

for epoch in range(max_epochs):
    utils.adjust_learning_rate(opt_feature, epoch, lr)
    utils.adjust_learning_rate(opt_upSampling, epoch, lr)
    utils.adjust_learning_rate(opt_refine, epoch, lr)

    avg_err, avg_psnr = 0, 0
    acc_rec = 0
    acc_f_diff = 0

    start_time = time.time()

    for z, data in enumerate(tqdm(trainloader)):
        ori_v = torch.autograd.Variable(data['ori'], requires_grad=False).cuda()
        de_v = torch.autograd.Variable(data['de'], requires_grad=False).cuda()
        residual = ori_v - de_v
        reconstruction, features = featExNets(residual)
        
        if epoch == 0:
            ori_psnr += utils.cal_psnr(ori_v.cpu().data.numpy(), de_v.cpu().data.numpy(), data_range=1.0).item()
        
        # epoch 0 to 4 we use real residual patterns to train upSamplingNets and refineNets
        # epoch 5 to - we use approximated residual patterns to train upSamplingNets and refineNets

        if epoch >= 5:
            c = 1 # weight for loss
            pick = []
            
            pre_kmmodel = utils.load_obj( opt.logging_root + '/kmeans/kmmodel_%d' % (epoch-1))
            pre_centerPatch = utils.load_obj( opt.logging_root + '/kmeans/centerPatch_%d' % (epoch-1))
            
            # dcp_f = features.squeeze().permute(1,2,0).contiguous().view(-1,features.size()[1]).cpu().detach().data.numpy()
            patchResFeat = features.permute(0, 2, 3, 1).reshape(-1, features.size()[1]).cpu().detach().numpy()
            prediction = pre_kmmodel.predict(patchResFeat.astype(np.float64)) # numpy float32 is a C double, so use np.float64

            for x in prediction:
                pick.append(pre_centerPatch[x])
            # pick = torch.from_numpy(np.array(pick)).permute(1,0).contiguous().view(-1,features.size()[2],features.size()[3]).unsqueeze(0)
            pick = torch.from_numpy(np.array(pick)).reshape(features.size()[0], features.size()[2], features.size()[3], features.size()[1]).permute(0, 3, 1, 2)

            pre_features = torch.autograd.Variable(pick.float(), requires_grad=False).cuda()
            f_diff = torch.abs(features - pre_features).mean()
            acc_f_diff += f_diff.item()
            
            upsampled_representation = upSamplingNets(pre_features)
        else:
            c = 0 
            f_diff = 0
            upsampled_representation = upSamplingNets(features)
            
        # Detail Reconstruction
        rec = refineNets(de_v, upsampled_representation)
        torch.autograd.set_detect_anomaly(True)
        # Reconstruction loss of featExNets 
        reconstruction_loss = torch.abs(reconstruction - residual).mean()
        # Reconstruction loss of entire framework
        err = torch.abs(rec - ori_v).mean()
        feature_loss = 0.5*reconstruction_loss + (0.2+(1-c)*0.3)*err + (c*0.3)*f_diff

        # # Update network
        opt_upSampling.zero_grad()
        opt_refine.zero_grad()
        opt_feature.zero_grad()
        feature_loss.backward()
        opt_upSampling.step()
        opt_feature.step()
        opt_refine.step()

        avg_psnr += utils.cal_psnr(ori_v.cpu().data.numpy(), rec.cpu().data.numpy(), data_range=1.0).item()
        acc_rec += reconstruction_loss.item()
        avg_err += err.item()

        del ori_v, de_v, err, f_diff, reconstruction_loss, reconstruction, features, rec

    
    # Prepare kmeans model and center/representative patch of each cluster for next epoch

    if epoch >= 4:

        # Collect patch-wise residual features from all training data
        with torch.no_grad():
            patchResFeat_bag = []
            for t, data in enumerate(trainloader):
                ori_v = data['ori'].cuda()
                de_v = data['de'].cuda()

                residual = ori_v - de_v

                residualRecon, features = featExNets(residual) # 45 * 150

                patchResFeat = features.permute(0, 2, 3, 1).reshape(-1, features.size()[1]).cpu().numpy()
                samplePatch = random.sample(range(features.size()[0]*features.size()[2]*features.size()[3]), 20*features.size()[0])

                for idx in samplePatch:
                    patchResFeat_bag.append(patchResFeat[idx])

                del ori_v, de_v, residual, patchResFeat
                gc.collect()

            # Perform clustering (discover representative patches for each clusters)

            centerPatch = [np.zeros((features.size()[1],)) for i in range(k)]
            kmmodel = MiniBatchKMeans(n_clusters=k, batch_size=5).fit(patchResFeat_bag)
            labels = list(kmmodel.labels_)

            for idx, patch in enumerate(patchResFeat_bag):
                centerPatch[labels[idx]] += patch / labels.count(labels[idx])

            utils.save_obj(kmmodel, opt.logging_root + '/kmeans/kmmodel_%d' % epoch)
            utils.save_obj(centerPatch, opt.logging_root + '/kmeans/centerPatch_%d' % epoch)

            del kmmodel, centerPatch, patchResFeat_bag, labels
            gc.collect()
    
    # Save to logs
    utils.cond_mkdir(opt.logging_root + '/models')
    torch.save(featExNets.state_dict(), opt.logging_root + '/models/featExNets_%d.pth' % epoch)
    torch.save(upSamplingNets.state_dict(), opt.logging_root + '/models/upSamplingNets_%d.pth' % epoch)
    torch.save(refineNets.state_dict(), opt.logging_root + '/models/refineNets_%d.pth' % epoch)
    
    with open(opt.logging_root + '/results.txt', "a") as out_file:
        out_file.write('\n=========================')
        out_file.write('\n[%d/%d] L1: %5f, ori_PSNR: %5f, PSNR: %5f, Time: %d'%(epoch, max_epochs, avg_err/len(trainset), ori_psnr/len(trainset), avg_psnr/len(trainset), time.time()-start_time))
        out_file.write('\nreconstruction loss = %5f, filter difference = %5f'% (acc_rec/len(trainset), acc_f_diff/len(trainset)))