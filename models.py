import torch
import torch.nn as nn

class featExtractionNets(nn.Module):
    def __init__(self):
        super(featExtractionNets, self).__init__()
        self.downsample1 = nn.Sequential(
            nn.Conv2d(3,32,3,stride=1,padding=1,bias=True), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()) # 32, 180, 600
        self.downsample2 = nn.Sequential(
            nn.Conv2d(32,64,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64,64,3,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU()) # 64, 90, 300
        self.downsample3 = nn.Sequential(
            nn.Conv2d(64,128,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128,128,3,stride=2,padding=1,bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU()) # 128, 45, 150
        self.downsample4 = nn.Sequential(
            nn.Conv2d(128,256,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256,256,3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU()) # 256, 45, 150
            
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding= 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, padding= 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.upsample2 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ConvTranspose2d(128, 128, kernel_size=3, padding= 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding= 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.upsample3 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ConvTranspose2d(64, 64, kernel_size=3, padding= 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding= 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        self.upsample4 = nn.Sequential(
            nn.Upsample(scale_factor=2,mode='bilinear'),
            nn.ConvTranspose2d(32, 32, kernel_size=3, padding= 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, padding= 1),
            nn.BatchNorm2d(3),
            nn.LeakyReLU())
        
    def forward(self,x):
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        x = self.downsample4(x)
        
        f = x

        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        return x, f
        
class upSamplingNets(nn.Module):
    def __init__(self):
        super(upSamplingNets, self).__init__()

        self.conv1 = nn.Conv2d(256, 256, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(256, 128, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(8)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.sigmoid(self.pixel_shuffle(self.conv4(x)))
        return x

class refineNets(nn.Module):
    def __init__(self):
        super(refineNets, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.conv8 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

    def forward(self, x, fm):
        out = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        out = out + fm
        out = self.conv8(self.conv7(self.conv6(self.conv5(out))))
        out = out + x 
        return out