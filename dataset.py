import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

def load_img(filepath):
    img = Image.open(filepath)
    if(len(img.split())==4):
        img,_,_,_ = img.split()
    return img

class Kinetics(Dataset):
    def __init__(self, ori_dir, cpr_dir):
        super(Kinetics, self).__init__()
        self.cpr_dir = cpr_dir
        self.ori_dir = ori_dir
        self.img_list = [img for img in os.listdir(self.cpr_dir) if img[-4:]=='.png']
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = self.img_list[idx]
        
        ori_img = load_img(os.path.join(self.ori_dir, img))
        cpr_img = load_img(os.path.join(self.cpr_dir, img))
        
        sample = {'de': cpr_img, 'ori':ori_img}
        
        if self.transform:
            sample['de'] = self.transform(sample['de'])
            sample['ori'] = self.transform(sample['ori'])
        return sample