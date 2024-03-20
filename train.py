from torch import nn
import torch
from torch.utils.data import DataLoader
from net import *
from data import *
import os
from torchvision.utils import save_image

#使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
weight_path = '/home/claire/deeplearning/learningdeeping/U-Net/parama/unet.pth'
save_path = '/home/claire/deeplearning/learningdeeping/U-Net/train_image'
data_path = r"/home/claire/deeplearning/VOCtrainval_11-May-2012/VOCdevkit/VOC2012"

if __name__ == '__main__':
  data_loader = DataLoader(MyDataSet(data_path),batch_size=4,shuffle=True)
  net = UNet().to(device)
  #加载权重
  if os.path.exists(weight_path):
    net.load_state_dict(torch.load(weight_path))
    print('successful load weight')
  else:
    print('not successful load weight')
  #损失采用BCE 优化器用Adam
  opt = torch.optim.Adam(net.parameters())
  loss_fun = nn.BCELoss()
  epoch = 1
  while True:
    for i,(image,segment_image) in enumerate(data_loader):
      image,segment_image = image.to(device),segment_image.to(device)
      out_image = net(image)
      train_loss = loss_fun(out_image,segment_image)
      opt.zero_grad()
      train_loss.backward()
      opt.step()
      
      if i%5==0:
        print(f'{epoch}-{i}-train_loss=>>{train_loss.item()}')
      if i%50==0:
        torch.save(net.state_dict(),weight_path)
      _image = image[0]
      _segment_image = segment_image[0]
      _out_image = out_image[0]
      img = torch.stack([_image,_segment_image,_out_image],dim=0)
      #查看当前训练图片
      save_image(img,f'{save_path}/i.png')
    epoch+=1
