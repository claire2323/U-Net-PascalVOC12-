from net import *
import os
from torch.utils.data import Dataset
import os
from utils import *
from torchvision import transforms
from data import *
from torchvision.utils import save_image
net = UNet().cuda()
#加载保存图像
weight = '/home/claire/deeplearning/learningdeeping/U-Net/parama/unet.pth'

if os.path.exists(weight):
  net.load_state_dict(torch.load(weight))
  print('sucessfully')
else:
  print('no loading')
  
  
_input = input('please input image path')

img = keep_image_size_open(_input)
img_data = transform(img).cuda()
img_data = torch.unsqueeze(img_data,dim=0)
out = net(img_data)
save_image(out,'/home/claire/deeplearning/learningdeeping/U-Net/result/result.jpg')
print(out)
