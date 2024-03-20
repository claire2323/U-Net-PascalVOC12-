from PIL import Image

#对图像大小进行统一转换
def keep_image_size_open(path,size=(256,256)):
  img = Image.open(path)
  temp = max(img.size)
  mask = Image.new('RGB',(temp,temp),(0,0,0))
  mask.paste(img,(0,0))
  mask = mask.resize(size)
  return mask
