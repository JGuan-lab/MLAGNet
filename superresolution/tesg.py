import torch.nn as nn
import torch
import re
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import  transforms
import matplotlib.pyplot as plt
from Gen import Generator
import os
import cv2
from tqdm import tqdm
from metric import calculate
import matplotlib.pyplot as plt
import config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# process testImgs data here
class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x

def load_model(model_path):
    model = Generator(3,64,2)
    #model=load_model("/home/Yb/espn/competionmodel/hatnet_epoch_4_500.pth")
    pretrained_dict = torch.load(model_path)
    #pretrained_dict =load_model("/home/Yb/espn/competionmodel/hatnet_epoch_4_500.pth")
    # get the current state dict of the generator
    model_dict = model.state_dict()
    # filter out keys that don't match (e.g., output layer)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # update current state dict with pretrained weights
    model_dict.update(pretrained_dict)
    # load the updated state dict into the generator
    model.load_state_dict(model_dict)
    model.to(device)
    #model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def test_model(model, image_path, result_path, source_img=True):
    pre_transform = transforms.Compose([transforms.ToTensor()])
    pil_img = Image.open(image_path)
    img = pre_transform(pil_img).unsqueeze(0).to(device)

    source = model(img)[0, :, :, :]
    source = source.cpu().detach().numpy()
    source = source.transpose((1, 2, 0))
    source = np.clip(source, 0, 1)

    if source_img:
        temp = np.clip(img[0, :, :, :].cpu().detach().numpy().transpose((1, 2, 0)), 0, 1)
        shape = temp.shape
        #plt.imshow(source)
        img = Image.fromarray(np.uint8(source * 255))
        img.save(result_path)
        print(f"Image saved to: {result_path}")
        return

    plt.imshow(source)
    img = Image.fromarray(np.uint8(source * 255))
    img.save(result_path[:-4] + '_result.jpg')

def test_images_in_folder(model, folder_path, result_folder_path):
    os.makedirs(result_folder_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            result_path = os.path.join(result_folder_path, filename)
            test_model(model, image_path, result_path, source_img=True)
def main():
    image_path = config.SR_TESG_IMAGE
    model_dir = config.SR_TESG_MODEL_DIR
    result_dir = config.SR_TESG_RESULT_DIR



    folder_path = config.SR_TESG_FOLDER  # replace result_dir with the actual path
    gt_path = config.SR_TESG_GT
    img2 = cv2.imread(gt_path)
    best_psnr = 0
    best_file = ''
    file_nums = []
    psnrs = []

    for filename in tqdm(os.listdir(folder_path)):
        img1 = cv2.imread(os.path.join(folder_path, filename))
        psnr, _ = calculate(img1, img2)
        print(psnr)
        file_num = int(filename.split('-')[0])
        file_nums.append(file_num)
        psnrs.append(psnr)
        if psnr > best_psnr:
            best_psnr = psnr
            best_file = filename

    print(f"Best file: {best_file}")
    print(f"Best PSNR: {best_psnr}")
    
    file_nums, psnrs = zip(*sorted(zip(file_nums, psnrs)))
    print(file_nums)
    print(psnrs)
    
    # plot PSNR curve
    plt.plot(file_nums, psnrs, label='PSNR values')
    plt.xlabel('File Number')
    plt.ylabel('PSNR')
    plt.title('PSNR values of different files')

    # annotate peak value
    max_index = psnrs.index(best_psnr)
    plt.annotate(f'Max PSNR: {best_psnr:.2f}\nFile: {file_nums[max_index]}',
                 xy=(file_nums[max_index], best_psnr), 
                 xytext=(file_nums[max_index], best_psnr + 1),
                 arrowprops=dict(facecolor='red', shrink=0.05))

    plt.legend()
    plt.savefig(config.SR_TESG_PSNR_PNG)
    plt.show()

if __name__ == "__main__":
    main()
# def main():
#      for i in range(0,1151,50):
       
#         image_path='/home/Yb/EUR-GAN/ctlow/1189l.png'
#         model_dir = '/home/Yb/uureal/model'
#         model_path = os.path.join(model_dir, f'ganmodelnetG_epoch_{i}.pth') 
#         model = load_model(model_path)
#         result_dir = '/home/Yb/uureal/result/1189/'
#         #result_path = os.path.join(result_dir, f'520-{i}test.jpg')
#         result_path = os.path.join(result_dir, f'{i}-test.jpg')
       
#         try:
#             test_model(model, image_path, result_path, source_img=True)
#         except Exception as e:
#             print(e)
#         torch.cuda.empty_cache()
    
#      folder_path = '/home/Yb/uureal/result/1189'
#      gt_path='/home/Yb/EUR-GAN/cttrue/1189.png'
#      img2=cv2.imread(gt_path)
#      best_psnr=0
#      best_file='a'
#      file_nums=[]
#      psnrs=[]
#      for filename in tqdm(os.listdir(folder_path)):
#         img1=cv2.imread(os.path.join(folder_path,filename))
       
#         psnr,_=calculate(img1,img2)
#         print(psnr)
#         file_num=int(filename.split('-')[0])
#         file_nums.append(file_num)
#         psnrs.append(psnr)
#         if(psnr>best_psnr):
#             best_psnr=psnr
#             best_file=filename
#      print(best_file)
#      print(best_psnr)  
#      file_nums,psnrs=zip(*sorted(zip(file_nums,psnrs)))
#      print(file_nums)
#      print(psnrs)
#      plt.plot(file_nums, psnrs)
#      plt.xlabel('File Number')
#      plt.ylabel('PSNR')
#      plt.title('PSNR values of different files')
#      plt.savefig('/home/Yb/uureal/picr/psnr1189.png')
#      plt.show()
# if __name__ == "__main__":
#     main()