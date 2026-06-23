import torch.nn as nn
import torch
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
import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
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
    # get current state dict of the generator network
    model_dict = model.state_dict()
    # remove mismatched keys (e.g. output layer) from pretrained dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # update current state dict
    model_dict.update(pretrained_dict)
    # load updated state dict into the generator network
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
    img.save(result_path)



def main():
    
    image_path=config.UI_TEST_IMAGE
    model_dir = config.UI_TEST_MODEL_DIR
    model_path = os.path.join(model_dir, f'netGl1loss_epoch_4_1000.pth') 
    model = load_model(model_path)
    result_dir = config.UI_TEST_RESULT_DIR
    #result_path = os.path.join(result_dir, f'520-{i}test.jpg')
    image_name=image_path.split('/')[-1].split('.')[0]
    result_path = os.path.join(result_dir, f'{image_name}.png')
    
    try:
        test_model(model, image_path, result_path, source_img=True)
    except Exception as e:
        print(e)
    torch.cuda.empty_cache()
    img1=cv2.imread(config.UI_TEST_IMG1)
    img2=cv2.imread(config.UI_TEST_IMG2)
    psnr,_=calculate(img1,img2)
    print(psnr)
    
if __name__ == "__main__":
    main()