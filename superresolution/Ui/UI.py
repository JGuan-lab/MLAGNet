import gradio as gr
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import torch
from Gen import Generator
import pathlib
from metric import calculate
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); import config
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
    #model.load_state_dict(torch.load(model_path, map_location=device))
    return model

def test_model(modelpath, pil_img,source_img=True):
    pre_transform = transforms.Compose([transforms.ToTensor()])
    #pil_img = Image.open(inputimg)
    #img = pre_transform(pil_img).unsqueeze(0).to(device)
    model=load_model(modelpath)
    img = pre_transform(pil_img).unsqueeze(0)
    source = model(img)[0, :, :, :]
    source = source.cpu().detach().numpy()
    source = source.transpose((1, 2, 0))
    source = np.clip(source, 0, 1)

    if source_img:
        #temp = np.clip(img[0, :, :, :].cpu().detach().numpy().transpose((1, 2, 0)), 0, 1)
        #shape = temp.shape
        #plt.imshow(source)
        img = Image.fromarray(np.uint8(source * 255))
        return img


if __name__ =="__main__":
    with gr.Blocks(analytics_enabled=False) as uu_interface:
        gr.Markdown("<div align='center'> <h2> uu-real: </span> </h2> </div>")
        with gr.Tab("single_imgae_resolution"):
            gr.Markdown("<div align='Single Image Super-Resolution'> <h2> uu-real: </span> </h2> </div>")
            with gr.Row():
                with gr.Column(variant='panel'):
                    with gr.TabItem('model'):
                        with gr.Column(variant='panel'):
                            model_info_box = gr.Textbox(value="Please select \"models\" under the checkpoint folder ", interactive=False, visible=True, show_label=False)
                            modelpath=gr.FileExplorer(glob="**/*.pth",value='',file_count='single',root_dir=config.UI_MODEL_DIR + '/',label='load .pth file to precit',interactive=True)
                    with gr.Row():
                        with gr.TabItem('input_img'):
                            with gr.Column(variant='panel)'):
                                # upload your own image
                                img_input=gr.Image(sources=["upload"],label="Upload Image",type='pil')
                                img_button = gr.Button("super_resoulution")
                                # example images; Path is the file path, displayed below
                                example_images=[path.as_posix() for path in sorted(pathlib.Path(config.UI_CTLOW_DIR).rglob('*.png'))]
                                gr.Examples(examples=example_images,inputs=[img_input])                       
                            
                            
                with gr.Column(variant='panel'):
                    out_super_resoulution = gr.Image(type='pil')
                    img_button.click(test_model,inputs=[modelpath,img_input],outputs=[out_super_resoulution],queue=True)
        with gr.Tab("caculate_metrics"):
             gr.Markdown("<div align='Metric Evaluation'> <h2> uu-real: </span> </h2> </div>")
             with gr.Row():
                 with gr.Column(variant='panel'):
                    with gr.TabItem('test_img'):
                        with gr.Column(variant='panel'):
                            # upload your own test image
                            img_input=gr.Image(sources=["upload"],label="Upload Test Image",type='numpy')
                            # example images; Path is the file path, displayed below
                            example_images=[config.UI_EXAMPLE_IMG]
                            gr.Examples(examples=example_images,inputs=[img_input])

                 with gr.Column(variant='panel'):
                    with gr.TabItem('gt_img'):
                        with gr.Column(variant='panel'):
                            # upload your own ground-truth image
                            gt_input=gr.Image(sources=["upload"],label="Upload Reference Image",type='numpy')
                            # example images; Path is the file path, displayed below
                            example_images=[config.UI_CTTRUE_1212]
                            gr.Examples(examples=example_images,inputs=[gt_input]) 
                            
             gt_button = gr.Button("calculate_metrics")         
             with gr.Row():
                with gr.TabItem('metrics'):
                    psnr_box=gr.Number(label="psnr")
                    ssim_box=gr.Number(label="ssim")
                    gt_button.click(calculate,inputs=[img_input,gt_input],outputs=[psnr_box,ssim_box],queue=True)
                    
    uu_interface.launch()