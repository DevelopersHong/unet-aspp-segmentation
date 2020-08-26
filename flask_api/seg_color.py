import logging
import os
import os.path as osp
import io
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import cv2

from unet import NestedUNet
from unet import UNet
from utils.dataset import BasicDataset
from utils.colors import get_colors
from config import UNetConfig

from flask import Response

cfg = UNetConfig()

def inference_one(net, image, device):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(image, cfg.scale))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        if cfg.deepsupervision:
            output = output[-1]

        if cfg.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)        # C x H x W

        tf = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((image.size[1], image.size[0])),
                    transforms.ToTensor()
                ]
        )

        if cfg.n_classes == 1:
            probs = tf(probs.cpu())
            mask = probs.squeeze().cpu().numpy()
            return mask > cfg.out_threshold
        else:
            masks = []
            for prob in probs:
                prob = tf(prob.cpu())
                mask = prob.squeeze().cpu().numpy()
                mask = mask > cfg.out_threshold
                masks.append(mask)
            return masks

def main_seg(imgs,filename):
    cfg = UNetConfig()
    ## 模型以及输入输出数据位置
    model='/home/barfoo/web/prostate_seg/prostate_seg/checkpoints/epoch_30.pth' 

    output='/usr/share/miniProgramImages/prostate_img/' 
    img = imgs

    net = eval(cfg.model)(cfg)
    logging.info("Loading model {}".format(model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))
    logging.info("Model loaded !")
    
    
    mask = inference_one(net=net,
                        image=img,
                            device=device)
    img_name_no_ext = osp.splitext(filename)[0]
    output_img_dir = osp.join(output, img_name_no_ext)
    os.makedirs(output_img_dir, exist_ok=True)
    seg_api_img = osp.join(output_img_dir,'output_seg.png')

    if cfg.n_classes == 1:
        image_idx = Image.fromarray((mask * 255).astype(np.uint8))
        image_idx.save(osp.join(output_img_dir, 'output_seg.png'))
    else:
        colors = get_colors(n_classes=cfg.n_classes)
        w, h = img.size
        img_mask = np.zeros([h, w, 3], np.uint8)
        for idx in range(0, len(mask)):
            image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
            array_img = np.asarray(image_idx)
            img_mask[np.where(array_img==255)] = colors[idx]
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        img_mask = cv2.cvtColor(np.asarray(img_mask),cv2.COLOR_RGB2BGR)
        output_img = cv2.addWeighted(img, 0.7, img_mask, 0.3, 0)
	
        ret = cv2.imwrite(osp.join(output_img_dir, 'output_seg.png'), output_img)
        target = osp.join(output_img_dir, 'output_seg.png') 
        print("test im write ", ret, target)
	
        
    return seg_api_img

