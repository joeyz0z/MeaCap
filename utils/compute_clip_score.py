import os, sys
from PIL import Image
import torch

# 库外模块导入
# python 3.8以上不可以使用sys.path.append来添加搜索路径
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from models.clip_utils import CLIP

clip = CLIP('/media/xieyan/Hard Disk2/pretrain_model/clip-vit-base-patch32')
clip.eval()
clip.to('cuda')
image = Image.open('')
text = ['']
with torch.no_grad():
    clip_score, clip_ref = clip.compute_image_text_similarity_via_Image_text(image, text)

print(f'clip score: {float(clip_score)}\nclip ref: {float(clip_ref)*2.5}')

