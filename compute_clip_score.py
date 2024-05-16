import os, sys
from PIL import Image
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from models.clip_utils import CLIP

clip = CLIP(r'F:\HuggingFace\clip-vit-base-patch32')
clip.eval()
clip.to('cuda')
image = Image.open('/singal_image/image_example\COCO_val2014_000000545385.jpg')
text = ['A large dessert eaten in the 2016 New Hampshire State Fair Hotel Association Hall.',
        'A butter pie served at the famous Mary Teresa restaurant.',
        'A plate topped with cake and fork.',
        'a piece of cake on a white plate with a fork.',
        'A small pie and a fork are on a plate.']

# image = Image.open('F:\ImageText\cbart\VCCap-0.21\image_example\spiderman4.jpg')
# text = ['Spider-Man in the Animated Series.',
#         'A very attractive spiderman typical marvel definition.',
#         'A red and white locomotive is being docked. ',
#         'A person that is on the ground and is holding his device.',
#         'Comics that depict a superhero Spiderman.']
with torch.no_grad():
    clip_score, clip_ref = clip.compute_image_text_similarity_via_Image_text(image, text)

print(f'clip score: {clip_score}\nclip ref: {clip_ref*2.5}')

