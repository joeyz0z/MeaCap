mport torch
import os
import sys
import tqdm
import json
from sentence_transformers import SentenceTransformer

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from models.clip_utils import CLIP

# clip_path = '/media/xieyan/Hard Disk2/pretrain_model/clip_weights'
# clip_model = CLIP(clip_path)
wte_path = '/media/xieyan/Hard Disk2/pretrain_model/all-Mini-L6-v2'
wte_model = SentenceTransformer(wte_path)
wte_model.eval()
wte_model.to('cuda')

memory_txt = '../data/coco/train.txt'
memory_save_path = '../data/memory/coco'
memory_captions_file = '../data/memory/coco/train_captions.json'
memory_embedding_file = '../data/memory/coco/train_embedding.pt'

if not os.path.exists(memory_save_path):
    os.makedirs(memory_save_path)

batch_size = 128

print(f'Loading Support Set from {memory_txt}...')
caption_list = []
with open(memory_txt, 'r', encoding='utf-8') as f:
    captions = f.readlines()
    for caption_ in tqdm.tqdm(captions):
        caption = caption_.rstrip('\n')
        caption_list.append(caption)
    f.close()
caption_num = len(caption_list)

with torch.no_grad():
    for batch_caption_id in tqdm.trange(0, caption_num, batch_size):
        batch_caption = caption_list[batch_caption_id: batch_caption_id + batch_size]
        if batch_caption_id == 0:
            batch_embedding = wte_model.encode(batch_caption, convert_to_tensor=True)
        else:
            batch_embedding_ = wte_model.encode(batch_caption, convert_to_tensor=True)
            batch_embedding = torch.cat((batch_embedding, batch_embedding_), dim=0)

print(f'Saving Memory into {memory_save_path}')
with open(memory_captions_file, 'w', encoding='utf-8') as _json:
    json.dump(caption_list, _json)
    _json.close()
print('Caption List Saved!')
torch.save(batch_embedding, memory_embedding_file)
print('Caption Embedding Saved!')
