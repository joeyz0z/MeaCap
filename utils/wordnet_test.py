from nltk.corpus import wordnet
import nltk
import torch
import os
import sys
from PIL import Image
from sentence_transformers import SentenceTransformer


dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from models.clip_utils import CLIP

'''
world net test
'''

# words = nltk.word_tokenize('stuffed bears')
# pos = nltk.pos_tag(words)
# print(pos)

# word1 = "person"
# word1_set = wordnet.synsets(word1, pos=wordnet.NOUN)[0]
# word2 = "man"
# word2_set = wordnet.synsets(word2, pos=wordnet.NOUN)[0]
#
# print(word1_set.lch_similarity(word2_set))
# synonyms = []
#
# for syn in wordnet.synsets(word1):
#     for lm in syn.lemmas():
#         synonyms.append(lm.name())
# print(set(synonyms))

# from models.clip_utils import CLIP

'''
Clip compute
'''
#
# clip_model = CLIP('/media/xieyan/Hard Disk2/pretrain_model/clip_weights').to('cuda')
#
# image = Image.open('../singal_image/COCO_val2014_000000233977.jpg')
#
# clip_score, clip_ref = clip_model.compute_image_text_similarity_via_Image_text(image, text)
#
# print(clip_score)
# print(clip_ref)

text = ['Image of person.',
        'Image of chef.',
        'Image of women.']
#
# text_embedding = clip_model.compute_text_representation(text)
# cos_sim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
# sim = cos_sim(text_embedding[0, :], text_embedding[1, :])
# print(sim)

model = SentenceTransformer('/media/xieyan/Hard Disk2/pretrain_model/all-Mini-L6-v2')
embeddings = model.encode(text, convert_to_tensor=True)

cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
sim = cos_sim(embeddings[0, :], embeddings)
print(sim)




