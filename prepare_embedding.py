# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

import time
import os
import sys
from args import get_args
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sentence_transformers import SentenceTransformer

from models.clip_utils import CLIP
import json
import copy
from tqdm import tqdm
import shutil

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.log import Logger
from src.transformers import BartForTextInfill, BartTokenizer
from language_models.language_model import LanguageModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset.ImgDataset import Imgdata, collate_img
from dataset.ImgDataset_img_return import Imgdata_img_return, collate_img_img_return

from utils.some_utils import set_seed, update_args_logger
from utils.detect_utils import detect_keyword
from utils.generate_utils_ import Get_shuffle_score, filter_text

if __name__ == "__main__":
    args = get_args()

    input_text_corpus_path = args.memory_path
    set_seed(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    vl_model = CLIP(args.clip_model)
    vl_model = vl_model.to(device)

    sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    wte_model = SentenceTransformer(args.wte_model_path)

    clip_embed_list = []
    wte_embed_list = []

    with open(input_text_corpus_path,'r') as json_f:
        textual_data = json.load(json_f)
    batch_size = 128
    for idx in tqdm(range(0,len(textual_data),batch_size)):
        text_list = textual_data[idx:idx+batch_size]
        clip_embeds = vl_model.compute_text_representation(text_list).detach().cpu()
        clip_embed_list.append(clip_embeds)
        wte_embeds = wte_model.encode(text_list, convert_to_tensor=True, normalize_embeddings=True).detach().cpu()
        wte_embed_list.append(wte_embeds)
        # if idx >= 200000:
        #     break
    all_clip_embeds = torch.cat(clip_embed_list)
    all_wte_embeds = torch.cat(wte_embed_list)
    save_path = f"data/memory/{args.memory_id}"
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    shutil.copy(args.memory_path, os.path.join(save_path, "memory_captions.json"))
    torch.save(all_clip_embeds, os.path.join(save_path, "memory_clip_embeddings.pt"))
    torch.save(all_wte_embeds, os.path.join(save_path, "memory_wte_embeddings.pt"))
