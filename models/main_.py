# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader

import time
import os
import sys
from main_args import get_args
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sentence_transformers import SentenceTransformer

from clip_utils import CLIP
import json

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.log import Logger
from src.transformers import BartForTextInfill, BartTokenizer
from language_models.language_model import LanguageModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset.ImgDataset import Imgdata, collate_img
from dataset.ImgDataset_img_return import Imgdata_img_return, collate_img_img_return

from utils.detect_utils import detect_keyword
from utils.generate_utils_ import Get_shuffle_score, filter_text

# use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
torch.manual_seed(0)


if __name__ == "__main__":

    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    using_prompt = args.use_prompt

    log_file = f'Log.log'
    log_path = f'../logs/'
    log_file = os.path.join(log_path, log_file)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = Logger(log_file)
    logger.logger.info(f'The log file is {log_file}')
    logger.logger.info(args)

    # load the pre-trained model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(args.model_path)
    model = BartForTextInfill.from_pretrained(args.model_path)
    model = model.to(device)
    logger.logger.info('Initialize BartForTextInfill from the checkpoint {}.'.format(args.model_path))

    # conzic sample & dataloader
    vl_model = CLIP(args.clip_model)
    vl_model = vl_model.to(device)

    sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    wte_model = SentenceTransformer(args.wte_model_path)
    wte_model.eval()

    # parser model for memory concepts extracting
    parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint)
    parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint)
    parser_model.eval()
    parser_model.to(device)

    # datasets
    img_data = Imgdata_img_return(dir_path=args.img_path, match_model=vl_model)
    train_loader = DataLoader(img_data, batch_size=1, collate_fn=collate_img_img_return, shuffle=False, drop_last=False)

    stop_tokens_tensor = torch.zeros(tokenizer.vocab_size).to(device)
    sub_tokens_tensor = torch.zeros(tokenizer.vocab_size).to(device)
    if 'bart' in tokenizer.__class__.__name__.lower():
        filename = '../data/tokens/bart_stop_tokens.txt'
        index = 0
        with open(filename, 'r') as fr:
            for line in fr:
                words = line.strip().split()
                token_id = int(words[0])
                stop_tokens_tensor[token_id] = 1
                index += 1
        print('Loading {} stop tokens from {} for {}.'.format(index, filename, tokenizer.__class__.__name__))

        # load sub tokens
        filename = '../data/tokens/bart_sub_tokens.txt'
        index = 0
        with open(filename, 'r') as fr:
            for line in fr:
                words = line.strip().split()
                token_id = int(words[0])
                sub_tokens_tensor[token_id] = 1
                index += 1
        print('Loading {} sub tokens from {} for {}.'.format(index, filename, tokenizer.__class__.__name__))

    if args.decoder_chain > 1:
        try:
            rank_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            rank_model = GPT2LMHeadModel.from_pretrained('gpt2')
            logger.logger.info('Initialize GPT2 with default parameters.')
        except:
            raise ValueError('can not load models.')
        rank_lm = LanguageModel(device, rank_model, rank_tokenizer)
    else:
        rank_lm = None
    # rank_lm = None

    if args.conzic_sample:

        result_dict = {}

        # load memory
        memory_clip_embeddings = torch.load(args.memory_clip_embedding_file)
        memory_wte_embeddings = torch.load(args.memory_wte_embedding_file)
        with open(args.memory_caption_file, 'r') as f:
            memory_captions = json.load(f)

        # start generating
        for batch_idx, (batch_image_embeds, batch_name_list, batch_img_list) in enumerate(train_loader):
            start = time.time()
            logger.logger.info(f'{batch_idx + 1}/{len(train_loader)}, image name: {batch_name_list[0]}')

            assert memory_clip_embeddings is not None
            clip_score, clip_ref = vl_model.compute_image_text_similarity_via_embeddings(batch_image_embeds, memory_clip_embeddings)

            select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)  
            select_memory_captions = [memory_captions[id] for id in select_memory_ids]  
            select_memory_wte_embeddings = memory_wte_embeddings[select_memory_ids]
            select_memory_clip_embeddings = memory_clip_embeddings[select_memory_ids]
            masked_sentences = detect_keyword(parser_model=parser_model, parser_tokenizer=parser_tokenizer, wte_model=wte_model,
                                              select_memory_captions=select_memory_captions,
                                              device=device, logger=logger, args=args)

            if args.use_prompt:
                gen_text_list = []
                
                prompt = args.prompt
                prompt_len = len(tokenizer.encode(' ' + prompt)) - 1
                args.prompt_len = prompt_len
                masked_sentences = [prompt] + masked_sentences
                
                gen_text = Get_shuffle_score(batch_image_embeds, masked_sentences, model, vl_model, wte_model, tokenizer,
                                             select_memory_wte_embeddings, stop_tokens_tensor, sub_tokens_tensor,
                                             rank_lm, logger, args, device)
                
                gen_text[0] = gen_text[0].split(prompt)[1].lstrip(' ')
                gen_text[0] = gen_text[0].lower().capitalize()
                best_text = gen_text[0]
            else:
                args.prompt_len = 0
                gen_text = Get_shuffle_score(batch_image_embeds, masked_sentences, model, vl_model, wte_model, tokenizer,
                                             select_memory_wte_embeddings, stop_tokens_tensor, sub_tokens_tensor,
                                             rank_lm, logger, args, device)

                gen_text[0] = gen_text[0].lower().capitalize()
                best_text = gen_text[0]
            logger.logger.info(f'Caption: {best_text}')
