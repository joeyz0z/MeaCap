# -*- coding: utf-8 -*-
# @Time    : 2020/7/27 5:45 PM
# @Author  : He Xingwei
import torch
from torch.utils.data import DataLoader

import time
from typing import Iterable, Optional, Tuple
import os
import sys
from main_args import get_args
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sentence_transformers import SentenceTransformer

# conzic sample use
from clip_utils import CLIP

# detect use
from ultralytics import YOLO
import json

# 库外模块导入
# python 3.8以上不可以使用sys.path.append来添加搜索路径
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
# sys.path.append('../')
from utils.log import Logger
from src.transformers import BartForTextInfill, BartTokenizer
from language_models.language_model import LanguageModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset.ImgDataset import Imgdata, collate_img
from dataset.ImgDataset_img_return import Imgdata_img_return, collate_img_img_return

from utils.detect_utils_new import detect_keyword
from utils.generate_utils_ import Get_shuffle_score, filter_text

torch.manual_seed(0)


if __name__ == "__main__":

    args = get_args()  # args in 'models/main_args.py'
    # set the gpu(s) for code
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    # save
    memory_datasets = args.memory_caption_file.split('/')[-2]
    test_datasets = args.img_path.split('/')[-1]
    model_datasets = 'one_billion'
    using_prompt = args.use_prompt
    # save_prefix = f'VCcap0.2_memory_datasets_{memory_datasets}_model_{model_datasets}_random_prompt_{using_prompt}_test_datasets_{test_datasets}_{args.alpha}_{args.beta}_{args.gamma}.json'
    save_prefix = 'demo.json'
    log_file = f'VCcap0.2_memory_datasets_{memory_datasets}_model_{model_datasets}_random_prompt_{using_prompt}_test_datasets{test_datasets}_{args.alpha}_{args.beta}_{args.gamma}.log'
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

    # conzic 采样的模型、dataloader
    if args.conzic_sample:
        vl_model = CLIP(args.clip_model)
        vl_model = vl_model.to(device)
        vl_model_sample = CLIP(args.shuffle_sample_model)
        vl_model_sample = vl_model_sample.to(device)

        sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        wte_model = SentenceTransformer(args.wte_model_path)

        # datasets
        if args.use_memory:
            img_data = Imgdata_img_return(dir_path=args.img_path, match_model=vl_model)
            train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img_img_return, shuffle=False, drop_last=False)
        else:
            img_data = Imgdata(dir_path=args.img_path, match_model=vl_model)
            train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img, shuffle=False, drop_last=False)

        # parser model for memory concepts extracting
        if args.use_memory:
            parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint)
            parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint)
            parser_model.eval()
            parser_model.to(device)
        else:
            parser_tokenizer = None
            parser_model = None
    else:
        vl_model = None
        vl_model_sample = None
        parser_tokenizer = None
        parser_model = None
        wte_model = None
        train_loader = None

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

    if args.conzic_sample:
        '''detect model use'''
        # if args.use_detect:
        #     detect_model = YOLO(args.detect_path)
        #     logger.logger.info(f'Using detection {detect_model.__class__.__name__}')
        # else:
        #     detect_model = None

        result_dict = {}

        if args.use_memory:
            memory_clip_embeddings = torch.load(args.memory_clip_embedding_file)
            memory_wte_embeddings = torch.load(args.memory_wte_embedding_file)
            with open(args.memory_caption_file, 'r') as f:
                memory_captions = json.load(f)
        else:
            memory_clip_embeddings = None
            memory_wte_embeddings = None
            memory_captions = None

        # cc3m 只能将memory放在CPU上
        # if memory_datasets == 'cc3m':
        if 'cc3m' in memory_datasets or 'ss1m' in memory_datasets:
            print('Memory is too big to compute on RTX 3090, Moving to CPU...')
            vl_model_cpu = CLIP(args.clip_model)
            memory_clip_embeddings_cpu = memory_clip_embeddings.to(cpu_device)
        else:
            memory_clip_embeddings_cpu = None
            vl_model_cpu = None

        for batch_idx, (batch_image_embeds, batch_name_list, batch_img_list) in enumerate(train_loader):
            start = time.time()
            logger.logger.info(f'{batch_idx + 1}/{len(train_loader)}, image name: {batch_name_list[0]}')

            # 准备需要caption的关键词
            if args.use_detect:
                if args.use_memory:
                    assert memory_clip_embeddings is not None
                    # 获取memory embedding和captions
                    # if memory_datasets != 'cc3m':
                    if 'cc3m' not in memory_datasets and 'ss1m' not in memory_datasets:
                        clip_score, clip_ref = vl_model.compute_image_text_similarity_via_embeddings(batch_image_embeds, memory_clip_embeddings)
                    else:
                        batch_image_embeds_cpu = batch_image_embeds.to(cpu_device)
                        # t0 = time.time()
                        clip_score_cpu, clip_ref_cpu = vl_model_cpu.compute_image_text_similarity_via_embeddings(batch_image_embeds_cpu,
                                                                                                             memory_clip_embeddings_cpu)
                        clip_score = clip_score_cpu.to(device)
                        clip_ref = clip_ref_cpu.to(device)
                    select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)  # 选出相似度最高的五句话
                    select_memory_captions = [memory_captions[id] for id in select_memory_ids]  # 相似度最高的五句话
                    # print(f'CPU retrieve using {time.time()-t0}')
                    select_memory_wte_embeddings = memory_wte_embeddings[select_memory_ids]
                    select_memory_clip_embeddings = memory_clip_embeddings[select_memory_ids]
                    # masked_sentences = detect_keyword(parser_model=parser_model, parser_tokenizer=parser_tokenizer, wte_model=wte_model,
                    #                                   select_memory_captions=select_memory_captions,
                    #                                   device=device, logger=logger, args=args)
                    '''usage'''
                    masked_sentences = detect_keyword(parser_model=parser_model, parser_tokenizer=parser_tokenizer, wte_model=wte_model,
                                                      select_memory_captions=select_memory_captions, vl_model=vl_model,
                                                      image_embeds=batch_image_embeds,
                                                      device=device, logger=logger)
                    # masked_sentences = ['Spider-Man']
                else:
                    print('Not be implemented!')
                    break
            else:
                # generate sentences with lexical constraints
                input_file = f'../data/{args.dataset}/{args.num_keywords}keywords.txt'
                print(f'Generate sentences with lexical constraints for {input_file}.')
                masked_sentences = []
                with open(input_file) as fr:
                    for i, line in enumerate(fr):
                        if i % 3 == 0:
                            continue
                        else:
                            line = line.strip().split('\t')[1]
                            if i % 3 == 1:
                                masked_sentences.append(line)  # lexical constraints
            if args.use_prompt:
                gen_text_list = []
                for prompt in args.prompt:
                    prompt_len = len(tokenizer.encode(' ' + prompt)) - 1
                    args.prompt_len = prompt_len
                    masked_sentences = [prompt] + masked_sentences
                    gen_text = Get_shuffle_score(batch_image_embeds, masked_sentences, model, vl_model, wte_model, tokenizer,
                                                 select_memory_wte_embeddings, stop_tokens_tensor, sub_tokens_tensor,
                                                 rank_lm, logger, args, device)

                    gen_text[0] = gen_text[0].split(prompt)[1].lstrip(' ')
                    if '.' in gen_text[0]:
                        gen_text[0] = gen_text[0].split('.')[0] + '.'
                        gen_text[0] = gen_text[0].lower().capitalize()
                    gen_text_list.append(gen_text[0])
                    masked_sentences = masked_sentences[1:]

                shuffle_clip_score, clip_ref = vl_model_sample.compute_image_text_similarity_via_Image_text(batch_img_list[0], gen_text_list)

                # 将memroy的五句话的embedding平均一下，作为聚类中心
                gen_text_wte_embedding = wte_model.encode(gen_text_list, convert_to_tensor=True)
                select_memory_wte_embeddings_avg = torch.mean(select_memory_wte_embeddings, dim=0).unsqueeze(0)
                memory_ref = torch.cosine_similarity(select_memory_wte_embeddings_avg, gen_text_wte_embedding, dim=1)
                memory_score = torch.softmax(memory_ref, dim=0).unsqueeze(0)

                final_score = args.shuffle_alpha * memory_score + args.shuffle_beta * shuffle_clip_score
                best_text_id = torch.argmax(final_score, dim=-1)
                best_text = gen_text_list[best_text_id]
            else:
                args.prompt_len = 0
                gen_text = Get_shuffle_score(batch_image_embeds, masked_sentences, model, vl_model, wte_model, tokenizer,
                                             select_memory_wte_embeddings, stop_tokens_tensor, sub_tokens_tensor,
                                             rank_lm, logger, args, device)
                if '.' in gen_text[0]:
                    gen_text[0] = gen_text[0].split('.')[0] + '.'
                else:
                    gen_text[0] = gen_text[0] + '.'
                gen_text[0] = gen_text[0].lstrip(' ')
                gen_text[0] = gen_text[0].lower().capitalize()
                best_text = gen_text[0]

            # result_text, prompt_id = filter_text(gen_text_list, batch_image_embeds, vl_model)
            logger.logger.info(f'Best shuffle results: {best_text}')

            result_dict[os.path.splitext(batch_name_list[0])[0]] = best_text
            used_time = time.time() - start
            logger.logger.info(f'using {used_time}s')
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_file = os.path.join(args.save_path, save_prefix)
        # save_file = '../singal_image/demo.json'
        logger.logger.info(f'Saving results to {save_file}')
        with open(save_file, 'w', encoding="utf-8") as _json:
            json.dump(result_dict, _json)



