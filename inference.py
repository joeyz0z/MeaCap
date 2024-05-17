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

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.log import Logger
from src.transformers import BartForTextInfill, BartTokenizer
from language_models.language_model import LanguageModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset.ImgDataset import Imgdata, collate_img
from dataset.ImgDataset_img_return import Imgdata_img_return, collate_img_img_return

from utils.some_utils import set_seed, update_args_logger, PROMPT_ENSEMBLING
from utils.detect_utils import retrieve_concepts
from utils.generate_utils_ import Get_shuffle_score, filter_text


if __name__ == "__main__":
    args = get_args()  
    set_seed(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")

    ## update logger
    memory_id = args.memory_id
    test_datasets = args.img_path.split("/")[-1]
    lm_training_datasets = args.lm_model_path.split("/")[-1]
    save_file = f'MeaCap_{test_datasets}_memory_{memory_id}_lmTrainingCorpus_{lm_training_datasets}_{args.alpha}_{args.beta}_{args.gamma}_k{args.conzic_top_k}.json'
    log_file = f'MeaCap_{test_datasets}_memory_{memory_id}_lmTrainingCorpus_{lm_training_datasets}_{args.alpha}_{args.beta}_{args.gamma}_k{args.conzic_top_k}.log'
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    log_path = os.path.join(args.output_path, log_file)
    logger = Logger(log_path)
    logger.logger.info(f'The log file is {log_path}')
    logger.logger.info(args)

    ## load the pre-trained model and tokenizer
    tokenizer = BartTokenizer.from_pretrained(args.lm_model_path)
    lm_model = BartForTextInfill.from_pretrained(args.lm_model_path)
    lm_model = lm_model.to(device)
    logger.logger.info('Load BartForTextInfill from the checkpoint {}.'.format(args.lm_model_path))

    vl_model = CLIP(args.vl_model)
    vl_model = vl_model.to(device)
    logger.logger.info('Load CLIP from the checkpoint {}.'.format(args.vl_model))

    sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    wte_model = SentenceTransformer(args.wte_model_path)
    logger.logger.info('Load sentenceBERT from the checkpoint {}.'.format(args.wte_model_path))

    # parser model for memory concepts extracting
    parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint)
    parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint)
    parser_model.eval()
    parser_model.to(device)
    logger.logger.info('Load Textual Scene Graph parser from the checkpoint {}.'.format(args.parser_checkpoint))

    # datasets
    if args.use_memory:
        img_data = Imgdata_img_return(dir_path=args.img_path, match_model=vl_model)
        train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img_img_return, shuffle=False, drop_last=False)
    else:
        img_data = Imgdata(dir_path=args.img_path, match_model=vl_model)
        train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img, shuffle=False, drop_last=False)

    stop_tokens_tensor = torch.zeros(tokenizer.vocab_size).to(device)
    sub_tokens_tensor = torch.zeros(tokenizer.vocab_size).to(device)
    if 'bart' in tokenizer.__class__.__name__.lower():
        filename = 'data/tokens/bart_stop_tokens.txt'
        index = 0
        with open(filename, 'r') as fr:
            for line in fr:
                words = line.strip().split()
                token_id = int(words[0])
                stop_tokens_tensor[token_id] = 1
                index += 1
        print('Loading {} stop tokens from {} for {}.'.format(index, filename, tokenizer.__class__.__name__))

        # load sub tokens
        filename = 'data/tokens/bart_sub_tokens.txt'
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

    ## Load textual memory bank
    if args.use_memory:
        memory_caption_path = os.path.join(f"data/memory/{memory_id}","memory_captions.json")
        memory_clip_embedding_file = os.path.join(f"data/memory/{memory_id}","memory_clip_embeddings.pt")
        memory_wte_embedding_file = os.path.join(f"data/memory/{memory_id}","memory_wte_embeddings.pt")
        memory_clip_embeddings = torch.load(memory_clip_embedding_file)
        memory_wte_embeddings = torch.load(memory_wte_embedding_file)
        with open(memory_caption_path, 'r') as f:
            memory_captions = json.load(f)

    # huge memeory bank cannot load on GPU
    if memory_id == 'cc3m' or memory_id == 'ss1m':
        retrieve_on_CPU = True
        print('CC3M/SS1M Memory is too big to compute on RTX 3090, Moving to CPU...')
        vl_model_retrieve = copy.deepcopy(vl_model).to(cpu_device)
        memory_clip_embeddings = memory_clip_embeddings.to(cpu_device)
    else:
        vl_model_retrieve = vl_model
        retrieve_on_CPU = False

    result_dict = {}
    for batch_idx, (batch_image_embeds, batch_name_list, batch_img_list) in enumerate(train_loader):
        start = time.time()
        logger.logger.info(f'{batch_idx + 1}/{len(train_loader)}, image name: {batch_name_list[0]}')

        if args.use_memory:
            if retrieve_on_CPU != True:
                clip_score, clip_ref = vl_model_retrieve.compute_image_text_similarity_via_embeddings(batch_image_embeds, memory_clip_embeddings)
            else:
                batch_image_embeds_cpu = batch_image_embeds.to(cpu_device)
                clip_score_cpu, clip_ref_cpu = vl_model_retrieve.compute_image_text_similarity_via_embeddings(batch_image_embeds_cpu,
                                                                                                     memory_clip_embeddings)
                clip_score = clip_score_cpu.to(device)
                clip_ref = clip_ref_cpu.to(device)
            select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
            select_memory_captions = [memory_captions[id] for id in select_memory_ids]
            select_memory_wte_embeddings = memory_wte_embeddings[select_memory_ids]
            masked_sentences = retrieve_concepts(parser_model=parser_model, parser_tokenizer=parser_tokenizer, wte_model=wte_model,
                                              select_memory_captions=select_memory_captions,image_embeds=batch_image_embeds,
                                              device=device, logger=logger)
        else:
            # use fixed concepts
            masked_sentences = ["man"]

        all_gen_texts = []
        if args.use_prompt:
            if args.prompt_ensembling == True:
                prompts = PROMPT_ENSEMBLING
            else:
                prompts = args.prompt
            for prompt in prompts:
                # prompt = args.prompt
                prompt_len = len(tokenizer.encode(' ' + prompt)) - 1
                args.prompt_len = prompt_len
                input_sentences = [prompt] + masked_sentences
                gen_text = Get_shuffle_score(batch_image_embeds, input_sentences, lm_model, vl_model, wte_model, tokenizer,
                                             select_memory_wte_embeddings, stop_tokens_tensor, sub_tokens_tensor,
                                             rank_lm, logger, args, device)
                if '.' in gen_text[0]:
                    gen_text[0] = gen_text[0].split('.')[0] + '.'
                else:
                    gen_text[0] = gen_text[0] + '.'
                gen_text[0] = gen_text[0].lstrip(' ')
                gen_text[0] = gen_text[0].lower().capitalize()
                all_gen_texts.append(gen_text[0])
            clip_score, clip_ref = vl_model.compute_image_text_similarity_via_raw_text(batch_image_embeds, all_gen_texts)
            best_text = all_gen_texts[torch.argmax(clip_score,dim=-1)]
        else:
            args.prompt_len = 0
            input_sentences = masked_sentences
            gen_text = Get_shuffle_score(batch_image_embeds, input_sentences, lm_model, vl_model, wte_model, tokenizer,
                                         select_memory_wte_embeddings, stop_tokens_tensor, sub_tokens_tensor,
                                         rank_lm, logger, args, device)
            if '.' in gen_text[0]:
                gen_text[0] = gen_text[0].split('.')[0] + '.'
            else:
                gen_text[0] = gen_text[0] + '.'
            gen_text[0] = gen_text[0].lstrip(' ')
            gen_text[0] = gen_text[0].lower().capitalize()
            best_text = gen_text[0]
        logger.logger.info(f'Best shuffle results: {best_text}')
        # if isinstance(gen_text, list):
        #     gen_text = gen_text[0]

        result_dict[os.path.splitext(batch_name_list[0])[0]] = best_text
        used_time = time.time() - start
        logger.logger.info(f'using {used_time}s')
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    save_file = os.path.join(args.output_path, save_file)
    with open(save_file, 'w', encoding="utf-8") as _json:
        json.dump(result_dict, _json,indent=2)



