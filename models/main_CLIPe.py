# -*- coding: utf-8 -*-
# @Time    : 2020/7/27 5:45 PM
# @Author  : He Xingwei
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
from typing import Iterable, Optional, Tuple
import os
import sys
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
# conzic sample use
from clip_utils import CLIP
from PIL import Image

# detect use
import math
from ultralytics import YOLO
from collections import Counter
from textblob import TextBlob
import json
import nltk

# python 3.8以上不可以使用sys.path.append来添加搜索路径
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
# sys.path.append('../')
from utils.parse_tool import parse, get_graph_phrases
from utils.log import Logger
from src.transformers import BartForTextInfill, BartTokenizer
from torch.nn.utils.rnn import pad_sequence
from bart import BARTDataset
from language_models.language_model import LanguageModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from dataset.ImgDataset import Imgdata, collate_img
from dataset.ImgDataset_img_return import Imgdata_img_return, collate_img_img_return

# use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA):
torch.manual_seed(0)

# encoder_labels : 0 for copy, 1 for replacement, 2 for insertion
# indicate_labels: 0 for copy, 1 for copy and insertion, 2 for copy, replacement and insertion, 3 for replacement


def generate_function(
        model,
        tokenizer,
        encoder_inputs,
        indicate_labels,
        encoder_loss_type,
        max_insert_label,
        device,
        decoder_inputs=None,
        stop_tokens_tensor=None,
        sub_tokens_tensor=None,
        num_beams=1,
        temperature=1,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1,
        refinement_steps=3,
        max_refinement_steps=10,
        adaptive=False,
        show_refine=1,
        threshold=0,
        decoder_chain=1,
        rank_lm=None,
        max_len=40,
        args=None,
        logger=None
    ):

    batch_size = len(indicate_labels)
    if do_sample:
        effective_batch_size = batch_size * decoder_chain
        if decoder_chain>1:
            # expand inputs
            encoder_inputs = [e.clone() for e in encoder_inputs for i in range(decoder_chain)]
            if decoder_inputs is not None:
                decoder_inputs = [e.clone() for e in decoder_inputs for i in range(decoder_chain)]
            indicate_labels = [e[:] for e in indicate_labels for i in range(decoder_chain)]
    else:
        effective_batch_size = batch_size

    batch_refinement_steps = torch.tensor([0] * effective_batch_size)

    if adaptive:
        current_refinement = 0
        done = False
        while not done:
            predict_outputs, indicate_labels, batch_refinement, decoder_lengths = generate_step_parallel(
                 model, tokenizer, encoder_inputs, indicate_labels, encoder_loss_type, max_insert_label, device,
                 decoder_inputs=decoder_inputs,
                 stop_tokens_tensor=stop_tokens_tensor,
                 sub_tokens_tensor = sub_tokens_tensor,
                 repetition_penalty=repetition_penalty,
                 num_beams=num_beams,
                 temperature=temperature,
                 do_sample=do_sample,
                 top_k=top_k,
                 top_p=top_p,
                 threshold=threshold,
                 max_len=max_len,
                 args=args
                 )
            encoder_inputs = predict_outputs
            print(predict_outputs)
            current_refinement +=1
            batch_refinement_steps += batch_refinement
            if torch.sum(batch_refinement) == 0 or current_refinement == max_refinement_steps:
                done = True
            decoder_inputs = None
    else:
        for i in range(refinement_steps):
            predict_outputs, indicate_labels, batch_refinement, decoder_lengths = generate_step_parallel(  # TODO: xieyan
                 model, tokenizer, encoder_inputs, indicate_labels, encoder_loss_type, max_insert_label, device,
                 decoder_inputs=decoder_inputs,
                 stop_tokens_tensor=stop_tokens_tensor,
                 sub_tokens_tensor=sub_tokens_tensor,
                 repetition_penalty=repetition_penalty,
                 num_beams=num_beams,
                 temperature=temperature,
                 do_sample=do_sample,
                 top_k=top_k,
                 top_p=top_p,
                 threshold=threshold,
                 max_len=max_len,
                 args=args
                 )
            # print(tokenizer.batch_decode(predict_outputs, skip_special_tokens=True))
            encoder_inputs = predict_outputs
            batch_refinement_steps += batch_refinement
            if torch.sum(batch_refinement) == 0:
                break
            else:
                if show_refine:
                    logger.logger.info(f"refinement {i+1}:")
                    for b in range(effective_batch_size):
                        logger.logger.info(tokenizer.decode(predict_outputs[b].tolist(), clean_up_tokenization_spaces=False))
                        # logger.logger.info(tokenizer.convert_ids_to_tokens(predict_outputs[b].tolist()))
            decoder_inputs = None
    predict_outputs = [predict_outputs[i][:length] for i, length in enumerate(decoder_lengths)]
    if do_sample and decoder_chain>1:
        _predict_outputs = []
        _batch_refinement_steps = []
        # use the rank_lm to select the best one from multi decoder chains
        log_ppls, probs = rank_lm.perplexity(input_ids = predict_outputs)
        log_ppls = log_ppls.view([batch_size, -1])
        indices = torch.argmax(-log_ppls,  dim=-1,keepdim=False)
        for b in range(batch_size):
            effective_index = b*decoder_chain + indices[b]
            _predict_outputs.append(predict_outputs[effective_index])
            _batch_refinement_steps.append(batch_refinement_steps[effective_index])

        batch_refinement_steps = _batch_refinement_steps
        predict_outputs = _predict_outputs

    return predict_outputs, batch_refinement_steps

def generate_step(
        model,
        tokenizer,
        encoder_inputs,
        indicate_labels,
        encoder_loss_type,
        max_insert_label,
        device,
        decoder_inputs=None,
        stop_tokens_tensor=None,
        sub_tokens_tensor=None,
        temperature=1,
        repetition_penalty=1,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        num_beams=1,
        threshold=0,
        max_len = None
    ):
    """

    :param model:
    :param encoder_inputs: list of one dimensional tensor
    :param indicate_labels: list of list of int, this tensor is used to denote which tokens are original,
    which tokens are generated. 0 for original tokens, 1 for boundary tokens, 2 for generated tokens.
    0 corresponds to encoder_labels [0], 1 corresponds to encoder_labels [0,2,3,4,5],
    2 corresponds to encoder_labels [0,1,2,3,4,5].
    :param encoder_loss_type: 0 for classification, 1 for regression
    :return:
    """
    start = time.time()
    mask_token_id = tokenizer.mask_token_id
    bos_token_id  = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        pre_predict_outputs = encoder_inputs
        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value= pad_token_id)
        attention_mask = torch.zeros(encoder_inputs.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(encoder_inputs != pad_token_id, 1)
        attention_mask = attention_mask.to(device)
        encoder_inputs = encoder_inputs.to(device)
        # s = time.time()
        # step 1: feed encoder_inputs into the encoder and get encoder_logits
        encoder_outputs, encoder_logits = model.get_encoder_logits(encoder_inputs, attention_mask=attention_mask)
        # e = time.time()
        bts = encoder_inputs.shape[0]
        # s = time.time()
        if decoder_inputs is None:
            # step 2: predict encoder_labels for input_ids based on encoder_logits
            indicate_labels, predict_labels_list = get_encoder_labels(encoder_logits, encoder_loss_type,indicate_labels,
                                                                 max_insert_label,threshold=threshold,max_len=max_len)

            # step 3: compute decoder_inputs based on encoder_inputs and encoder_labels
            decoder_inputs = [BARTDataset.create_decoder_inputs(encoder_inputs[i].tolist(),
                                                                predict_labels_list[i].tolist(), mask_token_id) for i in range(bts)]
        # e = time.time()
        # print(f'decoder inputs : {e-s}')
        # replace the eos_token_id with pad_token_id
        for i, _ in enumerate(decoder_inputs):
            decoder_inputs[i][-1] = pad_token_id

        decoder_lengths = [decoder_inputs[i].shape[0] for i in range(bts)]
        # create decoder_inputs by shifting the decoder_labels right,
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value = pad_token_id)

        decoder_labels = decoder_inputs.clone()
        decoder_inputs[:, 1:] = decoder_labels[:, :-1]
        decoder_inputs[:, 0] =  eos_token_id
        decoder_inputs = decoder_inputs.to(device)

        # step 4: feed decoder_inputs into the decoder and get decoder_logits in a non-auto-regressive way.
        # feed the encoder_outputs to avoid computing it again.
        encoder_logits, decoder_logits = model(input_ids=None, decoder_input_ids=decoder_inputs,
                                               attention_mask=attention_mask,encoder_outputs=encoder_outputs,
                                               use_cache=False)[:2]

        if num_beams>1:
            pass
        else:
            predict_outputs = _generate_no_beam_search(decoder_logits,decoder_inputs, bos_token_id, eos_token_id,
                                                       mask_token_id,indicate_labels,decoder_lengths,
                                                       stop_tokens_tensor=stop_tokens_tensor,
                                                       sub_tokens_tensor=sub_tokens_tensor ,
                                                       temperature=temperature,do_sample=do_sample,
                                                       top_k=top_k,top_p=top_p,repetition_penalty=repetition_penalty)
        refinement_steps = []
        for i in range(bts):
            if predict_outputs[i].shape[0]==pre_predict_outputs[i].shape[0] and all(predict_outputs[i]==pre_predict_outputs[i]):
                refinement_steps.append(0)
            else:
                refinement_steps.append(1)
        refinement_steps = torch.tensor(refinement_steps)
        # the predict_outputs is regarded as new encoder_inputs
    return predict_outputs, indicate_labels, refinement_steps, decoder_lengths


def generate_step_parallel(
        model,
        tokenizer,
        encoder_inputs,
        indicate_labels,
        encoder_loss_type,
        max_insert_label,
        device,
        decoder_inputs=None,
        stop_tokens_tensor=None,
        sub_tokens_tensor=None,
        temperature=1,
        repetition_penalty=1,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        num_beams=1,
        threshold=0,
        max_len = None,
        args=None
    ):
    """

    :param model:
    :param encoder_inputs: list of one dimensional tensor
    :param indicate_labels: list of list of int, this tensor is used to denote which tokens are original,
    which tokens are generated. 0 for original tokens, 1 for boundary tokens, 2 for generated tokens.
    0 corresponds to encoder_labels [0], 1 corresponds to encoder_labels [0,2,3,4,5],
    2 corresponds to encoder_labels [0,1,2,3,4,5].
    :param encoder_loss_type: 0 for classification, 1 for regression
    :return:
    """
    # start = time.time()
    mask_token_id = tokenizer.mask_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        if isinstance(encoder_inputs, list):
            encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=pad_token_id)
            encoder_inputs = encoder_inputs.to(device)

        attention_mask = torch.zeros(encoder_inputs.shape, dtype=torch.float32).to(device)
        attention_mask = attention_mask.masked_fill(encoder_inputs != pad_token_id, 1)
        pre_predict_outputs = encoder_inputs.clone()  # TODO: 这是个啥？
        # s = time.time()
        # step 1: feed encoder_inputs into the encoder and get encoder_logits
        encoder_outputs, encoder_logits = model.get_encoder_logits(encoder_inputs, attention_mask=attention_mask)
        # e = time.time()
        bts, seqlen = encoder_inputs.shape
        pre_decoder_lengths = [len(e) for e in indicate_labels]
        if decoder_inputs is None:
            # step 2: predict encoder_labels for input_ids based on encoder_logits
            indicate_labels, predict_labels_list = get_encoder_labels(encoder_logits, encoder_loss_type,indicate_labels,
                                                                 max_insert_label,threshold=threshold,max_len=max_len)

            decoder_inputs = [BARTDataset.create_decoder_inputs(encoder_inputs[i].tolist()[:pre_decoder_lengths[i]],
                                                                predict_labels_list[i].tolist(), mask_token_id) for i in range(bts)]
            # logger.logger.info(tokenizer.batch_decode(decoder_inputs))

        decoder_lengths = [len(e) for e in indicate_labels]
        # create decoder_inputs by shifting the decoder_labels right,
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value = pad_token_id)
        decoder_inputs = decoder_inputs.to(device)

        decoder_labels = decoder_inputs.clone()
        decoder_inputs[:, 1:] = decoder_labels[:, :-1]
        decoder_inputs[:, 0] = eos_token_id

        # step 4: feed decoder_inputs into the decoder and get decoder_logits in a non-auto-regressive way.
        # feed the encoder_outputs to avoid computing it again.
        encoder_logits, decoder_logits = model(input_ids=None, decoder_input_ids=decoder_inputs,
                                               attention_mask=attention_mask, encoder_outputs=encoder_outputs,
                                               use_cache=False)[:2]

        if num_beams > 1:
            pass
        else:
            indicate_labels_tensor = [torch.tensor(e) for e in indicate_labels]
            indicate_labels_tensor = pad_sequence(indicate_labels_tensor, batch_first=True, padding_value = 1000)
            indicate_labels_tensor = indicate_labels_tensor.to(device)
            predict_outputs = _generate_no_beam_search_parallel(  # TODO: xieyan
                    decoder_logits,
                    decoder_labels,
                    mask_token_id,
                    indicate_labels_tensor,
                    stop_tokens_tensor=stop_tokens_tensor,
                    sub_tokens_tensor=sub_tokens_tensor,
                    temperature=temperature, do_sample=do_sample,
                    top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                    tokenizer=tokenizer,
                    args=args
            )

        refinement_steps = torch.zeros(bts).long()
        for i in range(bts):
            length1 = decoder_lengths[i]
            length2 = pre_decoder_lengths[i]
            if length1 != length2:
                refinement_steps[i] = 1
            else:
                if torch.sum(predict_outputs[i, :length1] == pre_predict_outputs[i,:length1],dim=-1) != length1:
                    refinement_steps[i] = 1
    return predict_outputs, indicate_labels, refinement_steps, decoder_lengths


def get_encoder_labels(encoder_logits, encoder_loss_type, indicate_labels_list, max_insert_label = 1, threshold=0,
                       max_len = None):
    if encoder_loss_type==0: # classification
        # argmax
        if threshold>0:
            probs = torch.softmax(encoder_logits, dim=-1)
            # encoder_logits[:,:,1:] += 0.7
            _index = probs[:,:,0]>=threshold
            encoder_logits[_index] = 0
            predict_labels = torch.argmax(encoder_logits,dim=-1,keepdim=False)
            predict_labels[_index] = 0
        else:
            predict_labels = torch.argmax(encoder_logits, dim=-1, keepdim=False)
    else:  # regression, round and convert the output into torch.Long tensor
        predict_labels = torch.round(encoder_logits).long()

    for i, e in enumerate(indicate_labels_list):
        if len(e) > max_len+2:
            predict_labels[i][predict_labels[i] == 2] = 1  # change insert to replace

    bts = encoder_logits.shape[0]
    new_indicate_labels_list = []
    predict_labels_list = []
    for b in range(bts):
        new_indicate_labels = []
        indicate_labels = indicate_labels_list[b]
        for i ,e in enumerate(indicate_labels):
            predict_labels[b, i] = min(predict_labels[b, i], max_insert_label + 1)
            if e == 0:  # lexical constraints . only copy is allowed.
                if predict_labels[b, i] != 0:
                    predict_labels[b, i] = 0
            elif e == 1:  # the boundary token of lexical constraints. copy and insert are allowed.
                if predict_labels[b, i] == 1:  # change replacement to copy
                    predict_labels[b, i] = 0
            elif e == 2:  # generated tokens. all actions are allowed.
                pass
            elif e == 3:  # only replace is allowed.
                if predict_labels[b, i] == 2:  # change insertion to replacement
                    predict_labels[b, i] = 1
            else:
                raise ValueError(f'indicate_labels can only be [0,1,2,3].')

            if predict_labels[b, i] > 1:  # insert
                new_indicate_labels += [2]*(predict_labels[b, i]-1)
            new_indicate_labels.append(e)
        new_indicate_labels_list.append(new_indicate_labels)
        predict_labels_list.append(predict_labels[b, :len(indicate_labels)])
    return new_indicate_labels_list, predict_labels_list


def _generate_no_beam_search(
            decoder_logits,
            decoder_inputs,
            bos_token_id,
            eos_token_id,
            mask_token_id,
            indicate_labels_list,
            decoder_lengths,
            stop_tokens_tensor=None,
            sub_tokens_tensor=None,
            temperature=1,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1
):
    if temperature != 1:
        decoder_logits = decoder_logits / temperature
    bts = decoder_inputs.shape[0]
    new_encoder_inputs_list = []
    for b in range(bts):
        new_encoder_inputs = []
        indicate_labels = indicate_labels_list[b]
        seqlen = decoder_lengths[b]
        for i in range(seqlen):
            if i == 0:
                new_encoder_inputs.append(bos_token_id)  #
                continue
            if i == seqlen - 1:
                new_encoder_inputs.append(eos_token_id)  #
                continue
            if decoder_inputs[b, i+1] == mask_token_id:  # we need to predict the mask_token based on the i-th logits
                next_token_logits = decoder_logits[b, i].view(1, -1)
                # set the probability of stop tokens to 0
                if stop_tokens_tensor is not None:
                    next_token_logits = next_token_logits.masked_fill(stop_tokens_tensor >0 , -1e10)
                # forbid to insert sub tokens behind the lexical constraints
                if i>1 and indicate_labels[i-1]<2 and sub_tokens_tensor is not None:
                    next_token_logits = next_token_logits.masked_fill(sub_tokens_tensor > 0, -1e10)

                # repetition penalty
                prev_output_tokens = [new_encoder_inputs + decoder_inputs[b].tolist()]
                next_token_logits = enforce_repetition_penalty_\
                    (next_token_logits, 1, 1, prev_output_tokens=prev_output_tokens,
                     repetition_penalty=repetition_penalty)
                if do_sample:
                    next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    # Sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1)
                new_encoder_inputs.append(next_token.item())
            else:
                new_encoder_inputs.append(decoder_inputs[b, i+1])

        new_encoder_inputs_list.append(torch.tensor(new_encoder_inputs))

    return new_encoder_inputs_list


def generate_caption_step(out, gen_idx, mask, temperature=None, top_k=100):
    """ Generate a word from out[gen_idx]
    args:
        - out (torch.Tensor): tensor of logits of size (batch_size, seq_len, vocab_size)
        - gen_idx (int): location for which to generate for
        - mask (torch.Tensor): (1, vocab_size)
        - extend_ids: (batch_size, extend_len)
        - top_k (int): candidate k
    """
    # logits = out[:, gen_idx]
    logits = out.squeeze(0)
    if temperature is not None:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = probs.topk(top_k, dim=-1)

    # if (extend_ids is not None) and (extend_ids not in top_k_ids):
    # if extend_ids is not None:
    #     # Need to be optimize when extend_ids in top_k_ids
    #     top_k_probs = torch.cat((top_k_probs, torch.gather(probs, dim=-1, index=extend_ids)), dim=-1)
    #     top_k_ids = torch.cat((top_k_ids, extend_ids), dim=-1)

    return top_k_probs, top_k_ids


def conzic_sample_function(lm_logits=None,
                           tokenizer=None,
                           match_model=None,
                           unfinish_seq=None,
                           mask_pos=None,
                           args=None):
    '''
    conzic sample method for language model generate
    paper link: https://arxiv.org/abs/2303.02437
    '''
    probs = F.softmax(lm_logits, dim=-1)
    probs, idxs = probs.topk(args.conzic_top_k, dim=-1)
    topk_seq = unfinish_seq.repeat(idxs.shape[0], 1)
    topk_seq[:, int(mask_pos)] = idxs
    # topk_inp_batch = topk_seq.view(-1, topk_seq.shape[-1])
    batch_text_list = tokenizer.batch_decode(topk_seq, skip_special_tokens=True)
    clip_score, clip_ref = match_model.compute_image_text_similarity_via_raw_text(args.img_embeds, batch_text_list)
    final_score = args.alpha * probs + args.beta * clip_score
    best_clip_id = final_score.argmax(dim=1).view(-1, 1)
    generate_token = idxs[best_clip_id]
    return generate_token


def _generate_no_beam_search_parallel(
            decoder_logits,
            decoder_labels,
            mask_token_id,
            indicate_labels_tensor,
            stop_tokens_tensor=None,
            sub_tokens_tensor=None,
            temperature=1,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1,
            tokenizer=None,
            args=None

):
    """
    parallel for batch and seqlen
    :param decoder_logits:
    :param decoder_labels:
    :param mask_token_id:
    :param indicate_labels_tensor:
    :param stop_tokens_tensor:
    :param sub_tokens_tensor:
    :param temperature:
    :param do_sample:
    :param top_k:
    :param top_p:
    :param repetition_penalty:
    :return:
    """

    if temperature != 1:
        # [b, seq_len, vocab_size]
        decoder_logits = decoder_logits / temperature
    # set the probability of stop tokens to 0
    if stop_tokens_tensor is not None:
        decoder_logits = decoder_logits.masked_fill(stop_tokens_tensor > 0, -1e10)

    # repetition penalty
    decoder_logits = enforce_repetition_penalty_parallel(decoder_logits,
                                                         prev_output_tokens=decoder_labels,
                                                         repetition_penalty=repetition_penalty)

    if sub_tokens_tensor is not None:
        _tmp = indicate_labels_tensor.clone()
        _tmp[:, 1:] = indicate_labels_tensor[:, :-1]
        _tmp[:, 1] = 2
        # forbid to insert sub tokens behind the lexical constraints
        lexical_index = _tmp < 2
        decoder_logits[lexical_index] = decoder_logits[lexical_index].masked_fill(sub_tokens_tensor > 0, -1e10)
    # predict the mask tokens
    mask_token_index = decoder_labels == mask_token_id
    mask_positions = torch.nonzero(mask_token_index[0, :])
    logits = decoder_logits[mask_token_index]
    if logits.shape[0] == 0:
        return decoder_labels
    else:
        if args.conzic_sample:  # TODO: xieyan
            mask_num = logits.shape[0]
            for i in range(mask_num):
                mask_logits = logits[i, :]
                mask_pos = mask_positions[i]
                generate_token = conzic_sample_function(mask_logits, tokenizer, args.vl_model, mask_pos=mask_pos, unfinish_seq=decoder_labels, args=args)
                decoder_labels[:, int(mask_pos)] = generate_token


        # if do_sample:
        #     logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        #     # Sample
        #     probs = torch.softmax(logits, dim=-1)
        #     predict_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        # else:
        #     predict_tokens = torch.argmax(logits, dim=-1)

        # decoder_labels[mask_token_index] = predict_tokens

    return decoder_labels


def _generate_no_beam_search_parallel_v2(
            decoder_logits,
            decoder_labels,
            mask_token_id,
            indicate_labels_tensor,
            stop_tokens_tensor=None,
            sub_tokens_tensor=None,
            temperature = 1,
            do_sample = False,
            top_k  = 0,
            top_p = 1.0,
            repetition_penalty= 1
):
    """
    # the difference between _generate_no_beam_search_parallel and _generate_no_beam_search_parallel_v2 is that:
    prev_output_tokens of the latter method includes last step generated tokens and
    tokens generated in this step before the current token.

     parallel for batch
    :param decoder_logits:
    :param decoder_labels:
    :param mask_token_id:
    :param indicate_labels_tensor:
    :param stop_tokens_tensor:
    :param sub_tokens_tensor:
    :param temperature:
    :param do_sample:
    :param top_k:
    :param top_p:
    :param repetition_penalty:
    :return:
    """
    if temperature != 1:
        # [b, seq_len, vocab_size]
        decoder_logits = decoder_logits / temperature
    # set the probability of stop tokens to 0
    if stop_tokens_tensor is not None:
        decoder_logits = decoder_logits.masked_fill(stop_tokens_tensor > 0, -1e10)

    if sub_tokens_tensor is not None:
        _tmp = indicate_labels_tensor.clone()
        _tmp[:, 1:] = indicate_labels_tensor[:, :-1]
        _tmp[:, 1] = 2
        # forbid to insert sub tokens behind the lexical constraints
        lexical_index = _tmp < 2
        decoder_logits[lexical_index] = decoder_logits[lexical_index].masked_fill(sub_tokens_tensor > 0, -1e10)

    seqlen = decoder_labels.shape[1]
    for i in range(seqlen):
        # predict the mask tokens
        logits = decoder_logits[:, i, :]
        mask_token_index = decoder_labels[:, i] == mask_token_id
        logits = logits[mask_token_index]
        if logits.shape[0] == 0:
            continue
        else:
            # repetition penalty
            logits = enforce_repetition_penalty_parallel(logits,
                                                              prev_output_tokens=decoder_labels[mask_token_index],
                                                              repetition_penalty=repetition_penalty)
            if do_sample:
                logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                # Sample
                probs = torch.softmax(logits, dim=-1)
                predict_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                mask_logits = logits.squeeze(0)
                generate_token = conzic_sample_function(mask_logits, tokenizer, args.vl_model,
                                                        mask_pos=i, unfinish_seq=decoder_labels, args=args)
                # predict_tokens = torch.argmax(logits, dim=-1)

            decoder_labels[:, i][mask_token_index] = generate_token
    return decoder_labels

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    elif top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits

def enforce_repetition_penalty_( lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty=1):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i]):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty
    return  lprobs

def enforce_repetition_penalty_parallel( lprobs, prev_output_tokens, repetition_penalty=1):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    if len(lprobs.shape)==3:
        seqlen = lprobs.shape[1]
        prev_output_tokens = prev_output_tokens.unsqueeze(dim=1).expand(-1,seqlen,-1)
    gather_logits = torch.gather(lprobs, -1, prev_output_tokens)
    gather_logits[gather_logits>0] /= repetition_penalty
    gather_logits[gather_logits < 0] *= repetition_penalty
    lprobs.scatter_(-1, prev_output_tokens, gather_logits)
    return lprobs


def construct_model_inputs(masked_sentences, decoder_inputs_list=None):
    """
    masked_sentences: list of 'keyword1 keyword2 keyword3...'
    """
    indicate_labels_list = []
    encoder_inputs_list = []
    for masked_sentence in masked_sentences:
        indicate_labels = [0]
        encoder_inputs = [tokenizer.bos_token_id]
        words = masked_sentence.split()
        for i, w in enumerate(words):
            ids = tokenizer.encode(' ' + w, add_special_tokens=False)  # 将输入的词都是带着空格输入的，可能是为了与训练相匹配，并且训练的时候大部分词都是前面有空格的
            encoder_inputs += ids
            indicate_labels += [2] + [0] * (len(ids) - 1)  # can insert before the current token
        encoder_inputs.append(tokenizer.eos_token_id)
        indicate_labels.append(1)
        indicate_labels_list.append(indicate_labels)
        encoder_inputs_list.append(encoder_inputs)

    encoder_inputs_list = [torch.tensor(e) for e in encoder_inputs_list]
    if decoder_inputs_list is not None:
        decoder_inputs_list = [torch.tensor(e) for e in decoder_inputs_list]

    return encoder_inputs_list, decoder_inputs_list, indicate_labels_list


def detect_keyword_clip_flite(model, vision_name_list, batch_img_list, clip_model=None, parser_model=None, args=None):
    vision_name = vision_name_list[0]
    vision_dir = args.img_path
    vision_file = os.path.join(vision_dir, vision_name)
    detect_res = model.predict(source=vision_file, save=True)
    category_dict = detect_res[0].names

    # 获得图像的中心点坐标
    (w, h) = detect_res[0].boxes.orig_shape
    x_c, y_c = w / 2, h / 2

    # 将结果按照距离中心点的远近排序
    object_list = []

    distance_list = []
    area_dict = {}  # 用一个字典表示面积的大小，这样方便累加
    for obj_id in range(len(detect_res[0])):

        # 将实体放进列表
        obj_index = int(detect_res[0].boxes.cls[obj_id])
        obj_name = category_dict[obj_index].lower()
        object_list.append(obj_name)

        # 同时将距离计算出来
        obj_pos = detect_res[0].boxes.xywh[obj_id, :]
        obj_x = obj_pos[0]
        obj_y = obj_pos[1]
        obj_w = obj_pos[2]
        obj_h = obj_pos[3]
        area = obj_w * obj_h
        if obj_name not in area_dict.keys():
            area_dict[obj_name] = 0
            area_dict[obj_name] += area
        else:
            area_dict[obj_name] += area
        distance = math.sqrt((obj_x - x_c) ** 2 + (obj_y - y_c) ** 2)
        distance_list.append(distance)
    if len(object_list) == 0:
        return ['']
    else:
        distance_list, object_list = zip(*sorted(zip(distance_list, object_list)))
        distance_list = list(distance_list)
        object_list = list(object_list)

    if args.use_memory:
        print(object_list)
        # 读取memory中最相似的五句话中的concepts
        with open(args.memory_caption_file, 'r') as f:
            memory_captions = json.load(f)
        memory_embedding = torch.load(args.memory_embedding_file)
        image_embedding = args.batch_image_embeds
        clip_score, clip_ref = clip_model.compute_image_text_similarity_via_embeddings(image_embedding, memory_embedding)
        select_memory_ids = clip_score.topk(5, dim=-1)[1].squeeze(0)  # 选出相似度最高的五句话
        select_memory_captions = [memory_captions[id] for id in select_memory_ids]
        scene_graphs = parse(parser_model, parser_tokenizer,
                             text_input=select_memory_captions,
                             device=device)
        type_dict = {}
        concepts = get_graph_phrases(scene_graphs, type_dict)

        keyword_4_select = concepts + list(set(object_list))
        clip_score, clip_ref = clip_model.compute_image_text_similarity_via_Image_text(batch_img_list[0], keyword_4_select)
        select_keywords_id = clip_score.topk(3, dim=-1)[1].tolist()[0]
        keyword_list = []
        for id in select_keywords_id:
            keyword_list.append(keyword_4_select[id])
        print(keyword_list)
        return keyword_list


def detect_keyword(model, vision_name_list, batch_img_list, clip_model=None, parser_model=None, args=None):

    # TODO: batch的方法还没有实现

    # 初始化模型
    # model = YOLO(args.detect_model)

    # 检测
    # vision_name = vision_name_list[0]
    # vision_dir = args.img_path
    # vision_file = os.path.join(vision_dir, vision_name)
    # detect_res = model.predict(source=vision_file, save=True)
    # category_dict = detect_res[0].names
    #
    # # 获得图像的中心点坐标
    # (w, h) = detect_res[0].boxes.orig_shape
    # x_c, y_c = w/2, h/2
    #
    # # 将结果按照距离中心点的远近排序
    # object_list = []
    #
    # distance_list = []
    # area_dict = {}  # 用一个字典表示面积的大小，这样方便累加
    # for obj_id in range(len(detect_res[0])):
    #
    #     # 将实体放进列表
    #     obj_index = int(detect_res[0].boxes.cls[obj_id])
    #     obj_name = category_dict[obj_index]
    #     object_list.append(obj_name)
    #
    #     # 同时将距离计算出来
    #     obj_pos = detect_res[0].boxes.xywh[obj_id, :]
    #     obj_x = obj_pos[0]
    #     obj_y = obj_pos[1]
    #     obj_w = obj_pos[2]
    #     obj_h = obj_pos[3]
    #     area = obj_w * obj_h
    #     if obj_name not in area_dict.keys():
    #         area_dict[obj_name] = 0
    #         area_dict[obj_name] += area
    #     else:
    #         area_dict[obj_name] += area
    #     distance = math.sqrt((obj_x-x_c)**2+(obj_y-y_c)**2)
    #     distance_list.append(distance)
    # if len(object_list) == 0:
    #     return ['']
    # else:
    #     distance_list, object_list_sorted = zip(*sorted(zip(distance_list, object_list)))  # TODO: 原先设计的规则中，object_list变成了object_list_sorted
    #     distance_list = list(distance_list)
    #     object_list_sorted = list(object_list_sorted)

    '''
    memory part code
    用memory中的概念与检测器的label比较，这样可以换一个更合适的keyword
    '''

    if args.use_memory:
        # logger.logger.info(f"Origin Keywords: {object_list}")
        # 读取memory中最相似的五句话中的concepts
        with open(args.memory_caption_file, 'r') as f:
            memory_captions = json.load(f)
        memory_embedding = torch.load(args.memory_embedding_file)
        image_embedding = args.batch_image_embeds
        clip_score, clip_ref = clip_model.compute_image_text_similarity_via_embeddings(image_embedding, memory_embedding)
        select_memory_ids = clip_score.topk(1, dim=-1)[1].squeeze(0)  # 选出相似度最高的五句话
        select_memory_captions = [memory_captions[id] for id in select_memory_ids]

        return select_memory_captions
        scene_graphs = parse(parser_model, parser_tokenizer,
                             text_input=select_memory_captions,
                             device=device)
        type_dict = {}
        concepts = get_graph_phrases(scene_graphs, type_dict)
        logger.logger.info(f"Memory concepts: {concepts}")

        #
        object_set = set(object_list)
        object_indexs = {}

        for object in object_set:
            object_indexs[object] = []
            # 查找set中的这些词在list中出现的位置
            for id, object_in_list in enumerate(object_list):
                if object_in_list == object:
                    object_indexs[object].append(id)
                else:
                    continue

        # 计算每一类检测框的平均clip分数
        area_dict_memory = {}
        for object in object_set:
            concepts.append(object)
            indexs = object_indexs[object]
            object_num = len(indexs)
            img_clip_ref_sum = 0
            croped_clip_ref_sum = 0
            for index_in_list in indexs:
                bbox = [float(detect_res[0].boxes.xyxy[index_in_list, i]) for i in range(4)]
                img = batch_img_list[0]
                croped_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
                # croped_img.show()

                img_clip_score, img_clip_ref = vl_model.compute_image_text_similarity_via_Image_text(img, concepts)  # 合整个图片算相似度
                croped_clip_score, croped_clip_ref = vl_model.compute_image_text_similarity_via_Image_text(croped_img, concepts)  # 合局部图片算相似度
                img_clip_ref_sum += img_clip_ref
                croped_clip_ref_sum += croped_clip_ref
            img_clip_ref_avg = img_clip_ref_sum / object_num
            croped_clip_ref_avg = croped_clip_ref_sum / object_num
            img_clip_score_avg = torch.nn.functional.softmax(img_clip_ref_avg, dim=-1)
            croped_clip_score_avg = torch.nn.functional.softmax(croped_clip_ref_avg, dim=-1)

            # 计算memory中的概念与检测label的cos距离，作为clip分数的权重
            # concepts_embedding = clip_model.compute_text_representation(concepts)
            # label_embedding = concepts_embedding[-1, :].unsqueeze(0)
            # cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            # sim = cos_sim(label_embedding, concepts_embedding).unsqueeze(0)
            # croped_clip_score_avg_balanced = croped_clip_score_avg.mul(sim)

            # 先计算相似度，相似度大于0.8的单词才会被放行，之后再用clip采样
            concepts_embedding = clip_model.compute_text_representation(concepts)
            label_embedding = concepts_embedding[-1, :].unsqueeze(0)

            cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            sim = cos_sim(label_embedding, concepts_embedding).unsqueeze(0)
            best_clip_single = 0.2
            best_label = None
            for id, sim_single in enumerate(sim[0, :-1]):
                # clip_score_single = croped_clip_score_avg[:, id]  # 用局部图像作为重点
                clip_score_single = img_clip_ref_avg[:, id]  # 用整个图像作为重点
                if sim_single > 0.87 and clip_score_single > best_clip_single:
                    best_label = concepts[id]
                    best_clip_single = clip_score_single
                else:
                    continue

            if best_label is None:
                for index_in_list in indexs:
                    object_list[index_in_list] = ''
                logger.logger.info(f'{object} is delted.')
                continue

            for index_in_list in indexs:
                object_list[index_in_list] = best_label
            area_dict_memory[best_label] = area_dict[object]
            _ = concepts.pop(-1)
        logger.logger.info(f"Balanced Keywords: {object_list}")

        keyword_list_split = list(set(object_list))
        keyword_list = [' '.join(keyword_list_split).lower()]
        logger.logger.info(f"Final Keywords: {keyword_list}")
        return keyword_list
        # area_dict_memory = {}
        # for object_id, object_name in enumerate(object_list):
        #     # 切分图像
        #     bbox = [float(detect_res[0].boxes.xyxy[object_id, i]) for i in range(4)]
        #     img = batch_img_list[0]
        #     croped_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        #     croped_img.show()
        #
        #     # 计算被选label与局部图像的相似度
        #     concepts.append(object_name)
        #     croped_clip_score, croped_clip_ref = vl_model.compute_image_text_similarity_via_Image_text(croped_img, concepts)
        #     best_label_index = torch.argmax(croped_clip_score, dim=-1)
        #     best_label = concepts[best_label_index]
        #     object_list[object_id] = best_label
        #     area_dict_memory[best_label] = area_dict[object_name]


    '''
    rules for processing keywords
    '''
    # keyword_list = []
    # keyword_set = set(object_list)
    # keyword_count = Counter(object_list)
    #
    # if len(keyword_set) >= 3:
    #     keyword_cut_list = []
    #     if 'person' in keyword_set:
    #         keyword_cut_list.append('person')
    #         _ = [value for value in map(lambda index: area_dict.pop(index) if area_dict.get(index) else None, ['person'])]
    #         if args.use_memory:
    #             area_dict_order = sorted(area_dict_memory.items(), key=lambda x: x[1], reverse=True)
    #         else:
    #             area_dict_order = sorted(area_dict.items(), key=lambda x: x[1], reverse=True)
    #         for i, keyword_area in enumerate(area_dict_order):
    #             if i >= 2:
    #                 break
    #             else:
    #                 keyword_cut_list.append(keyword_area[0])
    #     else:
    #         if args.use_memory:
    #             area_dict_order = sorted(area_dict_memory.items(), key=lambda x: x[1], reverse=True)
    #         else:
    #             area_dict_order = sorted(area_dict.items(), key=lambda x: x[1], reverse=True)
    #         for i, keyword_area in enumerate(area_dict_order):
    #             if i >= 3:
    #                 break
    #             else:
    #                 keyword_cut_list.append(keyword_area[0])
    #     for keyword in keyword_cut_list:
    #         count = keyword_count[keyword]
    #         if count > 1:
    #             blob_keyword = TextBlob(keyword)
    #             if len(blob_keyword.words) > 1:
    #                 keyword_list.append(blob_keyword.words[0] + ' ' + blob_keyword.words.pluralize()[-1])
    #             else:
    #                 keyword_list.append(str(blob_keyword.words.pluralize()[0]))
    #         else:
    #             keyword_list.append(keyword)
    #     keyword_list = [' '.join(keyword_list)]
    # else:
    #     for keyword in keyword_count.keys():
    #         count = keyword_count[keyword]
    #         if count > 1:
    #             words = nltk.word_tokenize(keyword)
    #             pos = nltk.pos_tag(words)
    #             if pos[-1][1] == 'NNS':
    #                 keyword_list.append(keyword)
    #             else:
    #                 blob_keyword = TextBlob(keyword)
    #                 keyword_list.append(str(blob_keyword.words.pluralize()[-1]))
    #         else:
    #             keyword_list.append(keyword)
    #     keyword_list = [' '.join(keyword_list)]
    # logger.logger.info(f"Finally keywords: {keyword_list}")
    return keyword_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text infilling.")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--bart', type=str, default='large', choices=['base', 'large'])
    parser.add_argument('--refinement_steps', type=int, default=10, help='The number of refinements for each input.')
    parser.add_argument('--adaptive', type=bool, default=False, help='The number of refinements is on the fly but '
                                                                     'no bigger than max_refinement_steps')
    parser.add_argument('--max_refinement_steps', type=int, default=30, help='The maximum number of refinements for each input.')
    parser.add_argument('--max_len', type=int, default=40, help='The maximum length of the generated sentence.')
    parser.add_argument('--temperature', type=float, default=1,
                        help='The value used to module the next token probabilities. Must be strictly positive. Default to 1.0.')
    parser.add_argument('--repetition_penalty', type=float, default=2,
                        help='Between 1.0 and infinity.1.0 means no penalty.Default to 1.0.')
    parser.add_argument('--threshold', type=float, default=0,
                        help='Between 0 and 1. 0 means no threshold for copy action. Default to 0.')

    parser.add_argument('--top_k', type=int, default=0,
                        help='The number of highest probability vocabulary tokens to keep for top-k-filtering. Between 1 and infinity.')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling. '
                             'Must be between 0 and 1.')
    parser.add_argument('--decoder_chain', type=int, default=1,
                        help='the number of parallel chains for decoder, each chain refers to an unique token sequence.')
    parser.add_argument('--do_sample', type=int, default=0,
                        help='if 0 decode with greedy method, otherwise decode with top_k or top_p.')
    parser.add_argument('--encoder_loss_type', type=int, default=0, help='0 is classification loss, 1 is regression loss')
    parser.add_argument('--dataset', type=str, default='one-billion-words',
                        choices=['yelp_review', 'one-billion-words'])
    parser.add_argument('--insert_mode', type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help='0 means using the left part, 1 means using the middle part, 2 means using the right part,'
                             '3 means randomly selecting, 4 means selecting the tokens with highest weight')
    parser.add_argument('--max_insert_label', type=int, default=1, help='the maximum number of tokens to be inserted before a token.')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='0 for copy, 1 for replace, 2-5 means insert 1-4 tokens')
    parser.add_argument('--generate_mode', type=int, default=0, choices=[0, 1, 2, 3],
                        help = '0 for random, 1 for lm, 2 for combination')
    parser.add_argument('--masked_lm', type=float, default=0, help='0 for using language modeling for the decoder,'
                                                                   '1 for using mask language modeling for the decoder.')
    parser.add_argument('--full_mask', type=float, default=0, help='0 for using casual mask attention for decoder, '
                                                                    '1 for without using casual mask attention for decoder.')
    parser.add_argument('--w', type=float, default=1.0, help='The weight for the encoder loss')

    parser.add_argument('--num_keywords', type=int, default=4, choices=[1, 2, 3, 4, 5, 6])  # demo use
    parser.add_argument('--random_init', type=int, default=0, help='0 denotes initialization with BART; '
                                                                  '1 denotes random initialization.')

    # conzic sample paras
    parser.add_argument('--conzic_sample', type=bool, default=True, help='conzic sample means a way to process logits by conzic method'
                                                                         'https://arxiv.org/abs/2303.02437')
    parser.add_argument('--clip_model', type=str, default='/media/xieyan/Hard Disk2/pretrain_model/clip_weights')
    # COCO：/media/xieyan/Hard Disk2/datasets/mscoco/mscoco/test_images
    # flickr30k:/media/xieyan/Hard Disk2/datasets/flickr30k/flickr30k-images
    parser.add_argument('--img_path', type=str, default='/media/xieyan/Hard Disk2/datasets/mscoco/mscoco/test_images')
    parser.add_argument('--conzic_top_k', type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.2, help="weight for fluency")
    parser.add_argument("--beta", type=float, default=0.8, help="weight for image-matching degree")

    # detect paras
    parser.add_argument("--use_detect", type=bool, default=True)
    parser.add_argument("--wordnet", type=bool, default=False, help="use world Net to expand keywords")
    parser.add_argument("--detect_path", type=str, default='/media/xieyan/Hard Disk2/pretrain_model/yolov8/yolov8x_coco.pt')
    parser.add_argument("--save_path", type=str, default='./conzic_sample_results')

    parser.add_argument("--use_memory", type=bool, default=True)
    parser.add_argument("--memory_caption_file", type=str, default='../data/memory/mscoco/train_captions.json')
    parser.add_argument("--memory_embedding_file", type=str, default='../data/memory/mscoco/train_embedding.pt')
    parser.add_argument("--parser_checkpoint", type=str, default='/media/xieyan/Hard Disk2/pretrain_model/flan-t5-base-VG-factual-sg')
    args = parser.parse_args()
    # set the gpu(s) for code
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 是否使用自回归的方式构建decoder
    if args.masked_lm == 0:
        masked_lm = ''
    else:
        masked_lm = 'masked_lm_'

    # 自回归的decoder使用casual mask，mask lm使用双向注意力
    if args.full_mask == 0:
        full_mask = ''
    else:
        full_mask = 'full_mask_'

    # TODO：判断这个参数对模型生成有什么影响
    if args.generate_mode == 0:
        generate_mode = ''
    elif args.generate_mode == 1:
        generate_mode = 'lm_generate_'
    elif args.generate_mode == 2:
        generate_mode = f'combine_generate_{args.ratio1}_{args.ratio2}_'
    else:
        raise ValueError('Wrong generate mode.')

    # prefix 只对load模型和logger有影响
    prefix = '{}_{}{}{}w{}_max_insert_label{}_insert_mode{}_encoder_loss_type{}'.\
    format(args.dataset, masked_lm, full_mask, generate_mode, args.w, args.num_labels-2, args.insert_mode, args.encoder_loss_type)

    if args.random_init == 1:
        prefix = 'random_initialization_'+prefix
    prefix = f"cbart-{args.bart}_{prefix}"

    model_path = f'../checkpoints/{prefix}'
    if args.do_sample:
        if args.top_k > 0:
            prefix += f'_sample_top_k_{args.top_k}'
        else:
            prefix += f'_sample_top_p_{args.top_p}'
        if args.decoder_chain > 1:
            prefix += f'_decoder_chain{args.decoder_chain}'
    if args.threshold > 0:
        prefix += f'_threshold{args.threshold}'

    prefix += f'_{args.num_keywords}keywords'

    log_path = f'../logs/generate_keywords'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    output_path = f'../outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = '{}/{}.txt'.format(output_path, prefix)

    args.output_file = output_file
    args.model_path = model_path
    args.log_path = log_path

    if args.conzic_sample:
        log_file = f'conzic_flickr30k_alpha{args.alpha}_beta{args.beta}_detect_{args.use_detect}_wordNet{args.wordnet}.log'
    else:
        log_file = '{}/{}.log'.format(log_path, prefix)
    logger = Logger(log_file)
    logger.logger.info(f'The log file is {log_file}')
    logger.logger.info(f'output file is {args.output_file}')
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
        if args.use_memory:
            img_data = Imgdata_img_return(dir_path=args.img_path, match_model=vl_model)
            train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img_img_return, shuffle=False, drop_last=False)
        else:
            img_data = Imgdata(dir_path=args.img_path, match_model=vl_model)
            train_loader = DataLoader(img_data, batch_size=args.batch_size, collate_fn=collate_img, shuffle=False, drop_last=False)
        args.vl_model = vl_model
        if args.use_memory:
            parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint)
            parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint)
            parser_model.eval()
            parser_model.to(device)
    else:
        vl_model = None
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
        if args.use_detect:
            detect_model = YOLO(args.detect_path)
            logger.logger.info(f'Using detection {detect_model.__class__.__name__}')
        else:
            detect_model = None
        save_prefix = f'conzic_coco_alpha{args.alpha}_beta{args.beta}_{detect_model.__class__.__name__}_memory{args.use_memory}_CLIPe.json'
        result_dict = {}
        for batch_idx, (batch_embeds, batch_name_list, batch_img_list) in enumerate(train_loader):
            logger.logger.info(f'{batch_idx + 1}/31784, image name: {batch_name_list[0]}')
            args.batch_image_embeds = batch_embeds
            args.batch_image_name = batch_name_list

            # 准备需要caption的关键词
            if args.use_detect:
                assert detect_model is not None
                if args.use_memory:
                    masked_sentences = detect_keyword(detect_model, batch_name_list, batch_img_list,
                                                      clip_model=vl_model, parser_model=parser_model, args=args)
                    # masked_sentences = ['bus night']
                else:
                    masked_sentences = detect_keyword(detect_model, batch_name_list, args)
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
            #
            # # 准备模型的输入
            # args.img_embeds = batch_embeds
            #
            # encoder_inputs_list, decoder_inputs_list, indicate_labels_list = construct_model_inputs(masked_sentences,
            #                                                                                         decoder_inputs_list=None)
            #
            # indicate_labels = indicate_labels_list[0:1]
            # encoder_inputs = encoder_inputs_list[0:1]
            # masked_sentence = masked_sentences[0:1]
            # if decoder_inputs_list is not None:
            #     decoder_inputs = decoder_inputs_list[0:1]
            # else:
            #     decoder_inputs = None
            # length = len(encoder_inputs_list)
            # batch_size = args.batch_size
            start = time.time()
            """
            batch forward use
            # indicate_labels = indicate_labels_list[i:i + batch_size]
            # encoder_inputs = encoder_inputs_list[i:i + batch_size]
            # masked_sentence = masked_sentences[i:i + batch_size]
            # if decoder_inputs_list is not None:
            #     decoder_inputs = decoder_inputs_list[i:i + batch_size]
            # else:
            #     decoder_inputs = None
            """

            # 生成
            # predict_outputs, refinement_steps = generate_function(model, tokenizer, encoder_inputs, indicate_labels,  # TODO: xieyan
            #                                                       args.encoder_loss_type,
            #                                                       args.max_insert_label,
            #                                                       device,
            #                                                       decoder_inputs=decoder_inputs,
            #                                                       stop_tokens_tensor=stop_tokens_tensor,
            #                                                       sub_tokens_tensor=sub_tokens_tensor,
            #                                                       temperature=args.temperature,
            #                                                       do_sample=args.do_sample,
            #                                                       top_k=args.top_k,
            #                                                       top_p=args.top_p,
            #                                                       refinement_steps=args.refinement_steps,
            #                                                       max_refinement_steps=args.max_refinement_steps,
            #                                                       adaptive=args.adaptive,
            #                                                       repetition_penalty=args.repetition_penalty,
            #                                                       threshold=args.threshold,
            #                                                       decoder_chain=args.decoder_chain,
            #                                                       rank_lm=rank_lm,
            #                                                       max_len=args.max_len,
            #                                                       args=args,
            #                                                       logger=logger
            #                                                       )
            # result_dict[os.path.splitext(batch_name_list[0])[0]] = tokenizer.decode(predict_outputs[0], skip_special_tokens=True)
            result_dict[os.path.splitext(batch_name_list[0])[0]] = masked_sentences[0]
            used_time = time.time() - start
            logger.logger.info(f'using {used_time}s')
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        save_file = os.path.join(args.save_path, save_prefix)
        with open(save_file, 'w', encoding="utf-8") as _json:
            json.dump(result_dict, _json)

            # print(f'\rProcess {i + batch_size}/{length}, {used_time / (i + batch_size):.0f}, {used_time:.1f}',
            #       end='')

    """
    原版生成过程
    """
    #
    # # generate sentences with lexical constraints
    # input_file = f'../data/{args.dataset}/{args.num_keywords}keywords.txt'
    # print(f'Generate sentences with lexical constraints for {input_file}.')
    # masked_sentences = []
    # with open(input_file) as fr:
    #     for i, line in enumerate(fr):
    #         if i % 3 == 0:
    #             continue
    #         else:
    #             line = line.strip().split('\t')[1]
    #             if i % 3 == 1:
    #                 masked_sentences.append(line)  # lexical constraints
    #
    # # construct encoder_inputs and indicate labels for bart
    # # indicate labels是用于约束当前单词的生成label应该是什么
    # # encoder_labels : 0 for copy, 1 for replacement, 2 for insertion
    # # indicate_labels: 0 for copy, 1 for copy and insertion, 2 for copy, replacement and insertion, 3 for replacement
    # # 输入的关键词是除了开头外，全部都是1, 生成的单词indicate labels是2
    # indicate_labels_list = []
    # encoder_inputs_list = []
    # decoder_inputs_list = None
    # for masked_sentence in masked_sentences:
    #     indicate_labels = [0]
    #     encoder_inputs = [tokenizer.bos_token_id]
    #     words = masked_sentence.split()
    #     for i, w in enumerate(words):
    #         ids = tokenizer.encode(' '+w, add_special_tokens=False)  # 将输入的词都是带着空格输入的，可能是为了与训练相匹配，并且训练的时候大部分词都是前面有空格的
    #         encoder_inputs += ids
    #         indicate_labels += [1]+[0]*(len(ids)-1)  # can insert before the current token
    #     encoder_inputs.append(tokenizer.eos_token_id)
    #     indicate_labels.append(1)
    #     indicate_labels_list.append(indicate_labels)
    #     encoder_inputs_list.append(encoder_inputs)
    #
    # encoder_inputs_list = [torch.tensor(e) for e in encoder_inputs_list]
    # if decoder_inputs_list is not None:
    #     decoder_inputs_list = [torch.tensor(e) for e in decoder_inputs_list]
    #
    # length = len(encoder_inputs_list)
    # batch_size = args.batch_size
    # start = time.time()
    # with open(output_file, 'w') as fw:
    #     if args.conzic_sample:
    #         for batch_idx, (batch_embeds, batch_name_list) in enumerate(train_loader):
    #             i = 0
    #             args.img_embeds = batch_embeds
    #             indicate_labels = indicate_labels_list[i:i + batch_size]
    #             encoder_inputs = encoder_inputs_list[i:i + batch_size]
    #             masked_sentence = masked_sentences[i:i + batch_size]
    #             if decoder_inputs_list is not None:
    #                 decoder_inputs = decoder_inputs_list[i:i + batch_size]
    #             else:
    #                 decoder_inputs = None
    #             predict_outputs, refinement_steps = generate_function(model, tokenizer, encoder_inputs, indicate_labels,  # TODO: xieyan
    #                                                          args.encoder_loss_type,
    #                                                          args.max_insert_label,
    #                                                          device,
    #                                                          decoder_inputs=decoder_inputs,
    #                                                          stop_tokens_tensor=stop_tokens_tensor,
    #                                                          sub_tokens_tensor=sub_tokens_tensor,
    #                                                          temperature=args.temperature,
    #                                                          do_sample=args.do_sample,
    #                                                          top_k=args.top_k,
    #                                                          top_p=args.top_p,
    #                                                          refinement_steps=args.refinement_steps,
    #                                                          max_refinement_steps=args.max_refinement_steps,
    #                                                          adaptive=args.adaptive,
    #                                                          repetition_penalty=args.repetition_penalty,
    #                                                          threshold=args.threshold,
    #                                                          decoder_chain=args.decoder_chain,
    #                                                          rank_lm=rank_lm,
    #                                                          max_len=args.max_len,
    #                                                          args=args
    #                                                          )
    #             batch_size = len(indicate_labels)
    #             for b in range(batch_size):
    #                 fw.write(str(i + b) + '\n')
    #                 fw.write('Refinement steps:\t' + str(refinement_steps[b].item()) + '\n')
    #                 fw.write('Masked sentence:\t' + masked_sentence[b] + '\n')
    #                 fw.write('Generated sentence:\t' +
    #                          tokenizer.decode(predict_outputs[b].tolist()[1:-1], clean_up_tokenization_spaces=False) + '\n')
    #             used_time = time.time() - start
    #             print(f'\rProcess {i + batch_size}/{length}, {used_time / (i + batch_size):.0f}, {used_time:.1f}',
    #                   end='')
    #     else:
    #         for i in range(0, length, batch_size):
    #             indicate_labels = indicate_labels_list[i:i+batch_size]
    #             encoder_inputs = encoder_inputs_list[i:i+batch_size]
    #             masked_sentence = masked_sentences[i:i+batch_size]
    #             if decoder_inputs_list is not None:
    #                 decoder_inputs = decoder_inputs_list[i:i+batch_size]
    #             else:
    #                 decoder_inputs = None
    #             predict_outputs, refinement_steps = generate_function(model, tokenizer, encoder_inputs, indicate_labels,
    #                     args.encoder_loss_type,
    #                     args.max_insert_label,
    #                     device,
    #                     decoder_inputs=decoder_inputs,
    #                     stop_tokens_tensor=stop_tokens_tensor,
    #                     sub_tokens_tensor=sub_tokens_tensor,
    #                     temperature=args.temperature,
    #                     do_sample=args.do_sample,
    #                     top_k=args.top_k,
    #                     top_p=args.top_p,
    #                     refinement_steps=args.refinement_steps,
    #                     max_refinement_steps=args.max_refinement_steps,
    #                     adaptive=args.adaptive,
    #                     repetition_penalty=args.repetition_penalty,
    #                     threshold=args.threshold,
    #                     decoder_chain=args.decoder_chain,
    #                     rank_lm=rank_lm,
    #                     max_len = args.max_len,
    #                     vl_model=vl_model
    #             )
    #             batch_size = len(indicate_labels)
    #             for b in range(batch_size):
    #                 fw.write(str(i+b)+'\n')
    #                 fw.write('Refinement steps:\t'+str(refinement_steps[b].item())+'\n')
    #                 fw.write('Masked sentence:\t'+masked_sentence[b]+'\n')
    #                 fw.write('Generated sentence:\t'+
    #                          tokenizer.decode(predict_outputs[b].tolist()[1:-1], clean_up_tokenization_spaces=False)+'\n')
    #             used_time = time.time() - start
    #             print(f'\rProcess {i+batch_size}/{length}, {used_time/(i+batch_size):.0f}, {used_time:.1f}',
    #                   end='')
    #     logger.logger.info(f'\n{length} sentences using {used_time:.1f} seconds.')

