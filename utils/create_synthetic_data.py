# -*- coding: utf-8 -*-
# @Time    : 2020/4/15 11:19 AM
# @Author  : He Xingwei

"""
this script is used to create synthetic paired data for bart
copy: 0
replace: 1
insert: 2, 3, 4, 5 (2 means inserting 1 token)
"""

import torch
import numpy as np
import sys,os
import argparse
import random
import time
from transformers import BartTokenizer, BartForConditionalGeneration
from torch.nn.utils.rnn import pad_sequence
sys.path.append('../')


def create_replaced_samples(input_ids_list, length_list, tf_idf_list=None, model=None, tokenizer=None,
    insert_mode=None, max_insert_label = 4, generate_mode=0,**kwargs):
    """

    :param input_ids: list
    :param length: the length does not include the bos_token_id and the eos_token_id
    :return:
    """
    positions_list = []
    incorrect_input_ids_list = []
    label_ids_list = []
    target_ids_list = []
    original_positions_list = []
    original_input_ids_list = []
    for input_ids, length, tf_idf in zip(input_ids_list, length_list, tf_idf_list):
        for i in range(3):
            assert length >= args.min_length

            # randomly draw a segment from input_ids
            sublen = random.randint(args.min_length - 1, length)
            start_id = random.randint(0, length-sublen)
            end_id = start_id+sublen

            incorrect_input_ids = input_ids[start_id:end_id][:]
            incorrect_input_ids[0] = tokenizer.bos_token_id
            incorrect_input_ids[-1] = tokenizer.eos_token_id
            label_ids = [0]
            # insert
            pre_target_id = [tokenizer.bos_token_id]
            if start_id != 0:
                insert_label = min(max_insert_label, start_id)+1
                label_ids.append(insert_label)
                if start_id <= max_insert_label:
                    pre_target_id = input_ids[:start_id+1]
                else:
                    pos = insert(1, start_id+1, max_insert_label, insert_mode, tf_idf[1:start_id+1])
                    pre_target_id = [tokenizer.bos_token_id]
                    for p in pos:
                        pre_target_id.append(input_ids[p])
            if start_id == 0:
                label_ids += [0]*(sublen-2)
            else:
                label_ids += [0]*(sublen-3)
            pos_target_id = [tokenizer.eos_token_id]
            if end_id != length:
                insert_label = min(max_insert_label, length-end_id)+1
                label_ids.append(insert_label)

                if length - end_id <= max_insert_label:
                    pos_target_id = input_ids[end_id-1:]
                else:
                    pos_target_id = []
                    pos = insert(end_id-1, length, max_insert_label, insert_mode, tf_idf[end_id-1:length])
                    for p in pos:
                        pos_target_id.append(input_ids[p])
                    pos_target_id.append(tokenizer.eos_token_id)
            else:
                label_ids.append(0)

            target_ids = pre_target_id + input_ids[start_id+1:end_id-1]+ pos_target_id

            incorrect_input_ids_list.append(incorrect_input_ids)
            label_ids_list.append(label_ids)
            target_ids_list.append(target_ids)

            # sample the number of replace tokens
            num_replace_tokens = max(1,int(0.15*(sublen-2)))
            if start_id==0:
                replace_tokens_pos = np.random.choice(sublen - 2, num_replace_tokens, replace=False) + 1
            else:
                replace_tokens_pos = np.random.choice(sublen - 3, num_replace_tokens, replace=False) + 2

            replace_tokens_pos = replace_tokens_pos.tolist()
            replace_tokens_pos = sorted(replace_tokens_pos)
            # print(replace_tokens_pos)
            positions_list.append(replace_tokens_pos)

            original_positions_list.append([p+start_id for p in replace_tokens_pos])
            original_input_ids_list.append(input_ids)

        # construct 1 sentences only with replacement
        for j in range(2):
            # sample the number of replace tokens
            num_replace_tokens = max(1,int(0.15*(length-2)))
            replace_tokens_pos = np.random.choice(length - 2, num_replace_tokens, replace=False) + 1
            replace_tokens_pos = replace_tokens_pos.tolist()
            replace_tokens_pos = sorted(replace_tokens_pos)

            incorrect_input_ids = input_ids[:]
            label_ids = [0]*len(incorrect_input_ids)
            target_ids = input_ids[:]
            incorrect_input_ids_list.append(incorrect_input_ids)
            label_ids_list.append(label_ids)
            target_ids_list.append(target_ids)

            positions_list.append(replace_tokens_pos)

            original_positions_list.append(replace_tokens_pos[:])
            original_input_ids_list.append(input_ids)


    if generate_mode == 0:
        for incorrect_input_ids, label_ids, positions, target_ids in zip(incorrect_input_ids_list, label_ids_list, positions_list,target_ids_list):
            for p in positions:
                # random generate the replaced token id
                replaced_token_id = random.randint(0, tokenizer.vocab_size-1)
                while replaced_token_id in [incorrect_input_ids[p], tokenizer.bos_token_id, tokenizer.eos_token_id,tokenizer.pad_token_id, tokenizer.mask_token_id]:
                    replaced_token_id = random.randint(0, tokenizer.vocab_size - 1)
                incorrect_input_ids[p] = replaced_token_id
                label_ids[p] = 1
            assert len(incorrect_input_ids) == len(label_ids)
            assert sum([e if e>1 else 1 for e in label_ids]) == len(target_ids)

    else:
        # construct encoder_inputs and decoder_inputs
        encoder_inputs_list = []
        decoder_inputs_list = []
        for original_input_ids,  original_positions, positions in zip(original_input_ids_list, original_positions_list, positions_list):

            encoder_inputs = original_input_ids[:]
            for p in original_positions:
                encoder_inputs[p] = tokenizer.mask_token_id
            encoder_inputs_list.append(torch.tensor(encoder_inputs))
            decoder_inputs_list.append(torch.tensor(original_input_ids[:]))

        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(encoder_inputs_list, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        encoder_inputs = pad_sequence(encoder_inputs_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        # create decoder_inputs by shifting the decoder_labels right,
        _tmp = decoder_inputs.clone()
        decoder_inputs[:, 1:] = _tmp[:, :-1]
        decoder_inputs[:, 0] = tokenizer.eos_token_id
        with torch.no_grad():
            encoder_inputs = encoder_inputs.to('cuda')
            decoder_inputs = decoder_inputs.to('cuda')
            attention_mask = attention_mask.to('cuda')
            logits, = model(encoder_inputs, attention_mask=attention_mask, decoder_input_ids=decoder_inputs, labels=None, use_cache=False)[:1]

        i = 0
        for incorrect_input_ids, label_ids, original_positions, positions in \
                zip(incorrect_input_ids_list, label_ids_list, original_positions_list, positions_list):
            # print('-'*20)
            # print(tokenizer.convert_ids_to_tokens(incorrect_input_ids))
            # draw a replace token from top_20 tokens
            for original_p, p in zip(original_positions, positions):
                token_logits = logits[i, original_p]
                topk = 20
                top_k_conditional_probs, top_k_token_ids = torch.topk(token_logits, topk, dim=-1)
                top_k_token_ids = top_k_token_ids.cpu()
                # print(tokenizer.convert_ids_to_tokens(top_k_token_ids))
                # print(tokenizer.convert_ids_to_tokens([incorrect_input_ids[p]]))
                sample_id = random.randint(0, topk - 1)
                replaced_token_id = top_k_token_ids[sample_id]

                while replaced_token_id in [incorrect_input_ids[p], tokenizer.bos_token_id, tokenizer.eos_token_id,
                                            tokenizer.pad_token_id, tokenizer.mask_token_id]:
                    sample_id = (sample_id + 1) % topk
                    replaced_token_id = top_k_token_ids[sample_id]
                incorrect_input_ids[p] = replaced_token_id.item()
                label_ids[p] = 1

            # print(tokenizer.convert_ids_to_tokens(incorrect_input_ids))
            # print(tokenizer.convert_ids_to_tokens(target_ids_list[i]))
            # print(label_ids_list[i])
            assert len(incorrect_input_ids) == len(label_ids)
            assert sum([e if e>1 else 1 for e in label_ids]) == len(target_ids_list[i])
            i+=1



    return incorrect_input_ids_list, label_ids_list,target_ids_list


def create_inserted_samples(input_ids_list=None, length_list=None, tf_idf_list=None, model=None, tokenizer=None,
                            insert_mode=None, max_insert_label=4, **kwargs):
    """
    :param input_ids_list:
    :param length_list:
    :param insert_mode: 0 means using the left part, 1 means using the middle part, 2 means using the right part,
    3 means randomly selecting, 4 means selecting the tokens with highest weight
    :param max_insert_label:
    :param kwargs:
    :return:
    """
    incorrect_input_ids_list = []
    label_ids_list = []
    target_ids_list = []
    for input_ids, length, tf_idf in zip(input_ids_list,length_list, tf_idf_list):
        for i in range(5):
            assert length >3
            # sample the number  of deleted tokens
            num_delete_tokens = random.randint(1, length-3)
            delete_tokens = np.random.choice(length-2, num_delete_tokens, replace=False) + 1
            delete_tokens = delete_tokens.tolist()
            delete_tokens = sorted(delete_tokens)
            # add a token, so that while loop can end normally.
            # delete_tokens = [1,2,3,4,5,10,11,12,13,14,15,16]
            delete_tokens.append(100000)
            left_tokens = np.setdiff1d(np.arange(length), delete_tokens)
            left_tokens = left_tokens.tolist()
            label_ids = []
            target_ids = []
            j = 0
            # print(delete_tokens)
            # print(left_tokens)
            for i in left_tokens:
                if i <delete_tokens[j]: # copy
                    label_ids.append(0)
                else:
                    k = j
                    while i > delete_tokens[j]:
                        j+=1
                    # the blank is [k, j), so the number of deleted tokens is j-k
                    insert_label = min(max_insert_label, j-k)+1
                    label_ids.append(insert_label)
                    if j-k<=max_insert_label:
                        while k<j:
                            target_ids.append(input_ids[delete_tokens[k]])
                            k+=1
                    else:
                        # print(k,j, list(range(k,j)))
                        start = delete_tokens[k]
                        end = i
                        _tf_idf = tf_idf[start:end][:]
                        pos = insert(k, j, max_insert_label, insert_mode, _tf_idf)

                        # print(pos)
                        for p in pos:
                            target_ids.append(input_ids[delete_tokens[p]])
                # add left tokens
                target_ids.append(input_ids[i])
            incorrect_input_ids_list.append([input_ids[p] for p in left_tokens])
            label_ids_list.append(label_ids)
            target_ids_list.append(target_ids)
            # print(incorrect_input_ids_list)
            # print(label_ids)
            # print(target_ids)
            # print(sum([e  if e > 1 else 1 for e in label_ids]),len(target_ids))
            assert len(incorrect_input_ids_list[-1]) == len(label_ids)
            assert sum([e  if e > 1 else 1 for e in label_ids]) == len(target_ids)

    return incorrect_input_ids_list, label_ids_list, target_ids_list


def insert(start, end, max_insert_label, insert_mode, tf_idf):
    # [start, end)
    pos = []
    if insert_mode == 0:  # select the left part
        for z in range(max_insert_label):
            pos.append(z + start)
    elif insert_mode == 1:  # select the middle part
        start += (end - start - max_insert_label + 1) / 2
        start = int(start)
        for z in range(max_insert_label):
            pos.append(z + start)
    elif insert_mode == 2:  # select the right part
        for z in range(max_insert_label):
            pos.append(end - max_insert_label + z)
    elif insert_mode == 3:  # randomly select
        pos = np.random.choice(end - start, max_insert_label, replace=False) + start
        pos = pos.tolist()
        pos = sorted(pos)
    else:
        _, indices = torch.topk(torch.tensor(tf_idf), k=max_insert_label)
        indices += start
        pos = indices.tolist()
        pos = sorted(pos)
    return pos


def create_synthetic_data(output_file, input_tensors, lengths, tf_idf_list, model, tokenizer, args):
    dataset_size = args.dataset_size
    batch_size = args.batch_size

    incorrect_input_ids_list = []
    label_ids_list = []  # encoder label
    target_ids_list = []  # target sentence
    if dataset_size==-1:
        dataset_size = len(lengths)
    funcs = [create_replaced_samples,create_inserted_samples]
    # funcs = [create_replaced_samples,]
    j = 0
    start = time.time()
    sub_inputs = []
    sub_lengths = []
    sub_tf_idf = []
    total_len = 0
    index = 0
    generate_mode = args.generate_mode
    if generate_mode==2:
        generate_mode = 0
        print('combine generate mode and first generate with random mode.')
    elif generate_mode==1:
        print('lm generate mode.')
    elif generate_mode==0:
        print('random generate mode.')
    else:
        raise ValueError('wong generate mode.')
    for input_ids, length, tf_idf in zip(input_tensors, lengths, tf_idf_list):
        index += 1
        if length > args.max_length+2 or length < args.min_length+2:
            continue
        sub_inputs.append(input_ids[:])
        sub_lengths.append(length)
        sub_tf_idf.append(tf_idf[:])
        j += 1
        if j == batch_size or index == dataset_size:
            for f in funcs:
                incorrect_input_ids, label_ids, target_ids = f(input_ids_list=sub_inputs,length_list=sub_lengths, tf_idf_list=sub_tf_idf,
                                                               model=model, tokenizer=tokenizer, insert_mode=args.insert_mode,
                                                               max_insert_label=args.max_insert_label, generate_mode = generate_mode)
                incorrect_input_ids_list += incorrect_input_ids
                label_ids_list+=label_ids
                target_ids_list+=target_ids
            #     print('-'*100)
            #     print(incorrect_input_ids)
            #     print(label_ids)
            #     print(target_ids)
            # return
            total_len += j
            if total_len % 1000 == 0:
                print(f'''\r{total_len}/{dataset_size}, {len(label_ids_list)}, use {time.time()-start:.1f} seconds.''', end='')
            if total_len >= dataset_size:
                print()
                break
            sub_inputs = []
            sub_lengths = []
            sub_tf_idf = []
            j = 0
        if args.generate_mode == 2 and generate_mode == 0:
            if index >= args.ratio * dataset_size:
                generate_mode = 1
                print('combine generate mode and secondly generate with lm mode.')

    # save as list to save disk
    data_dict = {'incorrect_input_ids_list': incorrect_input_ids_list,
                 'label_ids_list': label_ids_list,
                 'target_ids_list':target_ids_list}
    print(f'incorrect_input_ids_list length: {len(incorrect_input_ids_list)}')
    print(f'label_ids_list length: {len(label_ids_list)}')
    print(f'target_ids_list length: {len(target_ids_list)}')
    torch.save(data_dict, output_file)
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use key words to generate sentence.")
    parser.add_argument('--dataset', type=str, default='coco_clip', choices=['yelp_review', 'one-billion-words'])
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--dataset_size', type=int, default=-1,
                        help='specify the number of sentences to be used to create synthetic data.')
    parser.add_argument('--max_length', type=int, default=100, help='the maximum length of the generated sentence.')
    parser.add_argument('--min_length', type=int, default=6, help='the minimum length of the generated sentence.')
    parser.add_argument('--max_insert_label', type=int, default=1, help='the maximum number of tokens to be inserted before a token.')
    parser.add_argument('--insert_mode', type=int, default=0, choices=[0,1,2,3,4],
                        help='0 means using the left part, 1 means using the middle part, 2 means using the right part,'
                             '3 means randomly selecting, 4 means selecting the tokens with highest weight')
    parser.add_argument('--generate_mode', type=int, default=0, choices=[0, 1, 2],
                        help = '0 for random, 1 for lm, 2 for combination')
    parser.add_argument('--ratio', type=float, default=0,
                        help = 'ratio for the random generation mode')
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    tokenizer = BartTokenizer.from_pretrained('/media/xieyan/Hard Disk2/pretrain_model/bart-large')
    if args.generate_mode > 0:
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        print('Initialize BartForConditionalGeneration with default parameters.')
        model.eval()
        model.to('cuda')
    else:
        model = None

    # for mode in ['dev', 'train']:
    for mode in ['train', 'dev']:
        input_file = '../data/{}/{}.pt'.format(args.dataset, mode)
        print(f'''Loading data from {input_file}''')
        data_dict = torch.load(input_file)
        # each element is the length of [tokenizer.bos_token_id] + input_ids
        lengths = data_dict['lengths']
        for id, length in enumerate(lengths):
            if length < args.min_length:
                print(f'{id}--{length}')
        # each element is [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        input_ids_list = data_dict['input_ids_list']

        tf_idf_list = data_dict['tf_idf_list']
        if args.generate_mode == 0:
            generate_mode = ''
        elif args.generate_mode == 1:
            generate_mode = '_lm_generate'
        elif args.generate_mode == 2:
            generate_mode = '_combine_generate'
        else:
            raise ValueError('wrong generate mode.')
        output_file = '../data/{}/{}_synthetic{}_max_insert_label{}_insert_mode{}.pt'.format\
            (args.dataset, mode, generate_mode, args.max_insert_label, args.insert_mode)
        print('The output file is ', output_file)
        create_synthetic_data(output_file, input_ids_list, lengths, tf_idf_list, model, tokenizer, args)






