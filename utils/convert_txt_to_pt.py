# -*- coding: utf-8 -*-
# @Time    : 2020/7/23 11:32 AM
# @Author  : He Xingwei
import torch
from transformers import BartTokenizer
import glob
import numpy as np
import argparse
tokenizer = BartTokenizer.from_pretrained('/media/xieyan/Hard Disk2/pretrain_model/bart-large')

def tfidf(input_ids_list):
    tf_list = []
    idf = {}
    num_doc = len(input_ids_list)
    for input_ids in input_ids_list:
        tf = []
        d = {}
        for e in input_ids:
            idf[e] = idf.get(e, 0)+1.0
            d[e] = d.get(e, 0)+1.0
        s = sum(d.values())*1.0
        for e in input_ids:
            tf.append(d.get(e)/s)
        tf_list.append(tf)
    for k,v in idf.items():
        idf[k] = np.log(num_doc/(v+1.0))

    tf_idf_list = []
    # ans=0
    for i, input_ids in enumerate(input_ids_list):
        tf = tf_list[i]
        for j in range(len(tf)):
            token = input_ids[j]
            tf[j] *= idf[token]
        tf_idf_list.append(tf)
    return tf_idf_list
def convert(input_files, output_file):
    lengths = []
    input_ids_list = []
    for input_file in input_files:
        with open(input_file, 'r') as fr:
            for line in fr:
                line = line.strip()
                ids = tokenizer.encode(line, add_special_tokens=True)
                # print(tokenizer.convert_ids_to_tokens(ids))
                # print(line)
                lengths.append(len(ids))
                input_ids_list.append(ids)
    tf_idf_list = tfidf(input_ids_list)

    data_dict = {'lengths': lengths, 'input_ids_list': input_ids_list, 'tf_idf_list':tf_idf_list}
    # data_dict = {'lengths': lengths, 'input_ids_list': input_ids_list}
    torch.save(data_dict, output_file)
    print(input_files, len(lengths), output_file)
modes = ['dev','train']
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use key words to generate sentence.")
    parser.add_argument('--dataset', type=str, default='flickr30k', choices=['yelp_review', 'one-billion-words'])
    args = parser.parse_args()
    dataset = args.dataset
    for mode in modes:
        files = glob.glob('../data/{}/*{}*txt'.format(dataset, mode))
        output_file = f'../data/{dataset}/{mode}.pt'
        convert(files, output_file)
