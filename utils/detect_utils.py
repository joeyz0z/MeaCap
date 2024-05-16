import os
import math
import torch
import json
from collections import OrderedDict
import traceback

from .parse_tool import parse, get_entitys, get_graph_dict, merge_graph_dict


def add_prompt(word_list: list = None,
               prompt: str = 'Image of '):
    sentence_list = []
    for word in word_list:
        sentence = prompt + word + '.'
        sentence_list.append(sentence)
    return sentence_list


def retrieve_concepts(parser_model=None, parser_tokenizer=None, wte_model=None, select_memory_captions=None,image_embeds=None,
                   device=None, logger=None, args=None,verbose=False):
    '''
    memory-based key concepts extracting
    '''
    torch.set_printoptions(sci_mode=False)

    scene_graphs = parse(parser_model, parser_tokenizer,
                         text_input=select_memory_captions,
                         device=device)
    type_dict = {}
    count_dict = OrderedDict()
    attribute_dict = {}
    entities_, count_dict_, entire_graph_dict = get_graph_dict(wte_model, scene_graphs, type_dict, attribute_dict)
    concepts, count_dict, filtered_graph_dict = merge_graph_dict(wte_model, entities_, count_dict_, entire_graph_dict, select_memory_captions)
    # concepts, count_dict = merge_sim_entities(args.wte_model, entities_, count_dict_, attribute_dict)
    if logger is not None:
        logger.logger.info(f"********************************************")
        logger.logger.info(f"Memory captions: {select_memory_captions}")
        logger.logger.info(f"Memory scene graphs: {scene_graphs}")
        logger.logger.info(f"Memory concepts: {concepts}")
        logger.logger.info(f"********************************************")

    return concepts[:4]

def retrieve_concepts_from_image(parser_model=None, parser_tokenizer=None, wte_model=None, select_memory_captions=None,image_path=None,
                   device=None, logger=None, args=None):
    '''
    memory-based key concepts extracting
    '''


    torch.set_printoptions(sci_mode=False)
    logger.logger.info(f"********************************************")
    logger.logger.info(f"Memory captions: {select_memory_captions}")
    scene_graphs = parse(parser_model, parser_tokenizer,
                         text_input=select_memory_captions,
                         device=device)
    logger.logger.info(f"Memory scene graphs: {scene_graphs}")
    type_dict = {}
    count_dict = OrderedDict()
    attribute_dict = {}
    entities_, count_dict_, entire_graph_dict = get_graph_dict(wte_model, scene_graphs, type_dict, attribute_dict)
    concepts, count_dict, filtered_graph_dict = merge_graph_dict(wte_model, entities_, count_dict_, entire_graph_dict, select_memory_captions)
    # concepts, count_dict = merge_sim_entities(args.wte_model, entities_, count_dict_, attribute_dict)

    logger.logger.info(f"Memory concepts: {concepts}")
    logger.logger.info(f"********************************************")

    return concepts[:4]

