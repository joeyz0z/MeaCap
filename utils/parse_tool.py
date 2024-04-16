from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
import torch
import nltk
from collections import OrderedDict
import numpy as np

NUMBER_DICT = {'2':"two","3":"three","4":"four","5":"five",'6':"six",'7':"seven","8":"eight","9":"nine"}

def merge_sim_node(entire_graph_dict, x, y):
    entire_graph_dict[x]["Relation"].update(entire_graph_dict[y]["Relation"])
    entire_graph_dict[x]["count"] += entire_graph_dict[y]["count"]
    for attr_key in list(entire_graph_dict[y]["Attribute"].keys()):
        if attr_key not in entire_graph_dict[x]["Attribute"]:
            entire_graph_dict[x]["Attribute"][attr_key] = entire_graph_dict[y]["Attribute"][attr_key]
        else:
            entire_graph_dict[x]["Attribute"][attr_key] += entire_graph_dict[y]["Attribute"][attr_key]

def filter_relation(graph_dict,sim_entity_dict ,remove_map, sentences, attribute_thresh=3):
    res_dict = {}
    nodes = list(graph_dict.keys())
    for node in nodes:
        pos_list = []
        for sentence in sentences:
            pos = sentence.find(node)/len(sentence)
            if pos > 0:
                pos_list.append(pos)
        final_pos = np.mean(pos_list) if pos_list else 1
        if node not in res_dict:
            res_dict[node] = {}
            res_dict[node]["rating"] = 0
        res_dict[node]["relative_pos"] = final_pos
        res_dict[node]["Attribute"] = graph_dict[node]["Attribute"]
        res_dict[node]["count"] = graph_dict[node]["count"]
        res_dict[node]["Relation"] = {}
        for obj in graph_dict[node]["Relation"]:
            if obj in nodes: #copy
                if obj in res_dict[node]["Relation"]:
                    res_dict[node]["Relation"][obj] += graph_dict[node]["Relation"][obj]
                else:
                    res_dict[node]["Relation"][obj] = graph_dict[node]["Relation"][obj]
                if obj not in res_dict:
                    res_dict[obj] = {}
                    res_dict[obj]["rating"] = 1
                else:
                    res_dict[obj]["rating"] += 1
                res_dict[node]["rating"] += 2
            elif obj in list(remove_map.keys()) and remove_map[obj] in nodes: # merge
                if remove_map[obj] in res_dict[node]["Relation"]:
                    res_dict[node]["Relation"][remove_map[obj]] += graph_dict[node]["Relation"][obj]
                else:
                    res_dict[node]["Relation"][remove_map[obj]] = graph_dict[node]["Relation"][obj]
                if remove_map[obj] not in res_dict:
                    res_dict[remove_map[obj]] = {}
                    res_dict[remove_map[obj]]["rating"] = 1
                else:
                    res_dict[remove_map[obj]]["rating"] += 1
                res_dict[node]["rating"] += 2
            else: # pass
                pass
        # res_dict[node]["rating"] += len(res_dict[node]["Relation"]) * 5

    # res_dict_sorted = OrderedDict(sorted(res_dict.items(), key=lambda item: item[1]["rating"], reverse=True))
    res_dict_sorted = OrderedDict(sorted(res_dict.items(), key=lambda item: item[1]["relative_pos"]))
    entities = []
    for entity in res_dict_sorted:
        flag = 0
        for attribute in res_dict_sorted[entity]["Attribute"]:
            if res_dict_sorted[entity]["Attribute"][attribute] >= attribute_thresh:
                entities.append(attribute +' '+ entity)
                flag = 1
                break
        if flag==0:
            entities.append(entity)
    # entities = list(res_dict_sorted.keys())

    return res_dict_sorted, entities




# def merge_sim_entities(model, entities, count_dict, attribute_dict):
#     entity_embeddings = model.encode(entities, convert_to_tensor=True, normalize_embeddings=True)
#     entity_correlation = torch.mm(entity_embeddings, entity_embeddings.T)
#     for idx in range(len(entity_correlation)):
#         entity_correlation[idx, idx] = 0
#     sim_index = torch.where(entity_correlation > 0.6)
#     sim_entity_dict = {}
#
#     remove_list = []
#     for ids, (x, y) in enumerate(zip(sim_index[0], sim_index[1])):
#         if entities[x] not in sim_entity_dict:
#             sim_entity_dict[entities[x]] = [entities[y]]
#         else:
#             sim_entity_dict[entities[x]].append(entities[y])
#         if entities[y] not in sim_entity_dict:
#             remove_list.append(entities[y])
#         count_dict[entities[x]] = count_dict[entities[x]] + count_dict[entities[y]]
#         if entities[y] in attribute_dict:
#             if entities[x] in attribute_dict:
#                 attribute_dict[entities[x]] = attribute_dict[entities[x]] + attribute_dict[entities[y]]
#             else:
#                 attribute_dict[entities[x]] = attribute_dict[entities[y]]
#     new_count_dict = OrderedDict()
#
#     for key in list(count_dict.keys()):
#         if key in remove_list or count_dict[key] <= 2:
#             continue
#         new_count_dict[key] = count_dict[key]
#     new_count_dict = OrderedDict(sorted(new_count_dict.items(), key=lambda item: item[1], reverse=True))
#     entities = list(new_count_dict.keys())
#
#     return entities, new_count_dict

def merge_graph_dict(model, entities, count_dict, entire_graph_dict, sentences):
    # compute similarity
    entity_embeddings = model.encode(entities, convert_to_tensor=True, normalize_embeddings=True)
    entity_correlation = torch.mm(entity_embeddings, entity_embeddings.T)
    for idx in range(len(entity_correlation)):
        entity_correlation[idx, idx] = 0
    sim_index = torch.where(entity_correlation > 0.55)  # TODO:xieyan
    sim_entity_dict = {}
    remove_entity_dict = {}
    remove_list = []
    for ids, (x, y) in enumerate(zip(sim_index[0], sim_index[1])):
        if entities[x] in remove_list:
            if entities[x] not in remove_entity_dict:
                remove_entity_dict[entities[x]] = [entities[y]]
            else:
                remove_entity_dict[entities[x]].append(entities[y])
        else:
            if entities[x] not in sim_entity_dict:
                sim_entity_dict[entities[x]] = [entities[y]]
            else:
                sim_entity_dict[entities[x]].append(entities[y])
            count_dict[entities[x]] = count_dict[entities[x]] + count_dict[entities[y]]
        if entities[y] not in sim_entity_dict:
            remove_list.append(entities[y])

        # if entities[y] in attribute_dict:
        #     if entities[x] in attribute_dict:
        #         attribute_dict[entities[x]] = attribute_dict[entities[x]] + attribute_dict[entities[y]]
        #     else:
        #         attribute_dict[entities[x]] = attribute_dict[entities[y]]
        merge_sim_node(entire_graph_dict, entities[x], entities[y])
    new_count_dict = OrderedDict()
    filterd_graph_dict = {}
    # update remove_list
    removed_map = {}
    remove_list = []
    for ent in sim_entity_dict:
        remove_list += sim_entity_dict[ent]
    for remove_wd in remove_list:
        try:
            removed_map[remove_wd] = [wd for wd in remove_entity_dict[remove_wd] if wd not in remove_list][0]
        except:
            print("remove wrong!")

    for key in list(count_dict.keys()):
        if key in remove_list or count_dict[key] <= 2:  # TODO: xieyan
            continue
        new_count_dict[key] = count_dict[key]
        filterd_graph_dict[key] = entire_graph_dict[key]
    if filterd_graph_dict:  # >1
        filterd_graph_dict_final, entities = filter_relation(filterd_graph_dict, sim_entity_dict, removed_map, sentences)
    else:
        # get the first one
        filterd_graph_dict_final = {}
        entities = []
        # key = next(iter(entire_graph_dict))
        # filterd_graph_dict_final[key] = entire_graph_dict[key]
        # entities = [key]

    new_count_dict = OrderedDict(sorted(new_count_dict.items(), key=lambda item: item[1], reverse=True))
    # entities = list(new_count_dict.keys())

    return entities, new_count_dict, filterd_graph_dict_final

def add_node_graph(scene_graph, subject, new_edge):
    # new_edge: (object, relation) or (attribute)
    if subject not in scene_graph:
        scene_graph[subject] = {
            "Relation":{},
            "Attribute":{},
            "count":1,
        }
        if len(new_edge)==2: # add relation
            scene_graph[subject]["Relation"][new_edge[0]] = [new_edge[1]]
        elif len(new_edge)==1: # add attribute
            scene_graph[subject]["Attribute"][new_edge[0]] = 1
        elif len(new_edge)==0: # only subject
            pass
        else:
            raise KeyError(f"{new_edge} is wrong")

    else:
        if len(new_edge)==2: # add relation
            if new_edge[0] not in scene_graph[subject]["Relation"]:
                scene_graph[subject]["Relation"][new_edge[0]] = [new_edge[1]]
            else:

                scene_graph[subject]["Relation"][new_edge[0]] += [new_edge[1]]
        elif len(new_edge) == 1:  # add attribute
            scene_graph[subject]["Attribute"][new_edge[0]] = 1
        elif len(new_edge) == 0:  # only subject
            pass
        else:
            raise KeyError(f"{new_edge} is wrong")
    return scene_graph

def merge_seperate_graph(scene_graph, new_graph):
    for key in list(new_graph.keys()):
        if key in scene_graph:
            scene_graph[key]["Relation"].update(new_graph[key]["Relation"])
            scene_graph[key]["count"]+= new_graph[key]["count"]
            for attr_key in list(new_graph[key]["Attribute"].keys()):
                if attr_key not in scene_graph[key]["Attribute"]:
                    scene_graph[key]["Attribute"][attr_key] = new_graph[key]["Attribute"][attr_key]
                else:
                    scene_graph[key]["Attribute"][attr_key] += new_graph[key]["Attribute"][attr_key]
        else:
            scene_graph[key] = new_graph[key]
    return scene_graph



def format_scene_graph(graph_str):
    return " ".join([item for item in graph_str.replace('(', ' ( ').replace(')', ' ) ').replace(',', ' , ').split() if item != ''])


def get_seg_list(graphs):
    if isinstance(graphs, str):
        seg_list = [scene_seg.replace('(', '').replace(')', '').strip() for scene_seg in format_scene_graph(graphs).split(') , (')]
    elif isinstance(graphs, list):
        seg_list = []
        for graph in graphs:
            seg_list.extend([scene_seg.replace('(', '').replace(')', '').strip() for scene_seg in format_scene_graph(graph).split(') , (')])
    else:
        raise ValueError('input should be either a string or a list of strings')
    return list(set(seg_list))

def get_seg_list_seperate(graphs):
    if isinstance(graphs, str):
        seg_list = [scene_seg.replace('(', '').replace(')', '').strip() for scene_seg in format_scene_graph(graphs).split(') , (')]
    elif isinstance(graphs, list):
        seg_list = []
        for graph in graphs:
            cur_list = []
            cur_list.extend([scene_seg.replace('(', '').replace(')', '').strip() for scene_seg in format_scene_graph(graph).split(') , (')])
            seg_list.append(cur_list)
    else:
        raise ValueError('input should be either a string or a list of strings')
    return list(seg_list)


def parse(parser, parser_tokenizer, text_input,
          max_input_length=128, max_output_length=128, beam_size=1, device="cuda:0"):
    '''
    :param text_input: one or a list of textual image descriptions
    :return: corresponding scene graphs of the input descriptions
    '''

    if isinstance(text_input, str):
        text_input = [text_input]

    # breakpoint()
    text_input = ['Generate Scene Graph: ' + text for text in text_input]
    with torch.no_grad():
        encoded_text = parser_tokenizer(
            text_input,
            max_length=max_input_length,
            truncation=True,
            padding=True,
            return_tensors='pt'
        )
        text_tokens = encoded_text['input_ids'].to(device)
        text_mask = encoded_text['attention_mask'].to(device)

        generated_ids = parser.generate(
            text_tokens,
            attention_mask=text_mask,
            use_cache=True,
            decoder_start_token_id=parser_tokenizer.pad_token_id,
            num_beams=beam_size,
            max_length=max_output_length,
            early_stopping=True
        )

        # output to text
        output_text = parser_tokenizer.batch_decode(generated_ids, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=True)
        output_text = [format_scene_graph(text.replace('Generate Scene Graph:', '').strip()) for text in output_text]
        return output_text


def get_graph_phrases(graph_str_list, type_dict):
    seg_list = get_seg_list(graph_str_list)
    #breakpoint()
    new_pairs = []
    for seg in seg_list:
        new_seg = [item.strip() for item in seg.split(',')]
        try:
            if len(new_seg) == 1 and len(seg_list) == 1:
                new_pairs.append(new_seg[0])
                type_dict[new_seg[0]] = "object"
                continue

            if len(new_seg) == 2:
                new_pairs.append(new_seg[1] + " " + new_seg[0])
                type_dict[new_seg[1] + " " + new_seg[0]] = "attribute"
                new_pairs.append(new_seg[0])
                type_dict[new_seg[0]] = "object"
                continue
            elif len(new_seg) == 3:
                sentence = new_seg[0] + " " + new_seg[1] + " " + new_seg[2]
                sentence_word = nltk.word_tokenize(sentence)
                pos_type = nltk.pos_tag(sentence_word)
                if new_seg[1] == 'is' and pos_type[-1][1] == 'JJ':
                    new_pairs.append(new_seg[2] + " " + new_seg[0])
                    type_dict[new_seg[2] + " " + new_seg[0]] = "attribute"
                    new_pairs.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                else:
                    # new_pairs.append(new_seg[0] + " " + new_seg[1] + " " + new_seg[2])
                    type_dict[new_seg[0] + " " + new_seg[1] + " " + new_seg[2]] = "fact"
                    new_pairs.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    if new_seg[1] == 'is':
                        continue
                    else:
                        new_pairs.append(new_seg[2])
                        type_dict[new_seg[2]] = "object"
            elif len(new_seg) > 3:
                # new_pairs.append(new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1])
                type_dict[new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1]] = "fact"
                new_pairs.append(new_seg[0])
                type_dict[new_seg[0]] = "object"
                new_pairs.append(new_seg[-1])
                type_dict[new_seg[-1]] = "object"
        except IndexError:
            print(seg_list)
            continue

    return list(set(new_pairs))

def get_graph_dict(model, graph_str_list,type_dict, attribute_dict):
    seg_lists = get_seg_list_seperate(graph_str_list)
    count_dict = OrderedDict()
    total_entity_lists = []
    total_graph_dicts = []
    # process graphs
    for seg_list in seg_lists:
        #breakpoint()
        entity_list = []
        cur_sg = dict()
        for seg in seg_list:
            new_seg = [item.strip() for item in seg.split(',')]
            try:
                if len(new_seg) == 1 and len(seg_list) == 1:
                    entity_list.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    add_node_graph(cur_sg, new_seg[0], [])
                    continue

                if len(new_seg) == 2:
                    # entity_list.append(new_seg[1] + " " + new_seg[0])
                    type_dict[new_seg[1] + " " + new_seg[0]] = "attribute"
                    entity_list.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    if new_seg[0] not in attribute_dict:
                        attribute_dict[new_seg[0]] = [new_seg[1]]
                    else:
                        attribute_dict[new_seg[0]].append(new_seg[1])
                    add_node_graph(cur_sg, new_seg[0], [new_seg[1]])
                    continue
                elif len(new_seg) == 3:
                    if new_seg[2] in list(NUMBER_DICT.keys()):
                        new_seg[2] = NUMBER_DICT[new_seg[2]]
                    sentence = new_seg[0] + " " + new_seg[1] + " " + new_seg[2]
                    # sentence_word = nltk.word_tokenize(sentence)
                    # pos_type = nltk.pos_tag(sentence_word)
                    if new_seg[1] == 'is':
                        # entity_list.append(new_seg[2] + " " + new_seg[0])
                        type_dict[new_seg[2] + " " + new_seg[0]] = "attribute"
                        entity_list.append(new_seg[0])
                        type_dict[new_seg[0]] = "object"
                        if new_seg[0] not in attribute_dict:
                            attribute_dict[new_seg[0]] = [new_seg[2]]
                        else:
                            attribute_dict[new_seg[0]].append(new_seg[2])
                        add_node_graph(cur_sg, new_seg[0], [new_seg[2]])
                    else:
                        # entity_list.append(new_seg[0] + " " + new_seg[1] + " " + new_seg[2])
                        type_dict[new_seg[0] + " " + new_seg[1] + " " + new_seg[2]] = "fact"
                        entity_list.append(new_seg[0])
                        type_dict[new_seg[0]] = "object"
                        if new_seg[1] == 'is':
                            continue
                        else:
                            entity_list.append(new_seg[2])
                            type_dict[new_seg[2]] = "object"
                            add_node_graph(cur_sg, new_seg[0], [new_seg[2],new_seg[1]])
                            add_node_graph(cur_sg, new_seg[2], [])
                elif len(new_seg) > 3:
                    # entity_list.append(new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1])
                    type_dict[new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1]] = "fact"
                    entity_list.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    entity_list.append(new_seg[-1])
                    type_dict[new_seg[-1]] = "object"
                    add_node_graph(cur_sg, new_seg[0], [new_seg[-1], new_seg[1:-1]])
                    add_node_graph(cur_sg, new_seg[-1], [])
            except IndexError:
                print(seg_list)
                continue
        entity_list = list(set(entity_list))
        for entity in entity_list:
            if entity not in count_dict:
                count_dict[entity] = 1
            else:
                count_dict[entity] += 1
        total_entity_lists.append(entity_list)
        total_graph_dicts.append(cur_sg)
    sorted_count_dict = OrderedDict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))
    entitys = list(sorted_count_dict.keys())
    entire_graph_dict = {}
    for graph_dict in total_graph_dicts:
        merge_seperate_graph(entire_graph_dict, graph_dict)


    return entitys, sorted_count_dict, entire_graph_dict


def get_entitys(graph_str_list,type_dict, attribute_dict):
    seg_lists = get_seg_list_seperate(graph_str_list)
    count_dict = OrderedDict()
    total_entity_lists = []
    for seg_list in seg_lists:
        #breakpoint()
        entity_list = []
        for seg in seg_list:
            new_seg = [item.strip() for item in seg.split(',')]
            try:
                if len(new_seg) == 1 and len(seg_list) == 1:
                    entity_list.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    continue

                if len(new_seg) == 2:
                    # entity_list.append(new_seg[1] + " " + new_seg[0])
                    type_dict[new_seg[1] + " " + new_seg[0]] = "attribute"
                    entity_list.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    if new_seg[0] not in attribute_dict:
                        attribute_dict[new_seg[0]] = [new_seg[1]]
                    else:
                        attribute_dict[new_seg[0]].append(new_seg[1])
                    continue
                elif len(new_seg) == 3:
                    if new_seg[2] in list(NUMBER_DICT.keys()):
                        new_seg[2] = NUMBER_DICT[new_seg[2]]
                    sentence = new_seg[0] + " " + new_seg[1] + " " + new_seg[2]
                    # sentence_word = nltk.word_tokenize(sentence)
                    # pos_type = nltk.pos_tag(sentence_word)
                    if new_seg[1] == 'is':
                        # entity_list.append(new_seg[2] + " " + new_seg[0])
                        type_dict[new_seg[2] + " " + new_seg[0]] = "attribute"
                        entity_list.append(new_seg[0])
                        type_dict[new_seg[0]] = "object"
                        if new_seg[0] not in attribute_dict:
                            attribute_dict[new_seg[0]] = [new_seg[2]]
                        else:
                            attribute_dict[new_seg[0]].append(new_seg[2])
                    else:
                        # entity_list.append(new_seg[0] + " " + new_seg[1] + " " + new_seg[2])
                        type_dict[new_seg[0] + " " + new_seg[1] + " " + new_seg[2]] = "fact"
                        entity_list.append(new_seg[0])
                        type_dict[new_seg[0]] = "object"
                        if new_seg[1] == 'is':
                            continue
                        else:
                            entity_list.append(new_seg[2])
                            type_dict[new_seg[2]] = "object"
                elif len(new_seg) > 3:
                    # entity_list.append(new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1])
                    type_dict[new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1]] = "fact"
                    entity_list.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    entity_list.append(new_seg[-1])
                    type_dict[new_seg[-1]] = "object"
            except IndexError:
                print(seg_list)
                continue
        entity_list = list(set(entity_list))
        for entity in entity_list:
            if entity not in count_dict:
                count_dict[entity] = 1
            else:
                count_dict[entity] += 1
        total_entity_lists.append(entity_list)
    sorted_count_dict = OrderedDict(sorted(count_dict.items(), key=lambda item: item[1], reverse=True))

    entitys = list(sorted_count_dict.keys())

    return entitys, sorted_count_dict

def get_graph_phrases_new(graph_str_list, type_dict, count_dict):
    seg_lists = get_seg_list_seperate(graph_str_list)


    total_pairs = []
    for seg_list in seg_lists:
        #breakpoint()
        new_pairs = []
        for seg in seg_list:
            new_seg = [item.strip() for item in seg.split(',')]
            try:
                if len(new_seg) == 1 and len(seg_list) == 1:
                    new_pairs.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    if new_seg[0] not in count_dict:
                        count_dict[new_seg[0]] = 1
                    else:
                        count_dict[new_seg[0]] += 1
                    continue

                if len(new_seg) == 2:
                    new_pairs.append(new_seg[1] + " " + new_seg[0])
                    type_dict[new_seg[1] + " " + new_seg[0]] = "attribute"
                    new_pairs.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    if new_seg[0] not in count_dict:
                        count_dict[new_seg[0]] = 1
                    else:
                        count_dict[new_seg[0]] += 1
                    continue
                elif len(new_seg) == 3:
                    sentence = new_seg[0] + " " + new_seg[1] + " " + new_seg[2]
                    sentence_word = nltk.word_tokenize(sentence)
                    pos_type = nltk.pos_tag(sentence_word)
                    if new_seg[1] == 'is' and pos_type[-1][1] == 'JJ':
                        new_pairs.append(new_seg[2] + " " + new_seg[0])
                        type_dict[new_seg[2] + " " + new_seg[0]] = "attribute"
                        new_pairs.append(new_seg[0])
                        type_dict[new_seg[0]] = "object"
                        if new_seg[0] not in count_dict:
                            count_dict[new_seg[0]] = 1
                        else:
                            count_dict[new_seg[0]] += 1
                    else:
                        # new_pairs.append(new_seg[0] + " " + new_seg[1] + " " + new_seg[2])
                        type_dict[new_seg[0] + " " + new_seg[1] + " " + new_seg[2]] = "fact"
                        new_pairs.append(new_seg[0])
                        type_dict[new_seg[0]] = "object"
                        if new_seg[0] not in count_dict:
                            count_dict[new_seg[0]] = 1
                        else:
                            count_dict[new_seg[0]] += 1
                        if new_seg[1] == 'is':
                            continue
                        else:
                            new_pairs.append(new_seg[2])
                            type_dict[new_seg[2]] = "object"
                            if new_seg[2] not in count_dict:
                                count_dict[new_seg[2]] = 1
                            else:
                                count_dict[new_seg[2]] += 1
                elif len(new_seg) > 3:
                    # new_pairs.append(new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1])
                    type_dict[new_seg[0] + " ".join(new_seg[1:-1]) + new_seg[-1]] = "fact"
                    new_pairs.append(new_seg[0])
                    type_dict[new_seg[0]] = "object"
                    if new_seg[0] not in count_dict:
                        count_dict[new_seg[0]] = 1
                    else:
                        count_dict[new_seg[0]] += 1
                    new_pairs.append(new_seg[-1])
                    type_dict[new_seg[-1]] = "object"
                    if new_seg[0] not in count_dict:
                        count_dict[new_seg[-1]] = 1
                    else:
                        count_dict[new_seg[-1]] += 1
            except IndexError:
                print(seg_list)
                continue
        total_pairs.append(new_pairs)

    all_pairs = [pair for pairs in total_pairs for pair in pairs]

    return list(set(all_pairs))


if __name__ == "__main__":
    device = "cuda"
    parser_checkpoint = "/media/xieyan/Hard Disk2/pretrain_model/flan-t5-base-VG-factual-sg"
    parser_tokenizer = AutoTokenizer.from_pretrained(parser_checkpoint)
    parser = AutoModelForSeq2SeqLM.from_pretrained(parser_checkpoint)

    parser.eval()
    parser.to(device)
    scene_graphs = parse(parser, parser_tokenizer,
                         ["A young girl inhales with the intent of blowing out a candle.",
            "A young girl is preparing to blow out her candle.",
            "A kid is to blow out the single candle in a bowl of birthday goodness.",
            "Girl blowing out the candle on an ice-cream",
            "A little girl is getting ready to blow out a candle on a small dessert."],
                                        device=device)

    # scene_graphs = parse(parser, parser_tokenizer,
    #                      ["People talk to each other."],
    #                      device=device)
    type_dict = {}
    concepts = get_graph_phrases(scene_graphs, type_dict)
    print(concepts)
