import torch
import numpy as np

from collections import OrderedDict

def merge_sim_node(entire_graph_dict, x, y):
    entire_graph_dict[x]["Relation"].update(entire_graph_dict[y]["Relation"])
    entire_graph_dict[x]["count"] += entire_graph_dict[y]["count"]
    for attr_key in list(entire_graph_dict[y]["Attribute"].keys()):
        if attr_key not in entire_graph_dict[x]["Attribute"]:
            entire_graph_dict[x]["Attribute"][attr_key] = entire_graph_dict[y]["Attribute"][attr_key]
        else:
            entire_graph_dict[x]["Attribute"][attr_key] += entire_graph_dict[y]["Attribute"][attr_key]

def filter_relation(graph_dict,sim_entity_dict ,remove_map, sentences):
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


    entities = list(res_dict_sorted.keys())

    return res_dict_sorted, entities




def merge_sim_entities(model, entities, count_dict, attribute_dict):
    entity_embeddings = model.encode(entities, convert_to_tensor=True, normalize_embeddings=True)
    entity_correlation = torch.mm(entity_embeddings, entity_embeddings.T)
    for idx in range(len(entity_correlation)):
        entity_correlation[idx, idx] = 0
    sim_index = torch.where(entity_correlation > 0.6)
    sim_entity_dict = {}

    remove_list = []
    for ids, (x, y) in enumerate(zip(sim_index[0], sim_index[1])):
        if entities[x] not in sim_entity_dict:
            sim_entity_dict[entities[x]] = [entities[y]]
        else:
            sim_entity_dict[entities[x]].append(entities[y])
        if entities[y] not in sim_entity_dict:
            remove_list.append(entities[y])
        count_dict[entities[x]] = count_dict[entities[x]] + count_dict[entities[y]]
        if entities[y] in attribute_dict:
            if entities[x] in attribute_dict:
                attribute_dict[entities[x]] = attribute_dict[entities[x]] + attribute_dict[entities[y]]
            else:
                attribute_dict[entities[x]] = attribute_dict[entities[y]]
    new_count_dict = OrderedDict()

    for key in list(count_dict.keys()):
        if key in remove_list or count_dict[key] <= 2:
            continue
        new_count_dict[key] = count_dict[key]
    new_count_dict = OrderedDict(sorted(new_count_dict.items(), key=lambda item: item[1], reverse=True))
    entities = list(new_count_dict.keys())

    return entities, new_count_dict

def merge_graph_dict(model, entities, count_dict, entire_graph_dict, sentences):
    # compute similarity
    entity_embeddings = model.encode(entities, convert_to_tensor=True, normalize_embeddings=True)
    entity_correlation = torch.mm(entity_embeddings, entity_embeddings.T)
    for idx in range(len(entity_correlation)):
        entity_correlation[idx, idx] = 0
    sim_index = torch.where(entity_correlation > 0.6)
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
        remove_list +=  sim_entity_dict[ent]
    for remove_wd in remove_list:
        try:
            removed_map[remove_wd] = [wd for wd in remove_entity_dict[remove_wd] if wd not in remove_list][0]
        except:
            print("remove wrong!")

    for key in list(count_dict.keys()):
        if key in remove_list or count_dict[key] <= 2:
            continue
        new_count_dict[key] = count_dict[key]
        filterd_graph_dict[key] = entire_graph_dict[key]
    if filterd_graph_dict: # >1
        filterd_graph_dict_final, entities = filter_relation(filterd_graph_dict,sim_entity_dict ,removed_map, sentences)
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