import os
import math
import torch
import json
from collections import OrderedDict

from .parse_tool_new import parse, get_entitys, get_graph_dict, merge_sim_entities, merge_graph_dict_new


def add_prompt(word_list: list = None,
               prompt: str = 'Image of '):
    sentence_list = []
    for word in word_list:
        sentence = prompt + word + '.'
        sentence_list.append(sentence)
    return sentence_list


def detect_keyword(parser_model=None, parser_tokenizer=None, wte_model=None, image_embeds=None,vl_model=None, select_memory_captions=None,
                   device=None, logger=None):
    '''
    memory only keyword extracting
    '''
    torch.set_printoptions(sci_mode=False)
    # 读取memory中最相似的五句话中的concepts

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
    concepts, count_dict, filtered_graph_dict = merge_graph_dict_new(wte_model,vl_model, image_embeds, entities_, count_dict_, entire_graph_dict, select_memory_captions)
    # concepts, count_dict = merge_sim_entities(args.wte_model, entities_, count_dict_, attribute_dict)
    logger.logger.info(f"Memory concepts: {concepts}")
    logger.logger.info(f"********************************************")

    return concepts[:4]


def detect_keyword_clip_only(model, vision_name_list, batch_img_list, clip_model=None, parser_model=None, args=None, logger=None):

    # TODO: batch的方法还没有实现

    '''
    memory part code
    用memory中的概念与检测器的label比较，这样可以换一个更合适的keyword
    '''

    if args.use_memory:
        torch.set_printoptions(sci_mode=False)
        # 读取memory中最相似的五句话中的concepts
        with open(args.memory_caption_file, 'r') as f:
            memory_captions = json.load(f)
        memory_clip_embeddings = args.memory_clip_embeddings  # 训练集所有句子的embedding
        image_embedding = args.batch_image_embeds
        clip_score, clip_ref = clip_model.compute_image_text_similarity_via_embeddings(image_embedding, memory_clip_embeddings)
        select_memory_ids = clip_score.topk(1, dim=-1)[1].squeeze(0)  # 选出相似度最高的五句话
        select_memory_captions = [memory_captions[id] for id in select_memory_ids]  # 相似度最高的五句话
        logger.logger.info(f'Select caption: {select_memory_captions}')
        scene_graphs = parse(parser_model, args.parser_tokenizer,
                             text_input=select_memory_captions,
                             device=args.device)
        type_dict = {}
        concepts = get_graph_phrases(scene_graphs, type_dict)
        logger.logger.info(f"Memory concepts: {concepts}")

        concepts_clip_score, concepts_clip_ref = clip_model.compute_image_text_similarity_via_raw_text(image_embedding, concepts)
        concepts_clip_list = concepts_clip_score.squeeze(0).tolist()
        list1, list2 = zip(*sorted(zip(concepts_clip_list, concepts), reverse=True))
        return_list = list2[:3]

        keyword_list = [' '.join(list(set(return_list))).lower()]
        logger.logger.info(f"Final Keywords: {keyword_list}")
        return return_list


def detect_keyword_delete(model, vision_name_list, batch_img_list, clip_model=None, parser_model=None, args=None, logger=None):

    # TODO: batch的方法还没有实现

    # 检测
    vision_name = vision_name_list[0]
    vision_dir = args.img_path
    vision_file = os.path.join(vision_dir, vision_name)
    detect_res = model.predict(source=vision_file, save=True)
    category_dict = detect_res[0].names

    # 获得图像的中心点坐标
    (w, h) = detect_res[0].boxes.orig_shape
    x_c, y_c = w/2, h/2

    # 将结果按照距离中心点的远近排序
    object_list = []

    distance_list = []
    area_dict = {}  # 用一个字典表示面积的大小，这样方便累加
    for obj_id in range(len(detect_res[0])):

        # 将实体放进列表
        obj_index = int(detect_res[0].boxes.cls[obj_id])
        obj_name = category_dict[obj_index]
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
        distance = math.sqrt((obj_x-x_c)**2+(obj_y-y_c)**2)
        distance_list.append(distance)
    if len(object_list) == 0:
        return ['']
    else:
        distance_list, object_list_sorted = zip(*sorted(zip(distance_list, object_list)))  # TODO: 原先设计的规则中，object_list变成了object_list_sorted
        distance_list = list(distance_list)
        object_list_sorted = list(object_list_sorted)

    '''
    memory part code
    用memory中的概念与检测器的label比较，这样可以换一个更合适的keyword
    '''

    if args.use_memory:
        logger.logger.info(f"Origin Keywords: {object_list}")
        # 读取memory中最相似的五句话中的concepts
        with open(args.memory_caption_file, 'r') as f:
            memory_captions = json.load(f)
        memory_embedding = torch.load(args.memory_embedding_file)  # 训练集所有句子的embedding
        image_embedding = args.batch_image_embeds
        clip_score, clip_ref = clip_model.compute_image_text_similarity_via_embeddings(image_embedding, memory_embedding)
        select_memory_ids = clip_score.topk(5, dim=-1)[1].squeeze(0)  # 选出相似度最高的五句话
        select_memory_captions = [memory_captions[id] for id in select_memory_ids]  # 相似度最高的五句话
        scene_graphs = parse(parser_model, args.parser_tokenizer,
                             text_input=select_memory_captions,
                             device=args.device)
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

                croped_clip_score, croped_clip_ref = vl_model.compute_image_text_similarity_via_Image_text(croped_img, concepts)  # 与局部图片算相似度
                croped_clip_ref_sum += croped_clip_ref
            img_clip_score, img_clip_ref = vl_model.compute_image_text_similarity_via_Image_text(batch_img_list[0], concepts)  # 与整个图片算相似度
            img_clip_ref_avg = img_clip_ref
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
            sen_concepts = add_prompt(concepts)
            concepts_embedding = args.wte_model.encode(sen_concepts, convert_to_tensor=True)
            label_embedding = concepts_embedding[-1, :].unsqueeze(0)

            cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
            sim = cos_sim(label_embedding, concepts_embedding).unsqueeze(0)
            best_clip_single = 0.2
            best_label = None
            for id, sim_single in enumerate(sim[0, :-1]):
                # clip_ref_single = croped_clip_score_avg[:, id]  # 用局部图像作为重点
                clip_ref_single = img_clip_ref_avg[:, id]  # 用整个图像作为重点
                if sim_single > 0.65 and clip_ref_single > best_clip_single:
                    best_label = concepts[id]
                    best_clip_single = clip_ref_single
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
    # return keyword_list


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
        scene_graphs = parse(parser_model, args.parser_tokenizer,
                             text_input=select_memory_captions,
                             device=args.device)
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


def visual_detect(model, vision_name_list, batch_img_list, args=None, logger=None):

    # TODO: batch的方法还没有实现

    # 检测
    vision_name = vision_name_list[0]
    vision_dir = args.img_path
    vision_file = os.path.join(vision_dir, vision_name)
    detect_res = model.predict(source=vision_file, save=True)
    category_dict = detect_res[0].names

    # 获得图像的中心点坐标
    (w, h) = detect_res[0].boxes.orig_shape
    x_c, y_c = w/2, h/2

    # 将结果按照距离中心点的远近排序
    object_list = []

    distance_list = []
    area_dict = {}  # 用一个字典表示面积的大小，这样方便累加
    for obj_id in range(len(detect_res[0])):

        # 将实体放进列表
        obj_index = int(detect_res[0].boxes.cls[obj_id])
        obj_name = category_dict[obj_index]
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
        distance = math.sqrt((obj_x-x_c)**2+(obj_y-y_c)**2)
        distance_list.append(distance)
    logger.logger.info(f'Detected object :{object_list}')
    if len(object_list) == 0:
        return []
    croped_img_list = []
    bbox_list = []
    for id, object in enumerate(object_list):
        bbox = [float(detect_res[0].boxes.xyxy[id, i]) for i in range(4)]
        img = batch_img_list[0]
        croped_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        croped_img_list.append(croped_img)
        bbox_list.append(bbox)
    return croped_img_list, bbox_list

