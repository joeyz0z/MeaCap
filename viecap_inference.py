import clip
import torch
import argparse
from PIL import Image
from viecap.ClipCap import ClipCaptionModel
from transformers import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel
from sentence_transformers import SentenceTransformer
from viecap.utils import compose_discrete_prompts
from viecap.search import greedy_search, beam_search, opt_search
from utils.detect_utils import retrieve_concepts
from models.clip_utils import CLIP
import os
import json

@torch.no_grad()
def main(args) -> None:
    # initializing
    device = args.device
    clip_name = args.clip_model.replace('/', '') 
    clip_hidden_size = 640 if 'RN' in args.clip_model else 512

    # loading model
    tokenizer = AutoTokenizer.from_pretrained(args.language_model)
    model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, clip_hidden_size, gpt_type = args.language_model)
    model.load_state_dict(torch.load(args.weight_path, map_location = device), strict = False)
    model.to(device)
    encoder, preprocess = clip.load(args.clip_model, device = device)

    vl_model = CLIP(args.vl_model)
    vl_model = vl_model.to(device)
    print('Load CLIP from the checkpoint {}.'.format(args.clip_model))

    sim_func = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    wte_model = SentenceTransformer(args.wte_model_path)
    print('Load sentenceBERT from the checkpoint {}.'.format(args.wte_model_path))

    # parser model for memory concepts extracting
    parser_tokenizer = AutoTokenizer.from_pretrained(args.parser_checkpoint)
    parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.parser_checkpoint)
    parser_model.eval()
    parser_model.to(device)
    print('Load Textual Scene Graph parser from the checkpoint {}.'.format(args.parser_checkpoint))

    # prepare memory bank
    memory_id = args.memory_id
    memory_caption_path = os.path.join(f"data/memory/{memory_id}", "memory_captions.json")
    memory_clip_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_clip_embeddings.pt")
    memory_wte_embedding_file = os.path.join(f"data/memory/{memory_id}", "memory_wte_embeddings.pt")
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

    image = preprocess(Image.open(args.image_path)).unsqueeze(dim = 0).to(device)
    image_features = encoder.encode_image(image).float()
    image_features /= image_features.norm(2, dim = -1, keepdim = True)
    continuous_embeddings = model.mapping_network(image_features).view(-1, args.continuous_prompt_length, model.gpt_hidden_size)
    if args.using_hard_prompt:
        batch_image_embeds = vl_model.compute_image_representation_from_image_path(args.image_path)

        if retrieve_on_CPU != True:
            clip_score, clip_ref = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
                batch_image_embeds, memory_clip_embeddings)
        else:
            batch_image_embeds_cpu = batch_image_embeds.to(cpu_device)
            clip_score_cpu, clip_ref_cpu = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
                batch_image_embeds_cpu,
                memory_clip_embeddings)
            clip_score = clip_score_cpu.to(device)
            clip_ref = clip_ref_cpu.to(device)
        select_memory_ids = clip_score.topk(args.memory_caption_num, dim=-1)[1].squeeze(0)
        select_memory_captions = [memory_captions[id] for id in select_memory_ids]
        select_memory_wte_embeddings = memory_wte_embeddings[select_memory_ids]
        detected_objects = retrieve_concepts(parser_model=parser_model, parser_tokenizer=parser_tokenizer,
                                             wte_model=wte_model,
                                             select_memory_captions=select_memory_captions,
                                             image_embeds=batch_image_embeds,
                                             device=device)

        print("memory concepts:", detected_objects)
        discrete_tokens = compose_discrete_prompts(tokenizer, detected_objects).unsqueeze(dim = 0).to(args.device)

        discrete_embeddings = model.word_embed(discrete_tokens)
        if args.only_hard_prompt:
            embeddings = discrete_embeddings
        elif args.soft_prompt_first:
            embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim = 1)
        else:
            embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim = 1)
    else:
        embeddings = continuous_embeddings

    if 'gpt' in args.language_model:
        if not args.using_greedy_search:
            sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt) # List[str]
            sentence = sentence[0] # selected top 1
        else:
            sentence = greedy_search(embeddings = embeddings, tokenizer = tokenizer, model = model.gpt)
    else:
        sentence = opt_search(prompts=args.text_prompt, embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.gpt)
        sentence=sentence[0]
    
    print(f'the generated caption: {sentence}')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--clip_model', default = 'ViT-B/32')
    parser.add_argument('--language_model', default = 'openai-community/gpt2')
    parser.add_argument('--vl_model', type=str, default=r'openai/clip-vit-base-patch32')
    parser.add_argument("--parser_checkpoint", type=str, default=r'lizhuang144/flan-t5-base-VG-factual-sg')
    parser.add_argument("--wte_model_path", type=str, default=r'sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--continuous_prompt_length', type = int, default = 10)
    parser.add_argument('--clip_project_length', type = int, default = 10)
    parser.add_argument('--temperature', type = float, default = 0.01)
    parser.add_argument('--top_k', type = int, default = 3)
    parser.add_argument('--threshold', type = float, default = 0.2)
    parser.add_argument('--disable_all_entities', action = 'store_true', default = False, help = 'whether to use entities with a single word only')
    parser.add_argument('--name_of_entities_text', default = 'coco_entities', choices = ('visual_genome_entities', 'coco_entities', 'open_image_entities', 'vinvl_vg_entities', 'vinvl_vgoi_entities'))
    parser.add_argument('--prompt_ensemble', action = 'store_true', default = False)
    parser.add_argument('--weight_path', default = 'checkpoints/train_coco/coco_prefix-0014.pt')
    parser.add_argument('--image_path', default = 'image_example/COCO_val2014_000000027440.jpg')
    parser.add_argument('--using_hard_prompt', action = 'store_true', default = True)
    parser.add_argument('--soft_prompt_first', action = 'store_true', default = False)
    parser.add_argument('--only_hard_prompt', action = 'store_true', default = False)
    parser.add_argument('--using_greedy_search', action = 'store_true', default = False, help = 'greedy search or beam search')
    parser.add_argument('--beam_width', type = int, default = 5, help = 'width of beam')
    parser.add_argument('--text_prompt', type = str, default = None)
    parser.add_argument("--memory_id", type=str, default=r"coco",help="memory name")
    parser.add_argument("--memory_caption_path", type=str, default='data/memory/coco/memory_captions.json')
    parser.add_argument("--memory_caption_num", type=int, default=5)
    args = parser.parse_args()
    print('args: {}\n'.format(vars(args)))

    main(args)
