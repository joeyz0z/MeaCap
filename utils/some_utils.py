import torch
import numpy as np
import random
import os
from .log import Logger

def set_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


def update_args_logger(prefix, args):
    if args.random_init == 1:
        prefix = 'random_initialization_'+prefix
    prefix = f"cbart-{args.bart}_{prefix}"

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

    log_path = f'outputs/generate_keywords'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    output_path = f'outputs'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_file = '{}/{}.txt'.format(output_path, prefix)
    args.output_file = output_file
    args.log_path = log_path

    if args.conzic_sample:
        log_file = f'outputs/{args.dataset}_alpha{args.alpha}_beta{args.beta}.log'
    else:
        log_file = '{}/{}.log'.format(log_path, prefix)
    logger = Logger(log_file)
    logger.logger.info(f'The log file is {log_file}')
    logger.logger.info(f'output file is {args.output_file}')
    logger.logger.info(args)

    return args, logger

PROMPT_ENSEMBLING = [
        'Attention! There is',
        'Attention! There are',
        'There is',
        'There are',
        'A picture showing',
        'The picture shows',
        'A photo of',
        'An image of',
        'See! There is',
        'See! There are',
        'The image depicts',
        'The image depicts that']