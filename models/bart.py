# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 10:17 AM
# @Author  : He Xingwei
"""
this script is used to train the CBART model on the synthetic data.
CBART consists of an encoder and a decoder.
The encoder is used to predict labels for each token.
The decoder is used to predict original input.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import time
import os
import sys
from pympler import asizeof
import argparse
# sys.path.append('../')

# python 3.8以上不可以使用sys.path.append来添加搜索路径
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

from utils.log import Logger
from src.transformers import BartForTextInfill,BartTokenizer, AdamW, BartConfig
def safe_check(a,type='uint8'):
    d= { 'uint8': [0,255],
        'uint16': [0,65535]
         }
    range = d[type]
    for l in a:
        for e in l:
            assert e>=range[0] and e<=range[1]

class BARTDataset(Dataset):
    def __init__(self, dataset, mode, tokenizer=None, num_labels=-1, insert_mode=-1, max_sentence_length=40,
                 encoder_loss_type=0, statistics = False,local_rank=-2, generate_mode=0,
                 ratio1=0.5, ratio2=0.5):
        self.encoder_loss_type = encoder_loss_type
        assert mode in ["train", "test", 'dev']
        self.mode = mode
        if self.mode=='test' or self.mode=='dev':
            self.is_train = False
        else:
            self.is_train = True
        self.tokenizer = tokenizer
        self.max_sentence_length = max_sentence_length + 2 # the bos and eos tokens
        self.encoder_inputs = []
        self.encoder_labels = []
        self.decoder_labels = []

        data_dict_paths = []
        if generate_mode in [0,2]:
            _generate_mode = ''
            data_dict_path = '../data/{}/{}_synthetic{}_max_insert_label{}_insert_mode{}.pt'.format(dataset, mode,_generate_mode,num_labels-2, insert_mode)
            data_dict_paths.append(data_dict_path)
        if generate_mode in [1,2]:
            _generate_mode = '_lm_generate'
            data_dict_path = '../data/{}/{}_synthetic{}_max_insert_label{}_insert_mode{}.pt'.format(dataset, mode,_generate_mode,num_labels-2, insert_mode)
            data_dict_paths.append(data_dict_path)

        for r, data_dict_path in zip([ratio1,ratio2], data_dict_paths):
            if os.path.exists(data_dict_path):
                print(f'''Loading data from {data_dict_path}''')
                # data_dict = pickle.load(open(data_dict_path, 'rb'))
                data_dict = torch.load(data_dict_path)
                if generate_mode == 2:
                    length = int(len(data_dict['incorrect_input_ids_list'])*r)
                    self.encoder_inputs += data_dict['incorrect_input_ids_list'][:length]
                    self.encoder_labels += data_dict['label_ids_list'][:length]
                    self.decoder_labels += data_dict['target_ids_list'][:length]
                else:
                    self.encoder_inputs += data_dict['incorrect_input_ids_list']
                    self.encoder_labels += data_dict['label_ids_list']
                    self.decoder_labels += data_dict['target_ids_list']
            else:
                print(f'Please create the synthetic datafile {data_dict_path} with create_synthetic_data.py.')

        self.len = len(self.encoder_inputs)
        if statistics and local_rank in [-1, 0]:
            print('Statistics for sentence length:')
            lengths = [len(e) for e in self.decoder_labels]
            (unique, counts) = np.unique(lengths, return_counts=True)
            for k, v in zip(unique,counts):
                print(f'sentence length{k}: {v}')
            print('Statistics for sentence labels:')

            labels = [e for s in self.encoder_labels for e in s]
            (unique, counts) = np.unique(labels, return_counts=True)
            for k, v in zip(unique,counts):
                print(f'Label {k}: {v}')


    def __getitem__(self, idx):
        return torch.tensor(self.encoder_inputs[idx], dtype=torch.long), \
               torch.tensor(self.encoder_labels[idx], dtype=torch.long ), \
               torch.tensor(self.decoder_labels[idx], dtype=torch.long )

    def __len__(self):
        return self.len

    def create_mini_batch(self, samples):
        encoder_inputs = [s[0] for s in samples]
        encoder_labels = [s[1] for s in samples]
        decoder_labels = [s[2] for s in samples]

        # Mask to avoid performing attention on padding token indices in encoder_inputs.
        _mask = pad_sequence(encoder_inputs, batch_first=True, padding_value=-100)
        attention_mask = torch.zeros(_mask.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(_mask != -100, 1)

        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        encoder_labels = pad_sequence(encoder_labels, batch_first=True, padding_value=-100)
        if self.encoder_loss_type==1: # labels for mse loss
            encoder_labels=encoder_labels.float()

        decoder_labels = pad_sequence(decoder_labels, batch_first=True, padding_value=-100)
        # avoid computing loss on the first token, i.e. bos_token
        decoder_labels[:,0] = -100

        # this method is for non-autoregressive decoding.
        decoder_inputs = [self.create_decoder_inputs(s[0], s[1], tokenizer.mask_token_id) for s in samples]

        # replace the eos_token_id with pad_token_id
        for i, _ in enumerate(decoder_inputs):
            decoder_inputs[i][-1] = self.tokenizer.pad_token_id

        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        # create decoder_inputs by shifting the decoder_labels right,
        _tmp = decoder_inputs.clone()
        decoder_inputs[:, 1:] = _tmp[:, :-1]
        decoder_inputs[:, 0] = self.tokenizer.eos_token_id

        # construct labels for masked lm loss
        masked_lm_labels = decoder_labels.clone()
        masked_lm_labels[_tmp!=tokenizer.mask_token_id] = -100

        return encoder_inputs, encoder_labels, decoder_inputs, decoder_labels, masked_lm_labels, attention_mask

    @staticmethod
    def create_decoder_inputs(encoder_inputs, encoder_labels, mask_token_id):
        """
        :param encoder_inputs: list, each element is an int
        :param encoder_labels: list, each element is an int
        :return:
        """
        decoder_inputs = []
        for i, l in zip(encoder_inputs, encoder_labels):
            if l == 0:
                decoder_inputs.append(i)
            elif l==1:
                decoder_inputs.append(mask_token_id)
            else:
                decoder_inputs += [mask_token_id]*(l-1)
                decoder_inputs.append(i)
        return torch.tensor(decoder_inputs,dtype =torch.long)

    @staticmethod
    def compute_accuracy(model, mode, local_rank, dataloader, datasize, device, num_labels,encoder_loss_type,merge_insert = False,):
        """
        compute negative log-likelihood on dataloader with model.
        :param model:
        :param dataloader:
        :return:
        """
        if local_rank in [-1, 0]:
            print()
        model.eval()
        correct = {}
        recalls = {}
        precisions = {}
        f1s = {}
        for i in range(num_labels):
            recalls[i] = 0.0
            precisions[i] = 0.0
            correct[i] = 0.0
        total_encoder_loss = 0
        total_decoder_loss = 0
        total_masked_decoder_loss = 0
        with torch.no_grad():
            start = time.time()
            step = 0
            for data in dataloader:
                data = [t.to(device) for t in data]
                encoder_inputs, encoder_labels, decoder_inputs, decoder_labels, masked_lm_labels, attention_mask = data
                encoder_loss, decoder_loss, encoder_logits, logits = model(encoder_inputs, encoder_labels=encoder_labels,
                                                           decoder_input_ids=decoder_inputs, labels=decoder_labels,
                                                           attention_mask=attention_mask)[:4]
                bts = encoder_inputs.shape[0]
                total_encoder_loss += encoder_loss*bts
                total_decoder_loss += decoder_loss*bts

                loss_fct = torch.nn.CrossEntropyLoss()
                # only compute labels for mask tokens
                total_masked_decoder_loss += loss_fct(logits.view(-1, logits.shape[-1]), masked_lm_labels.view(-1))*bts

                # compute accuracy
                if encoder_loss_type == 0:  # classification
                    # argmax
                    predict_label = torch.argmax(encoder_logits, dim=-1, keepdim=False)
                else:  # regression, round and convert the output into torch.Long tensor
                    predict_label = torch.round(encoder_logits).long()

                if merge_insert:
                    predict_label[predict_label > 2] = 2
                    encoder_labels[encoder_labels > 2] = 2

                for i in range(num_labels):
                    correct[i] += ((predict_label == i) & (encoder_labels == i)).sum()
                    recalls[i] += (encoder_labels == i).sum()
                    precisions[i] += ((predict_label == i) & (encoder_labels != -100)).sum()

                step+=bts
                if local_rank in [-1, 0]:
                    print(f'\r{mode} set {step}/{datasize/torch.cuda.device_count()}, time: {time.time()-start:.1f} seconds.',end='')
                # if step>=100:
                    # break
            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce_multigpu([total_encoder_loss])
                torch.distributed.all_reduce_multigpu([total_decoder_loss])
                torch.distributed.all_reduce_multigpu([total_masked_decoder_loss])
            total_encoder_loss = total_encoder_loss.item()
            total_decoder_loss = total_decoder_loss.item()
            total_masked_decoder_loss = total_masked_decoder_loss.item()

            total_loss = total_encoder_loss + total_decoder_loss
            average_encoder_loss = total_encoder_loss / datasize
            average_decoder_loss = total_decoder_loss / datasize
            average_masked_decoder_loss = total_masked_decoder_loss/datasize
            average_loss = total_loss / datasize

            # merge results
            for i in range(num_labels):
                if torch.cuda.device_count()>1:
                    torch.distributed.all_reduce_multigpu([correct[i]])
                    torch.distributed.all_reduce_multigpu([recalls[i]])
                    torch.distributed.all_reduce_multigpu([precisions[i]])
                correct[i] = correct[i].item()
                recalls[i] = recalls[i].item()
                precisions[i] = precisions[i].item()

            for i in range(num_labels):
                if recalls[i]!=0:
                    recalls[i] = correct[i]/recalls[i]
                else:
                    recalls[i] = 0

                if precisions[i]!=0:
                    precisions[i] = correct[i]/precisions[i]
                else:
                    precisions[i] = 0

                if precisions[i]!=0:
                    f1s[i] = 2*recalls[i]*precisions[i]/(recalls[i]+precisions[i])
                else:
                    f1s[i] = 0

            used_time = time.time() - start
        model.train()
        return average_encoder_loss, average_decoder_loss, average_masked_decoder_loss, average_loss, \
               used_time, recalls, precisions, f1s



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text infilling.")
    parser.add_argument('--bart', type=str, default='large', choices=['base', 'large'])
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--num_labels', type=int, default=3,
                        help='0 for copy, 1 for replace, 2-5 means insert 1-4 tokens')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--random_init', type=int, default=0, help='0 denotes initialization with BART; '
                                                                  '1 denotes random initialization;')

    parser.add_argument('--w', type=float, default=1.0, help='The weight for the encoder loss')

    parser.add_argument('--masked_lm', type=float, default=0, help='0 for using language modeling for the decoder,'
                                                                   '1 for using mask language modeling for the decoder.')

    parser.add_argument('--full_mask', type=float, default=0, help='0 for using casual mask attention for decoder, '
                                                                    '1 for without using casual mask attention for decoder.')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--train', type=int, default=1)
    parser.add_argument('--encoder_loss_type', type=int, default=0,help='0 is classification loss, 1 is regression loss')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--dataset', type=str, default='coco_clip', choices=['yelp_review','one-billion-words'])
    parser.add_argument('--insert_mode', type=int, default=0, choices=[0,1,2,3,4],
                        help='0 means using the leftmost word(s), '
                             '1 means using the middle word(s), '
                             '2 means using the rightmost word(s),'
                             '3 means randomly selecting some words, '
                             '4 means selecting the token(s) with highest weight(s).')
    parser.add_argument('--generate_mode', type=int, default=0, choices=[0, 1, 2],
                        help = '0 for random, 1 for lm, 2 for combination')
    parser.add_argument('--ratio1', type=float, default=0.5)
    parser.add_argument('--ratio2', type=float, default=0.5)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    args.n_gpu = torch.cuda.device_count()

    if args.masked_lm ==0:
        masked_lm=''
    else:
        masked_lm='_masked_lm'
    if args.full_mask==0:
        full_mask=''
    else:
        full_mask='_full_mask'

    if args.generate_mode == 0:
        generate_mode = ''
    elif args.generate_mode == 1:
        generate_mode = '_lm_generate'
    elif args.generate_mode == 2:
        generate_mode = f'_combine_generate_{args.ratio1}_{args.ratio2}'
    else:
        raise ValueError('Wrong generate mode.')
    if args.random_init==1:
        prefix = 'random_initialization_{}{}{}{}_w{}_max_insert_label{}_insert_mode{}_encoder_loss_type{}'.\
        format(args.dataset,masked_lm, full_mask, generate_mode, args.w, args.num_labels-2, args.insert_mode, args.encoder_loss_type)
    elif args.random_init==0:
        prefix = '{}{}{}{}_w{}_max_insert_label{}_insert_mode{}_encoder_loss_type{}'.\
        format(args.dataset,masked_lm, full_mask, generate_mode, args.w, args.num_labels-2, args.insert_mode, args.encoder_loss_type)
    else:
        raise ValueError('Wrong initialization method.')


    model_path = f'../checkpoints/cbart-{args.bart}_{prefix}'
    log_path = f'../logs/keyword_generate'

    # args.model_path = model_path
    args.model_path = '../checkpoints/cbart-large_one-billion-words_w1.0_max_insert_label1_insert_mode0_encoder_loss_type0'
    args.log_path = log_path

    if args.local_rank in [-1, 0]:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_file = '{}/{}_{}.log'.format(log_path, f"cbart-{args.bart}", prefix)
        logger = Logger(log_file)
        logger.logger.info(f'The log file is {log_file}')
        logger.logger.info(args)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        logger.logger.info('Use {} gpus to train the model.'.format(args.n_gpu))

    try:
        # load the pre-trained model and tokenizer
        tokenizer = BartTokenizer.from_pretrained(args.model_path)
        model = BartForTextInfill.from_pretrained(args.model_path, num_labels=args.num_labels,encoder_loss_type=args.encoder_loss_type)
        if args.local_rank in [-1, 0]:
            logger.logger.info('Initialize BartForTextInfill from checkpoint {}.'.format(args.model_path))
    except:
        tokenizer = BartTokenizer.from_pretrained(f'facebook/bart-{args.bart}')
        if args.random_init ==1:
            #  load pre-trained config
            config = BartConfig.from_pretrained(f'facebook/bart-{args.bart}')
            # pass the config to model constructor instead of from_pretrained
            # this creates the model as per the params in config
            # but with weights randomly initialized
            model = BartForTextInfill(config)
            if args.local_rank in [-1, 0]:
                logger.logger.info(f'Random initialize the bart-{args.bart} model.')
        else:
            model = BartForTextInfill.from_pretrained(f'facebook/bart-{args.bart}', num_labels=args.num_labels,
                                                      encoder_loss_type=args.encoder_loss_type,full_mask=args.full_mask)
            if args.local_rank in [-1, 0]:
                logger.logger.info(f'Initialize the bart-{args.bart} model with default parameters.')

    if args.local_rank == -1 or args.n_gpu<=1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        torch.distributed.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print('local_rank:', args.local_rank)
    print("device:", device)
    model = model.to(device)
    if args.train==1:
        trainset = BARTDataset(args.dataset, "train", tokenizer=tokenizer, num_labels=args.num_labels,
                               insert_mode=args.insert_mode,encoder_loss_type=args.encoder_loss_type,
                               local_rank=args.local_rank,generate_mode=args.generate_mode, ratio1=args.ratio1, ratio2=args.ratio2)
        if args.local_rank in [-1, 0]:
            print(f'RAM memory size for train set: {asizeof.asizeof(trainset)/(1024.0**3):.2f}G.')

        if args.local_rank in [-1, 0]:
            logger.logger.info(f'The size of the train set is {len(trainset)}.')
        if args.local_rank == -1 or args.n_gpu <= 1:
            train_sampler = torch.utils.data.RandomSampler(trainset)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank])
            train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = DataLoader(trainset, num_workers=0, batch_size=args.batch_size, sampler=train_sampler,
                                 collate_fn=trainset.create_mini_batch)

    testset = BARTDataset(args.dataset, mode='dev', tokenizer=tokenizer,num_labels=args.num_labels,
                          insert_mode=args.insert_mode,encoder_loss_type=args.encoder_loss_type,
                          local_rank=args.local_rank,generate_mode=args.generate_mode,ratio1=args.ratio1, ratio2=args.ratio2)
    if args.local_rank in [-1, 0]:
        print(f'RAM memory size  for dev set: {asizeof.asizeof(testset)/(1024.0**3):.2f}G.')

    if args.local_rank in [-1, 0]:
        logger.logger.info(f'''The size of the dev set is {len(testset)}.''')
    # assert len(testset)%(args.test_batch_size*args.n_gpu) ==0
    if args.local_rank == -1 or args.n_gpu <= 1:
        test_sampler = torch.utils.data.SequentialSampler(testset)
    else:
        test_sampler = torch.utils.data.distributed.DistributedSampler(testset)
    testloader = DataLoader(testset, num_workers=0, batch_size=args.test_batch_size, sampler=test_sampler, collate_fn=testset.create_mini_batch)

    average_encoder_loss, average_decoder_loss, average_masked_decoder_loss, average_loss, used_time, recalls, precisions, f1s = \
        BARTDataset.compute_accuracy(model, 'dev', args.local_rank, testloader, len(testset), device, args.num_labels,
                                     encoder_loss_type=args.encoder_loss_type)
    if args.local_rank in [-1, 0]:
        logs = f'\n   dev set, ave loss {average_loss:.3f}, encoder loss {average_encoder_loss:.3f}, decoder loss {average_decoder_loss:.3f},' \
               f' mask decoder loss {average_masked_decoder_loss:.3f}, uses {used_time:.1f} seconds.'
        Macro_P = np.mean(list(precisions.values()))
        Macro_R = np.mean(list(recalls.values()))
        Macro_F1 = np.mean(list(f1s.values()))

        for i in range(len(f1s)):
            logs += f'''\n      Label_{i}: Precision={precisions[i]:.3f},  Recall={recalls[i]:.3f}, F1:{f1s[i]:.3f};'''
        logs += f'''\n      Macro_P={Macro_P:.3f},  Macro_R={Macro_R:.3f}, Macro_F1={Macro_F1:.3f}.'''
        logger.logger.info(logs)
    if args.train==0:
        exit()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.1, patience=2, verbose=True, min_lr=1e-6)
    scheduler.step(average_decoder_loss)
    if args.masked_lm:
        best_loss = average_encoder_loss + average_masked_decoder_loss
    else:
        best_loss = average_encoder_loss + average_decoder_loss

    evaluate_steps = max(int(len(trainset)/args.batch_size/5), 1)
    # evaluate_steps = 10
    print_steps = 10

    global_steps = 0
    local_step = 0

    start = time.time()
    total_loss = 0
    total_encoder_loss = 0
    total_decoder_loss = 0
    # fine-tune bart on the training dataset
    for epoch in range(args.epochs):
        # if args.n_gpu>1:
        #     # shuffle the data for each epoch
        #     train_sampler.set_epoch(epoch)
        for i, data in enumerate(trainloader):
            global_steps +=1
            local_step +=1
            data = [t.to(device) for t in data]
            encoder_inputs, encoder_labels, decoder_inputs, decoder_labels, masked_lm_labels, attention_mask = data
            if args.masked_lm:
                decoder_labels = masked_lm_labels
            encoder_loss, decoder_loss, encoder_logits, logits = model(encoder_inputs, encoder_labels = encoder_labels,
                     decoder_input_ids=decoder_inputs,labels = decoder_labels,attention_mask=attention_mask)[:4]
            # zero the parameter gradients
            optimizer.zero_grad()
            # backward
            loss = args.w*encoder_loss+decoder_loss
            # loss =decoder_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_encoder_loss += encoder_loss.item()
            total_decoder_loss += decoder_loss.item()
            if global_steps%print_steps==0 and args.local_rank in [-1, 0]:
                print("\rEpoch {}/{}, {}/{}, global steps {}, average loss is {:.3f},  average encoder loss is {:.3f}, average decoder loss is {:.3f},"
                      " {} steps uses {:.1f} seconds.".format(epoch+1, args.epochs,i+1,len(trainloader),global_steps, total_loss/local_step,
                                                              total_encoder_loss/local_step, total_decoder_loss/local_step, local_step,
                                                              time.time()-start), end='')
            if global_steps%evaluate_steps==0 :

                average_encoder_loss, average_decoder_loss, average_masked_decoder_loss, average_loss, used_time,recalls, precisions, f1s = \
                    BARTDataset.compute_accuracy(model, 'dev',args.local_rank, testloader, len(testset), device, args.num_labels,
                                                 encoder_loss_type=args.encoder_loss_type)
                if args.local_rank in [-1, 0]:
                    logs = f'\n   Dev set, ave loss {average_loss:.3f}, encoder loss {average_encoder_loss:.3f},' \
                           f' decoder loss {average_decoder_loss:.3f}, mask decoder loss {average_masked_decoder_loss:.3f}, uses {used_time:.1f} seconds.'
                    Macro_P = np.mean(list(precisions.values()))
                    Macro_R = np.mean(list(recalls.values()))
                    Macro_F1 = np.mean(list(f1s.values()))
                    for i in range(len(f1s)):
                        logs += f'''\n      Label_{i}: Precision={precisions[i]:.3f},  Recall={recalls[i]:.3f}, F1:{f1s[i]:.3f};'''
                    logs += f'''\n      Macro_P={Macro_P:.3f},  Macro_R={Macro_R:.3f}, Macro_F1={Macro_F1:.3f}.'''
                    logger.logger.info(logs)

                if args.masked_lm:
                    cur_loss = average_encoder_loss + average_masked_decoder_loss
                else:
                    cur_loss = average_encoder_loss + average_decoder_loss
                if cur_loss < best_loss:
                    if args.masked_lm:
                        best_loss = average_encoder_loss + average_masked_decoder_loss
                    else:
                        best_loss = average_encoder_loss + average_decoder_loss
                    if args.local_rank in [-1, 0]:
                        model_to_save = model.module if hasattr(model, "module") else model
                        # Simple serialization for models and tokenizers
                        logger.logger.info('Save the model at {}'.format(args.model_path))
                        model_to_save.save_pretrained(args.model_path)
                        tokenizer.save_pretrained(args.model_path)

                if args.local_rank in [-1, 0]:
                    step_path = f'{args.model_path}/global_steps{global_steps}'
                    if not os.path.exists(step_path):
                        os.makedirs(step_path)
                        model_to_save = model.module if hasattr(model, "module") else model
                        model_to_save.save_pretrained(step_path)
                        tokenizer.save_pretrained(step_path)


                scheduler.step(average_decoder_loss)
                start = time.time()
                total_loss = 0
                total_encoder_loss = 0
                total_decoder_loss = 0
                local_step = 0
