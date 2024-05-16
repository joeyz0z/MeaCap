# -*- coding: utf-8 -*-
# @Time    : 2020/8/11 3:41 PM
# @Author  : He Xingwei

import torch
import time
from torch.nn.utils.rnn import pad_sequence

class LanguageModel(object):

    def __init__(self, device, model, tokenizer, repetition_penalty = 1):
        """

        :param device:
        :param forward_lm: an instance for LSTMLanguageModel, GPT2 LM .
        :param forward_lm_tokenizer:
        """
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.repetition_penalty = repetition_penalty
        self.model.to(self.device)
        self.model.eval()
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')


    @staticmethod
    def enforce_repetition_penalty_parallel(lprobs, prev_output_tokens, repetition_penalty=1):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        if len(lprobs.shape) == 3:
            seqlen = lprobs.shape[1]
            prev_output_tokens = prev_output_tokens.unsqueeze(dim=1).expand(-1, seqlen, -1)
        gather_logits = torch.gather(lprobs, -1, prev_output_tokens)
        gather_logits[gather_logits > 0] /= repetition_penalty
        gather_logits[gather_logits < 0] *= repetition_penalty
        lprobs.scatter_(-1, prev_output_tokens, gather_logits)
        return lprobs

    def conditional_distribution(self,input_ids,  previous_is_keyword = False,
                                 sub_tokens_tensor=None, stop_tokens_tensor=None, past = None,
                                 prev_output_tokens = None):
        """
        this function is meant to get the distribution of p(x_n |x<n ).
        :param language_model: an instance for LSTMLanguageModel, GPT2 LM or XLNet LM.
        :param input_ids: one dimensional list. the input_ids will start with the bos_token_id.
        :param device: default None
        :return: the top_k probabilities and tokens
        """
        language_model = self.model
        device = self.device
        _input_ids = torch.tensor([input_ids])
        input_ids = _input_ids.to(device)

        with torch.no_grad():
            # s = time.time()
            outputs = language_model(input_ids[:,-1].view(-1,1), past=past)
            # print('-------')
            # print( time.time()-s)
            # print(outputs[0][:,-1,:10])
            # s = time.time()
            # outputs = language_model(input_ids, past=None)
            # print(time.time()-s)
            #  shape: [1, 1, vocab_size]
            logits, past = outputs[:2]
            # print(logits.shape)
            # print(outputs[0][:,-1,:10])

            logits = logits[0, -1, :]
            # set the probability of stop tokens to 0
            if stop_tokens_tensor is not None:
                logits = logits.masked_fill(stop_tokens_tensor > 0, -1e10)
            # forbid to insert sub tokens behind the lexical constraints
            if sub_tokens_tensor is not None and previous_is_keyword:
                logits = logits.masked_fill(sub_tokens_tensor > 0, -1e10)

            if self.repetition_penalty!=1 and prev_output_tokens is not None:
                prev_output_tokens = prev_output_tokens.to(self.device)
                logits = logits.reshape(1,1,-1)
                logits = self.enforce_repetition_penalty_parallel(logits, prev_output_tokens, self.repetition_penalty)
                logits = logits.reshape(-1)

            conditional_probs = torch.softmax(logits, -1)
        return conditional_probs, past


    def conditional_distribution_unidirectional_lm(self, input_ids, previous_is_keyword=False, top_k = -1,
                                                   sub_tokens_tensor=None, stop_tokens_tensor=None, past=None,
                                                   prev_output_tokens = None):
        """
        this function is meant to get the distribution of p(x_n |x<n)
        :param input_ids:
        :param top_k:
        :return:
        """

        conditional_probs, past = self.conditional_distribution( input_ids,
                                                                previous_is_keyword=previous_is_keyword,
                                                                sub_tokens_tensor = sub_tokens_tensor,
                                                                stop_tokens_tensor = stop_tokens_tensor,
                                                                past=past,prev_output_tokens = prev_output_tokens)
        if top_k != -1:
            # select the top_k probabilities and tokens
            top_k_conditional_probs, top_k_token_ids = torch.topk(conditional_probs, top_k)
            top_k_conditional_probs = top_k_conditional_probs.cpu().numpy()
            top_k_token_ids = top_k_token_ids.cpu().numpy()
        else:
            top_k_conditional_probs = None
            top_k_token_ids = None
        conditional_probs = conditional_probs.cpu().numpy()

        return top_k_conditional_probs, top_k_token_ids, conditional_probs, past



    def decode(self, input_ids, hx=None, past=None, previous_is_keyword = False, sub_tokens_tensor=None,
               stop_tokens_tensor=None, prev_output_tokens = None):
        # sequential decoding from left to right
        if hx is not None:
            past = hx
        language_model = self.model
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = language_model(input_ids[:,-1].view(-1,1), past=past)
            logits, past = outputs[:2]
            logits = logits[0, -1, :]
            # set the probability of stop tokens to 0
            if stop_tokens_tensor is not None:
                logits = logits.masked_fill(stop_tokens_tensor > 0, -1e10)
            # forbid to insert sub tokens behind the lexical constraints
            if sub_tokens_tensor is not None and previous_is_keyword:
                logits = logits.masked_fill(sub_tokens_tensor > 0, -1e10)

            if self.repetition_penalty!=1 and prev_output_tokens is not None:
                prev_output_tokens = prev_output_tokens.to(self.device)
                logits = logits.reshape(1,1,-1)
                logits = self.enforce_repetition_penalty_parallel(logits, prev_output_tokens, self.repetition_penalty)
                logits = logits.reshape(-1)

            conditional_probs = torch.softmax(logits, -1)
        return torch.log(conditional_probs+1e-10), past

    def perplexity(self, input_ids=None, input_texts=None):
        if input_ids is None:
            assert input_texts is not None
            input_ids = []
            for text in input_texts:
                ids = self.tokenizer.encode(text)
                ids = [self.tokenizer.bos_token_id] + ids + [self.tokenizer.eos_token_id]
                input_ids.append(torch.tensor(ids))

        label_ids = [s.clone() for s in input_ids]
        lengths_tensors = torch.tensor([len(s)-1 for s in input_ids])
        # gpt2 does not have the [PAD] token.
        # pad input with 0 (the padded value can be arbitrary number.)
        input_tensors = pad_sequence(input_ids, batch_first=True, padding_value=0)
        # pad label with -100 (can not be other number.)
        labels_tensors = pad_sequence(label_ids, batch_first=True, padding_value=-100)
        # 1 for real tokens and 0 for padded tokens
        masks_tensors = torch.zeros(labels_tensors.shape,dtype=torch.float32)
        masks_tensors = masks_tensors.to(self.device)
        input_tensors = input_tensors.to(self.device)
        labels_tensors = labels_tensors.to(self.device)
        lengths_tensors = lengths_tensors.to(self.device)
        masks_tensors = masks_tensors.masked_fill(labels_tensors != -100, 1)
        labels_tensors = labels_tensors[:, 1:]

        outputs = self.model(input_tensors, attention_mask=masks_tensors)
        logits = outputs[0]
        logits = logits[:, :-1, :]
        loss_ = self.loss_func(logits.reshape(-1, logits.shape[-1]), labels_tensors.reshape(-1))
        loss_ = loss_.reshape(labels_tensors.shape)
        loss_ = torch.sum(loss_, dim=-1).double()
        # log_ppls = (loss_ / lengths_tensors).cpu().numpy()
        # probs = torch.exp(-loss_).cpu().numpy()
        log_ppls = (loss_ / lengths_tensors)
        probs = torch.exp(-loss_)
        return log_ppls, probs


