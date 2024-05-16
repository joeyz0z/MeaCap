import os
import sys
import time
import torch.nn.functional as F
import torch
import itertools
from torch.nn.utils.rnn import pad_sequence

# python 3.8以上不可以使用sys.path.append来添加搜索路径
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)
# sys.path.append('../')
from models.bart import BARTDataset

# encoder_labels : 0 for copy, 1 for replacement, 2 for insertion
# indicate_labels: 0 for copy, 1 for copy and insertion, 2 for copy, replacement and insertion, 3 for replacement


def enforce_repetition_penalty_( lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty=1):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i]):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty
    return  lprobs


def enforce_repetition_penalty_parallel( lprobs, prev_output_tokens, repetition_penalty=1):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    if len(lprobs.shape)==3:
        seqlen = lprobs.shape[1]
        prev_output_tokens = prev_output_tokens.unsqueeze(dim=1).expand(-1,seqlen,-1)
    gather_logits = torch.gather(lprobs, -1, prev_output_tokens)
    gather_logits[gather_logits>0] /= repetition_penalty
    gather_logits[gather_logits < 0] *= repetition_penalty
    lprobs.scatter_(-1, prev_output_tokens, gather_logits)
    return lprobs


def generate_function(
        model,
        tokenizer,
        vl_model,
        wte_model,
        select_memory_wte_embeddings,
        encoder_inputs,
        indicate_labels,
        encoder_loss_type,
        max_insert_label,
        device,
        decoder_inputs=None,
        stop_tokens_tensor=None,
        sub_tokens_tensor=None,
        num_beams=1,
        temperature=1,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1,
        refinement_steps=3,
        max_refinement_steps=10,
        adaptive=False,
        show_refine=1,
        threshold=0,
        decoder_chain=1,
        rank_lm=None,
        max_len=20,
        args=None,
        logger=None
    ):

    batch_size = len(indicate_labels)
    if do_sample:
        effective_batch_size = batch_size * decoder_chain
        if decoder_chain > 1:
            # expand inputs
            encoder_inputs = [e.clone() for e in encoder_inputs for i in range(decoder_chain)]
            if decoder_inputs is not None:
                decoder_inputs = [e.clone() for e in decoder_inputs for i in range(decoder_chain)]
            indicate_labels = [e[:] for e in indicate_labels for i in range(decoder_chain)]
    else:
        effective_batch_size = batch_size

    batch_refinement_steps = torch.tensor([0] * effective_batch_size)

    if adaptive:
        current_refinement = 0
        done = False
        while not done:
            predict_outputs, indicate_labels, batch_refinement, decoder_lengths = generate_step_parallel(
                 model, tokenizer, vl_model, wte_model, select_memory_wte_embeddings,
                 encoder_inputs, indicate_labels,
                 encoder_loss_type, max_insert_label, device,
                 decoder_inputs=decoder_inputs,
                 stop_tokens_tensor=stop_tokens_tensor,
                 sub_tokens_tensor = sub_tokens_tensor,
                 repetition_penalty=repetition_penalty,
                 num_beams=num_beams,
                 temperature=temperature,
                 do_sample=do_sample,
                 top_k=top_k,
                 top_p=top_p,
                 threshold=threshold,
                 max_len=max_len,
                 args=args
                 )
            encoder_inputs = predict_outputs
            print(predict_outputs)
            current_refinement +=1
            batch_refinement_steps += batch_refinement
            if torch.sum(batch_refinement) == 0 or current_refinement == max_refinement_steps:
                done = True
            decoder_inputs = None
    else:
        for i in range(refinement_steps):
            predict_outputs, indicate_labels, batch_refinement, decoder_lengths = generate_step_parallel(  # TODO: xieyan
                 model, tokenizer, vl_model, wte_model, select_memory_wte_embeddings,
                 encoder_inputs, indicate_labels,
                 encoder_loss_type, max_insert_label, device,
                 decoder_inputs=decoder_inputs,
                 stop_tokens_tensor=stop_tokens_tensor,
                 sub_tokens_tensor=sub_tokens_tensor,
                 repetition_penalty=repetition_penalty,
                 num_beams=num_beams,
                 temperature=temperature,
                 do_sample=do_sample,
                 top_k=top_k,
                 top_p=top_p,
                 threshold=threshold,
                 max_len=max_len,
                 args=args
                 )
            # 输出的predict_outputs和indicate_labels会带有prompt，需要将他们在refinement过程中去掉
            # predict_outputs = torch.cat((predict_outputs[:, :1], predict_outputs[:, 4:]), dim=1)
            # indicate_labels[0] = indicate_labels[0][3:]

            encoder_inputs = predict_outputs
            batch_refinement_steps += batch_refinement
            if torch.sum(batch_refinement) == 0:
                break
            else:
                if show_refine:
                    logger.logger.info(f"refinement {i+1}:")
                    for b in range(effective_batch_size):
                        print(tokenizer.decode(predict_outputs[b].tolist(), skip_special_tokens=False))
                        logger.logger.info(tokenizer.decode(predict_outputs[b].tolist(), clean_up_tokenization_spaces=False))
                        # logger.logger.info(tokenizer.convert_ids_to_tokens(predict_outputs[b].tolist()))
            decoder_inputs = None
    predict_outputs = [predict_outputs[i][:length] for i, length in enumerate(decoder_lengths)]
    if do_sample and decoder_chain > 1:
        _predict_outputs = []
        _batch_refinement_steps = []
        # use the rank_lm to select the best one from multi decoder chains
        log_ppls, probs = rank_lm.perplexity(input_ids = predict_outputs)
        log_ppls = log_ppls.view([batch_size, -1])
        indices = torch.argmax(-log_ppls,  dim=-1, keepdim=False)
        for b in range(batch_size):
            effective_index = b*decoder_chain + indices[b]
            _predict_outputs.append(predict_outputs[effective_index])
            _batch_refinement_steps.append(batch_refinement_steps[effective_index])

        batch_refinement_steps = _batch_refinement_steps
        predict_outputs = _predict_outputs

    return predict_outputs, batch_refinement_steps

def generate_step(
        model,
        tokenizer,
        encoder_inputs,
        indicate_labels,
        encoder_loss_type,
        max_insert_label,
        device,
        decoder_inputs=None,
        stop_tokens_tensor=None,
        sub_tokens_tensor=None,
        temperature=1,
        repetition_penalty=1,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        num_beams=1,
        threshold=0,
        max_len = None
    ):
    """

    :param model:
    :param encoder_inputs: list of one dimensional tensor
    :param indicate_labels: list of list of int, this tensor is used to denote which tokens are original,
    which tokens are generated. 0 for original tokens, 1 for boundary tokens, 2 for generated tokens.
    0 corresponds to encoder_labels [0], 1 corresponds to encoder_labels [0,2,3,4,5],
    2 corresponds to encoder_labels [0,1,2,3,4,5].
    :param encoder_loss_type: 0 for classification, 1 for regression
    :return:
    """
    start = time.time()
    mask_token_id = tokenizer.mask_token_id
    bos_token_id  = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        pre_predict_outputs = encoder_inputs
        encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value= pad_token_id)
        attention_mask = torch.zeros(encoder_inputs.shape,dtype=torch.float32)
        attention_mask = attention_mask.masked_fill(encoder_inputs != pad_token_id, 1)
        attention_mask = attention_mask.to(device)
        encoder_inputs = encoder_inputs.to(device)
        # s = time.time()
        # step 1: feed encoder_inputs into the encoder and get encoder_logits
        encoder_outputs, encoder_logits = model.get_encoder_logits(encoder_inputs, attention_mask=attention_mask)
        # e = time.time()
        bts = encoder_inputs.shape[0]
        # s = time.time()
        if decoder_inputs is None:
            # step 2: predict encoder_labels for input_ids based on encoder_logits
            indicate_labels, predict_labels_list = get_encoder_labels(encoder_logits, encoder_loss_type,indicate_labels,
                                                                 max_insert_label,threshold=threshold,max_len=max_len)

            # step 3: compute decoder_inputs based on encoder_inputs and encoder_labels
            decoder_inputs = [BARTDataset.create_decoder_inputs(encoder_inputs[i].tolist(),
                                                                predict_labels_list[i].tolist(), mask_token_id) for i in range(bts)]
        # e = time.time()
        # print(f'decoder inputs : {e-s}')
        # replace the eos_token_id with pad_token_id
        for i, _ in enumerate(decoder_inputs):
            decoder_inputs[i][-1] = pad_token_id

        decoder_lengths = [decoder_inputs[i].shape[0] for i in range(bts)]
        # create decoder_inputs by shifting the decoder_labels right,
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value = pad_token_id)

        decoder_labels = decoder_inputs.clone()
        decoder_inputs[:, 1:] = decoder_labels[:, :-1]
        decoder_inputs[:, 0] =  eos_token_id
        decoder_inputs = decoder_inputs.to(device)

        # step 4: feed decoder_inputs into the decoder and get decoder_logits in a non-auto-regressive way.
        # feed the encoder_outputs to avoid computing it again.
        encoder_logits, decoder_logits = model(input_ids=None, decoder_input_ids=decoder_inputs,
                                               attention_mask=attention_mask,encoder_outputs=encoder_outputs,
                                               use_cache=False)[:2]

        if num_beams>1:
            pass
        else:
            predict_outputs = _generate_no_beam_search(decoder_logits,decoder_inputs, bos_token_id, eos_token_id,
                                                       mask_token_id,indicate_labels,decoder_lengths,
                                                       stop_tokens_tensor=stop_tokens_tensor,
                                                       sub_tokens_tensor=sub_tokens_tensor ,
                                                       temperature=temperature,do_sample=do_sample,
                                                       top_k=top_k,top_p=top_p,repetition_penalty=repetition_penalty)
        refinement_steps = []
        for i in range(bts):
            if predict_outputs[i].shape[0]==pre_predict_outputs[i].shape[0] and all(predict_outputs[i]==pre_predict_outputs[i]):
                refinement_steps.append(0)
            else:
                refinement_steps.append(1)
        refinement_steps = torch.tensor(refinement_steps)
        # the predict_outputs is regarded as new encoder_inputs
    return predict_outputs, indicate_labels, refinement_steps, decoder_lengths


def generate_step_parallel(
        model,
        tokenizer,
        vl_model,
        wte_model,
        select_memory_wte_embeddings,
        encoder_inputs,
        indicate_labels,
        encoder_loss_type,
        max_insert_label,
        device,
        decoder_inputs=None,
        stop_tokens_tensor=None,
        sub_tokens_tensor=None,
        temperature=1,
        repetition_penalty=1,
        do_sample=False,
        top_k=0,
        top_p=1.0,
        num_beams=1,
        threshold=0,
        max_len = None,
        args=None
    ):
    """

    :param model:
    :param encoder_inputs: list of one dimensional tensor
    :param indicate_labels: list of list of int, this tensor is used to denote which tokens are original,
    which tokens are generated. 0 for original tokens, 1 for boundary tokens, 2 for generated tokens.
    0 corresponds to encoder_labels [0], 1 corresponds to encoder_labels [0,2,3,4,5],
    2 corresponds to encoder_labels [0,1,2,3,4,5].
    :param encoder_loss_type: 0 for classification, 1 for regression
    :return:
    """
    # start = time.time()
    mask_token_id = tokenizer.mask_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    model.eval()
    with torch.no_grad():
        if isinstance(encoder_inputs, list):
            encoder_inputs = pad_sequence(encoder_inputs, batch_first=True, padding_value=pad_token_id)
            encoder_inputs = encoder_inputs.to(device)

        attention_mask = torch.zeros(encoder_inputs.shape, dtype=torch.float32).to(device)
        attention_mask = attention_mask.masked_fill(encoder_inputs != pad_token_id, 1)
        pre_predict_outputs = encoder_inputs.clone()
        # s = time.time()
        # step 1: feed encoder_inputs into the encoder and get encoder_logits
        encoder_outputs, encoder_logits = model.get_encoder_logits(encoder_inputs, attention_mask=attention_mask)
        # e = time.time()
        bts, seqlen = encoder_inputs.shape
        pre_decoder_lengths = [len(e) for e in indicate_labels]
        if decoder_inputs is None:
            # step 2: predict encoder_labels for input_ids based on encoder_logits
            indicate_labels, predict_labels_list = get_encoder_labels(encoder_logits, encoder_loss_type, indicate_labels, max_insert_label,
                                                                      threshold=threshold, max_len=max_len, min_len=args.min_len,
                                                                      use_prompt=args.use_prompt,prompt_len=args.prompt_len,
                                                                      device=device)

            decoder_inputs = [BARTDataset.create_decoder_inputs(encoder_inputs[i].tolist()[:pre_decoder_lengths[i]],
                                                                predict_labels_list[i].tolist(), mask_token_id) for i in range(bts)]

        decoder_lengths = [len(e) for e in indicate_labels]
        # create decoder_inputs by shifting the decoder_labels right,
        decoder_inputs = pad_sequence(decoder_inputs, batch_first=True, padding_value=pad_token_id)
        decoder_inputs = decoder_inputs.to(device)

        decoder_labels = decoder_inputs.clone()
        decoder_inputs[:, 1:] = decoder_labels[:, :-1]
        decoder_inputs[:, 0] = eos_token_id

        # step 4: feed decoder_inputs into the decoder and get decoder_logits in a non-auto-regressive way.
        # feed the encoder_outputs to avoid computing it again.
        encoder_logits, decoder_logits = model(input_ids=None, decoder_input_ids=decoder_inputs,
                                               attention_mask=attention_mask, encoder_outputs=encoder_outputs,
                                               use_cache=False)[:2]

        if num_beams > 1:
            pass
        else:
            # if args.use_prompt:
            #     indicate_labels[0] = [0, 0, 0] + indicate_labels[0]  # prompt 会把句子变长，输出的decoder logit需要使用indicate_label进行处理，需要同时处理
            indicate_labels_tensor = [torch.tensor(e) for e in indicate_labels]

            indicate_labels_tensor = pad_sequence(indicate_labels_tensor, batch_first=True, padding_value = 1000)
            indicate_labels_tensor = indicate_labels_tensor.to(device)
            predict_outputs = _generate_no_beam_search_parallel(  # TODO: xieyan
                    decoder_logits,
                    decoder_labels,
                    mask_token_id,
                    indicate_labels_tensor,
                    stop_tokens_tensor=stop_tokens_tensor,
                    sub_tokens_tensor=sub_tokens_tensor,
                    temperature=temperature, do_sample=do_sample,
                    top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty,
                    tokenizer=tokenizer,
                    vl_model=vl_model,
                    wte_model=wte_model,
                    select_memory_wte_embeddings=select_memory_wte_embeddings,
                    args=args
            )

        refinement_steps = torch.zeros(bts).long()
        for i in range(bts):
            length1 = decoder_lengths[i]
            length2 = pre_decoder_lengths[i]
            if length1 != length2:
                refinement_steps[i] = 1
            else:
                if torch.sum(predict_outputs[i, :length1] == pre_predict_outputs[i, :length1], dim=-1) != length1:
                    refinement_steps[i] = 1
    return predict_outputs, indicate_labels, refinement_steps, decoder_lengths


def get_encoder_labels(encoder_logits, encoder_loss_type, indicate_labels_list, max_insert_label=1, threshold=0,
                       max_len=None, min_len=None, use_prompt=None, prompt_len=4, device=None):
    if encoder_loss_type == 0:  # classification
        # argmax
        if threshold > 0:
            probs = torch.softmax(encoder_logits, dim=-1)
            # encoder_logits[:,:,1:] += 0.7
            _index = probs[:, :, 0] >= threshold
            encoder_logits[_index] = 0
            predict_labels = torch.argmax(encoder_logits, dim=-1, keepdim=False)
            predict_labels[_index] = 0
        else:
            predict_labels = torch.argmax(encoder_logits, dim=-1, keepdim=False)
    else:  # regression, round and convert the output into torch.Long tensor
        predict_labels = torch.round(encoder_logits).long()

    for i, e in enumerate(indicate_labels_list):
        if len(e) > max_len+2:
            predict_labels[i][predict_labels[i] == 2] = 1  # change insert to replace

    if use_prompt:
        prompt_encoder_label = torch.zeros(1, prompt_len, dtype=torch.int64).to(device)
        predict_labels = torch.cat((prompt_encoder_label, predict_labels[:, prompt_len:]), dim=1)
        # prompt_encoder_label = torch.tensor([[0, 0, 0, 0]]).to(device)
        # predict_labels = torch.cat((prompt_encoder_label, predict_labels[:, 4:]), dim=1)

    # 在重新构造indicate_label和predict_label之前，这两个是等长的，都是与输入句子的长度相同（包含BOS与EOS）
    # 重新构造后，predict_label中有insert操作会改变indicate_label的长度
    # 因此如果我们将倒数第二位的predict_label改成insert，我们就需要将indicate
    # min_len + prompt_len是我们想要的句子长度
    if torch.sum(predict_labels) == 0 and predict_labels.shape[1] < min_len + prompt_len:
        # 同时修改indicate_labels和predict_labels
        # indicate_labels要保证句号位置可以insert，句号是倒数第二个单词
        predict_labels_list_tmp = predict_labels.squeeze(0).tolist()
        predict_labels_list_tmp[-2] = 2
        predict_labels = torch.tensor([predict_labels_list_tmp])
        print('expanding length ... ')

    bts = encoder_logits.shape[0]
    new_indicate_labels_list = []
    predict_labels_list = []
    for b in range(bts):
        new_indicate_labels = []
        indicate_labels = indicate_labels_list[b]
        for i ,e in enumerate(indicate_labels):
            predict_labels[b, i] = min(predict_labels[b, i], max_insert_label + 1)
            if e == 0:  # lexical constraints . only copy is allowed.
                if predict_labels[b, i] != 0:
                    predict_labels[b, i] = 0
            elif e == 1:  # the boundary token of lexical constraints. copy and insert are allowed.
                if predict_labels[b, i] == 1:  # change replacement to copy
                    predict_labels[b, i] = 0
            elif e == 2:  # generated tokens. all actions are allowed.
                pass
            elif e == 3:  # only replace is allowed.
                if predict_labels[b, i] == 2:  # change insertion to replacement
                    predict_labels[b, i] = 1
            else:
                raise ValueError(f'indicate_labels can only be [0,1,2,3].')

            if predict_labels[b, i] > 1:  # insert
                if use_prompt and (i < prompt_len):  # prompt的长度为4,因此4之前的predict_label插入操作会在后续被取消，因此indicate_label也要保持不变
                    pass
                else:
                    new_indicate_labels += [2]*(predict_labels[b, i]-1)
                    # new_indicate_labels += [2]
            new_indicate_labels.append(e)
        new_indicate_labels_list.append(new_indicate_labels)
        predict_labels_list.append(predict_labels[b, :len(indicate_labels)])
    return new_indicate_labels_list, predict_labels_list


def _generate_no_beam_search(
            decoder_logits,
            decoder_inputs,
            bos_token_id,
            eos_token_id,
            mask_token_id,
            indicate_labels_list,
            decoder_lengths,
            stop_tokens_tensor=None,
            sub_tokens_tensor=None,
            temperature=1,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1
):
    if temperature != 1:
        decoder_logits = decoder_logits / temperature
    bts = decoder_inputs.shape[0]
    new_encoder_inputs_list = []
    for b in range(bts):
        new_encoder_inputs = []
        indicate_labels = indicate_labels_list[b]
        seqlen = decoder_lengths[b]
        for i in range(seqlen):
            if i == 0:
                new_encoder_inputs.append(bos_token_id)  #
                continue
            if i == seqlen - 1:
                new_encoder_inputs.append(eos_token_id)  #
                continue
            if decoder_inputs[b, i+1] == mask_token_id:  # we need to predict the mask_token based on the i-th logits
                next_token_logits = decoder_logits[b, i].view(1, -1)
                # set the probability of stop tokens to 0
                if stop_tokens_tensor is not None:
                    next_token_logits = next_token_logits.masked_fill(stop_tokens_tensor >0 , -1e10)
                # forbid to insert sub tokens behind the lexical constraints
                if i>1 and indicate_labels[i-1]<2 and sub_tokens_tensor is not None:
                    next_token_logits = next_token_logits.masked_fill(sub_tokens_tensor > 0, -1e10)

                # repetition penalty
                prev_output_tokens = [new_encoder_inputs + decoder_inputs[b].tolist()]
                next_token_logits = enforce_repetition_penalty_\
                    (next_token_logits, 1, 1, prev_output_tokens=prev_output_tokens,
                     repetition_penalty=repetition_penalty)
                next_token = torch.argmax(next_token_logits, dim=-1)
                new_encoder_inputs.append(next_token.item())
            else:
                new_encoder_inputs.append(decoder_inputs[b, i+1])

        new_encoder_inputs_list.append(torch.tensor(new_encoder_inputs))

    return new_encoder_inputs_list


def generate_caption_step(out, gen_idx, mask, temperature=None, top_k=100):
    """ Generate a word from out[gen_idx]
    args:
        - out (torch.Tensor): tensor of logits of size (batch_size, seq_len, vocab_size)
        - gen_idx (int): location for which to generate for
        - mask (torch.Tensor): (1, vocab_size)
        - extend_ids: (batch_size, extend_len)
        - top_k (int): candidate k
    """
    # logits = out[:, gen_idx]
    logits = out.squeeze(0)
    if temperature is not None:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    top_k_probs, top_k_ids = probs.topk(top_k, dim=-1)

    # if (extend_ids is not None) and (extend_ids not in top_k_ids):
    # if extend_ids is not None:
    #     # Need to be optimize when extend_ids in top_k_ids
    #     top_k_probs = torch.cat((top_k_probs, torch.gather(probs, dim=-1, index=extend_ids)), dim=-1)
    #     top_k_ids = torch.cat((top_k_ids, extend_ids), dim=-1)

    return top_k_probs, top_k_ids


def _generate_no_beam_search_parallel(
            decoder_logits,
            decoder_labels,
            mask_token_id,
            indicate_labels_tensor,
            stop_tokens_tensor=None,
            sub_tokens_tensor=None,
            temperature=1,
            do_sample=False,
            top_k=0,
            top_p=1.0,
            repetition_penalty=1,
            tokenizer=None,
            vl_model=None,
            wte_model=None,
            select_memory_wte_embeddings=None,
            args=None

):
    """
    parallel for batch and seqlen
    :param decoder_logits:
    :param decoder_labels:
    :param mask_token_id:
    :param indicate_labels_tensor:
    :param stop_tokens_tensor:
    :param sub_tokens_tensor:
    :param temperature:
    :param do_sample:
    :param top_k:
    :param top_p:
    :param repetition_penalty:
    :return:
    """

    if temperature != 1:
        # [b, seq_len, vocab_size]
        decoder_logits = decoder_logits / temperature
    # set the probability of stop tokens to 0
    if stop_tokens_tensor is not None:
        decoder_logits = decoder_logits.masked_fill(stop_tokens_tensor > 0, -1e10)

    # repetition penalty
    decoder_logits = enforce_repetition_penalty_parallel(decoder_logits,
                                                         prev_output_tokens=decoder_labels,
                                                         repetition_penalty=repetition_penalty)

    if sub_tokens_tensor is not None:
        _tmp = indicate_labels_tensor.clone()
        _tmp[:, 1:] = indicate_labels_tensor[:, :-1]
        _tmp[:, 1] = 2
        # forbid to insert sub tokens behind the lexical constraints
        lexical_index = _tmp < 2
        decoder_logits[lexical_index] = decoder_logits[lexical_index].masked_fill(sub_tokens_tensor > 0, -1e10)
    # predict the mask tokens
    mask_token_index = decoder_labels == mask_token_id
    mask_positions = torch.nonzero(mask_token_index[0, :])
    logits = decoder_logits[mask_token_index]
    if logits.shape[0] == 0:
        return decoder_labels
    else:
        if args.conzic_sample:  # TODO: xieyan
            mask_num = logits.shape[0]
            for i in range(mask_num):
                mask_logits = logits[i, :]
                mask_pos = mask_positions[i]
                generate_token = conzic_sample_function(mask_logits, tokenizer, vl_model, wte_model,
                                                        mask_pos=mask_pos, unfinish_seq=decoder_labels,
                                                        select_memory_wte_embeddings=select_memory_wte_embeddings, args=args)
                decoder_labels[:, int(mask_pos)] = generate_token


        # if do_sample:
        #     logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        #     # Sample
        #     probs = torch.softmax(logits, dim=-1)
        #     predict_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        # else:
        #     predict_tokens = torch.argmax(logits, dim=-1)

        # decoder_labels[mask_token_index] = predict_tokens

    return decoder_labels


def conzic_sample_function(lm_logits=None,
                           tokenizer=None,
                           match_model=None,
                           wte_model=None,
                           unfinish_seq=None,
                           mask_pos=None,
                           select_memory_wte_embeddings=None,
                           args=None):
    '''
    conzic sample method for language model generate
    paper link: https://arxiv.org/abs/2303.02437
    '''
    t_start = time.time()
    probs = F.softmax(lm_logits, dim=-1)
    probs, idxs = probs.topk(args.conzic_top_k, dim=-1)
    topk_seq = unfinish_seq.repeat(idxs.shape[0], 1)
    topk_seq[:, int(mask_pos)] = idxs
    # topk_inp_batch = topk_seq.view(-1, topk_seq.shape[-1])
    batch_text_list = tokenizer.batch_decode(topk_seq, skip_special_tokens=True)
    gen_text_embedding = match_model.compute_text_representation(batch_text_list)
    clip_score, clip_ref = match_model.compute_image_text_similarity_via_embeddings(args.img_embeds, gen_text_embedding)

    # 获得memroy中最相似的五句话和它们的embedding
    if args.use_memory:

        gen_text_wte_embedding = wte_model.encode(batch_text_list, convert_to_tensor=True)
        memroy_text_wte_embedding = torch.mean(select_memory_wte_embeddings, dim=0).unsqueeze(0)
        memory_ref = torch.cosine_similarity(memroy_text_wte_embedding, gen_text_wte_embedding, dim=1)
        memory_score = torch.softmax(memory_ref, dim=0).unsqueeze(0)
        clip_score = args.beta * clip_ref + args.gamma * memory_ref

    final_score = args.alpha * probs + clip_score
    best_clip_id = final_score.argmax(dim=1).view(-1, 1)
    generate_token = idxs[best_clip_id]
    # print(f'conzic sample using {time.time() - t_start}s')
    return generate_token


def construct_model_inputs(masked_sentences, tokenizer, decoder_inputs_list=None, args=None):
    """
    masked_sentences: list of 'keyword1 keyword2 keyword3...'
    """
    indicate_labels_list = []
    encoder_inputs_list = []
    for masked_sentence in masked_sentences:
        indicate_labels = [0]
        encoder_inputs = [tokenizer.bos_token_id]
        words = masked_sentence.split()
        for i, w in enumerate(words):
            ids = tokenizer.encode(' ' + w, add_special_tokens=False)  # 将输入的词都是带着空格输入的，可能是为了与训练相匹配，并且训练的时候大部分词都是前面有空格的
            encoder_inputs += ids
            indicate_labels += [1] + [0] * (len(ids) - 1)  # can insert before the current token
        encoder_inputs.append(tokenizer.eos_token_id)
        indicate_labels.append(1)
        indicate_labels_list.append(indicate_labels)
        encoder_inputs_list.append(encoder_inputs)

    encoder_inputs_list = [torch.tensor(e) for e in encoder_inputs_list]
    if decoder_inputs_list is not None:
        decoder_inputs_list = [torch.tensor(e) for e in decoder_inputs_list]

    return encoder_inputs_list, decoder_inputs_list, indicate_labels_list


def Get_shuffle_score(batch_embeds, masked_sentences, model, match_model, wte_model, tokenizer, select_memory_wte_embeddings,
                      stop_tokens_tensor, sub_tokens_tensor, rank_lm, logger, args, device):
    args.img_embeds = batch_embeds

    shuffle_list = masked_sentences
    all_masked_sentences = []
    all_masked_sentences.append([' '.join(shuffle_list)])
    gen_text = []
    for i in range(len(all_masked_sentences)):
        masked_sentences = all_masked_sentences[i]
        logger.logger.info(f'   Now input is: {masked_sentences} ')
        logger.logger.setLevel(logger.level_relations.get('warning'))

        encoder_inputs_list, decoder_inputs_list, indicate_labels_list = construct_model_inputs(masked_sentences, tokenizer,
                                                                                                decoder_inputs_list=None,
                                                                                                args=args)

        indicate_labels = indicate_labels_list[0:1]
        encoder_inputs = encoder_inputs_list[0:1]
        masked_sentence = masked_sentences[0:1]
        if decoder_inputs_list is not None:
            decoder_inputs = decoder_inputs_list[0:1]
        else:
            decoder_inputs = None
        length = len(encoder_inputs_list)
        batch_size = args.batch_size

        # generate
        predict_outputs, refinement_steps = generate_function(model, tokenizer, match_model, wte_model, select_memory_wte_embeddings,
                                                              encoder_inputs, indicate_labels,
                                                              args.encoder_loss_type,
                                                              args.max_insert_label,
                                                              device,
                                                              decoder_inputs=decoder_inputs,
                                                              stop_tokens_tensor=stop_tokens_tensor,
                                                              sub_tokens_tensor=sub_tokens_tensor,
                                                              temperature=args.temperature,
                                                              do_sample=args.do_sample,
                                                              top_k=args.top_k,
                                                              top_p=args.top_p,
                                                              refinement_steps=args.refinement_steps,
                                                              max_refinement_steps=args.max_refinement_steps,
                                                              adaptive=args.adaptive,
                                                              repetition_penalty=args.repetition_penalty,
                                                              threshold=args.threshold,
                                                              decoder_chain=args.decoder_chain,
                                                              rank_lm=rank_lm,
                                                              max_len=args.max_len,
                                                              args=args,
                                                              logger=logger
                                                              )
        gen_text.append(tokenizer.decode(predict_outputs[0], skip_special_tokens=True))
        logger.logger.setLevel(logger.level_relations.get('debug'))
        logger.logger.info(f'   Now result is: {tokenizer.decode(predict_outputs[0], skip_special_tokens=True)} ')

    return gen_text


def filter_text(text_list, image_embedding, match_model):
    clip_score, clip_ref = match_model.compute_image_text_similarity_via_raw_text(image_embedding, text_list)
    best_text_id = torch.argmax(clip_score, dim=-1)
    best_text = text_list[best_text_id]

    return best_text, best_text_id
