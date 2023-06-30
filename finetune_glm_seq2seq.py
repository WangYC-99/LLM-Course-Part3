# -*- encoding: utf-8 -*-
'''
@File    :   pretrain_gpt2.py
@Time    :   2021/10/06 00:58:32
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
import torch
import argparse
import numpy as np

from SwissArmyTransformer.training import initialize_distributed, set_random_seed
from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import BinaryDataset
from SwissArmyTransformer.tokenization.cogview import TextCodeTemplate

local_cnt = 0

def setup_model(args):
    return GLMModel.from_pretrained(args, args.ckpt_path)

def get_batch(data_iterator, args, timers):
    # Items and their type.
    keys = ['tokens', 'labels', 'loss_mask', 'position_ids', 'attention_mask']
    datatype = torch.int64

    # Broadcast data.
    timers('data loader').start()
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    timers('data loader').stop()

    data_b = mpu.broadcast_data(keys, data, datatype)
    # Unpack.
    tokens = data_b['tokens'].long()
    labels = data_b['labels'].long()
    loss_mask = data_b['loss_mask'].long()
    attention_mask = data_b['attention_mask'].float()
    position_ids = data_b['position_ids'].long()

    # Convert
    if args.fp16:
        attention_mask = attention_mask.half()

    return tokens, labels, loss_mask, attention_mask, position_ids

def forward_step(data_iterator, model, args, timers):
    """Forward step."""
    global local_cnt

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator, args, timers)
    # print('>>>>>>>>>>>>>>>>>>>>this is line 67 and tokens is : {}'.format(tokens))
    timers('batch generator').stop()
    # Forward model.
    if local_cnt < 3:
        print(tokens.shape)
        # print('tokens', tokens.shape, tokens)
        # print('position_ids', position_ids.shape, position_ids)
    # exit(0)
    logits, *mems = model(tokens, position_ids, attention_mask)
    if local_cnt < 3:
        print(logits.shape)
        # print(logits.shape, logits)
    losses = mpu.vocab_parallel_cross_entropy(
        logits.contiguous().float(), labels)
    # scaling loss mask
    loss_mask = loss_mask.view(-1)

    losses = losses.view(-1) * loss_mask
    loss = torch.sum(losses) / loss_mask.sum()
    local_cnt += 1
    return loss, {}


def create_dataset_function(path, args):
    sample_length = 160
    layout = [0, 160, 160+160, 160+160+160*2, 160+160+160*2+2]  # FIXME

    def process_fn(row):
        row = row.astype(np.int64)
        codes = [row[layout[i-1]:layout[i]] for i in range(1, len(layout))]
        tokens, labels, position_ids, att_flags = codes
        tokens = tokens.reshape(sample_length)
        labels = labels.reshape(sample_length)
        position_ids = position_ids.reshape(2, sample_length)
        attention_mask = np.ones(
            (1, sample_length, sample_length), dtype=np.int64)
        context_length, full_length = att_flags
        attention_mask = np.tril(attention_mask)
        attention_mask[:, :, :context_length] = 1
        attention_mask[:, full_length:, :] = 0
        loss_mask = np.zeros_like(tokens)
        loss_mask[context_length:full_length] = 1
        return dict(tokens=tokens, labels=labels, loss_mask=loss_mask, position_ids=position_ids, attention_mask=attention_mask)
    return BinaryDataset(path, process_fn, length_per_sample=layout[-1], dtype=np.uint16)


if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--ckpt-path', type=str, default=None)
    known, args_list = py_parser.parse_known_args()
    # print(args_list)
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    model, args = setup_model(args)
    training_main(args, model_cls=model, forward_step_function=forward_step,
                  create_dataset_function=create_dataset_function)