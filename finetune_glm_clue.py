# -*- encoding: utf-8 -*-
'''
@File    :   finetune_glm_clue.py
@Time    :   2023/01/01 00:58:32
@Author  :   Yuanchun Wang
@Contact :   frederickwang99@gmail.com
'''

# here put the import lib
import os
import sys
import math
import random
from venv import create
import torch
import argparse
import numpy as np

from SwissArmyTransformer.training import initialize_distributed, set_random_seed
from SwissArmyTransformer import mpu, get_args, get_tokenizer
from SwissArmyTransformer.model import GLMModel
from SwissArmyTransformer.training.deepspeed_training import training_main
from SwissArmyTransformer.data_utils import CLUEDataset
from SwissArmyTransformer.tokenization.cogview import TextCodeTemplate
from SwissArmyTransformer.model.mixins import MLPHeadMixin, PrefixTuningMixin

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class ClassificationModel(GLMModel):
    def __init__(self, args, transformer=None, parallel_output=True):
        super().__init__(args, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('classification_head', MLPHeadMixin(args.hidden_size, 2048, 17))
        # self.add_mixin('prefix-tuning', PrefixTuningMixin(args.num_layers, args.hidden_size // args.num_attention_heads, args.num_attention_heads, args.prefix_len))
    def disable_untrainable_params(self):
        self.transformer.word_embeddings.requires_grad_(False)
        # for layer_id in range(len(self.transformer.layers)):
        #     self.transformer.layers[layer_id].requires_grad_(False)

local_cnt = 0

def setup_model(args):
    return ClassificationModel.from_pretrained(args, args.ckpt_path)

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
    # seq_length = 160
    # position_ids = torch.zeros(2, seq_length, device=tokens.device, dtype=torch.long)
    # torch.arange(0, seq_length, out=position_ids[0, :seq_length])
    # position_ids = position_ids.unsqueeze(0)
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
    timers('batch generator').stop()
    # Forward model.
    if local_cnt < 3:
        print('tokens shape:', tokens.shape)

    logits, *mems = model(tokens, position_ids, attention_mask)
    if local_cnt < 3:
        print('logits shape:', logits.shape)
        torch.set_printoptions(threshold=np.inf)
        # print('logits pred: ', logits)

    # pred = logits.contiguous().float().squeeze(-1)[..., -1]
    # pred_01 = pred
    pred = logits.contiguous().float().squeeze(-1)[..., -1, :]

    gap = torch.tensor(100)
    labels = labels - gap
    # soft = torch.nn.Softmax(dim=0)
    # pred = soft(pred)
    if local_cnt < 3:
        print('pred shape:', pred.shape)
        print('labels shape:', labels.shape)
        print('argmax pred:', torch.argmax(pred, dim=-1).shape)
        # print('pred_01_shape:', pred_01.shape)
        # print('pred_01:', pred_01)
        # print('pred_02_shape:', pred_02.shape)
        # print('pred_02:', pred_02)
        print('pred argmax:{}, labels:{}'.format(torch.argmax(pred, dim=-1), labels))
    local_cnt += 1
    loss = torch.nn.functional.cross_entropy(
        pred,
        labels.long()
    )
    # print('acc:{}, result:{},  total:{}'.format((torch.argmax(pred, dim=-1)).long() == labels, ((torch.argmax(pred, dim=-1)).long() == labels).sum() , labels.numel()))
    # print('pred 0 :', pred[0])
    acc = ((torch.argmax(pred, dim=-1)).long() == labels).sum() / labels.numel()

    return loss, {'acc': acc}


def create_dataset_function(path, args):
    sample_length = 160
    seq_length = 160

    def process_fn(row):
        tokens = torch.tensor(row['tokens'])
        labels = torch.tensor(int(row['label']))
        # row = row.astype(np.int64)
        # codes = [row[layout[i-1]:layout[i]] for i in range(1, len(layout))]
        # tokens, labels, position_ids, att_flags = codes
        # tokens = tokens.reshape(sample_length)
        # labels = labels.reshape(sample_length)
        # position_ids = position_ids.reshape(2, sample_length)
        attention_mask = np.ones(
            (1, sample_length, sample_length), dtype=np.int64)
        # context_length, full_length = att_flags
        attention_mask = np.tril(attention_mask)
        attention_mask[:, :, :sample_length] = 1
        attention_mask[:, sample_length:, :] = 0
        loss_mask = np.zeros_like(tokens)
        loss_mask[0:sample_length] = 1
        position_ids = torch.zeros(2, seq_length, device=tokens.device, dtype=torch.long)
        torch.arange(0, seq_length, out=position_ids[0, :seq_length])
        # position_ids = position_ids.unsqueeze(0)
        return dict(tokens=tokens, labels=labels, loss_mask=loss_mask, position_ids = position_ids, attention_mask=attention_mask)
    return CLUEDataset(path, process_fn)


if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    py_parser.add_argument('--ckpt-path', type=str, default=None)
    known, args_list = py_parser.parse_known_args()
    # print(args_list)
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    model, args = setup_model(args)
    dataset = create_dataset_function('/data/WangYC/clue_tokens.jsonl', args)
    # print(dataset[0])
    training_main(args, model_cls=ClassificationModel, forward_step_function=forward_step,
                  create_dataset_function=create_dataset_function)