# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

import pandas as pd
import json
import time

import logging
import os
import argparse
import random
import pickle
from tqdm import tqdm, trange
import spacy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import *
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class SwagExample(object):
    """A single training/test example for the SWAG dataset."""

    def __init__(self,
                 swag_id,
                 context_sentence,
                 start_ending,
                 ending_0,
                 ending_1,
                 ending_2,
                 ending_3,
                 que_heads,
                 que_types,
                 que_span,
                 doc_heads,
                 doc_types,
                 doc_span,
                 token_doc,
                 token_que,
                 token_opt,
                 opt_heads,
                 opt_types,
                 opt_span,
                 label=None):
        self.swag_id = swag_id
        self.context_sentence = context_sentence
        self.start_ending = start_ending
        self.endings = [
            ending_0,
            ending_1,
            ending_2,
            ending_3,
        ]
        self.label = label
        self.que_heads = que_heads
        self.que_types = que_types
        self.que_span = que_span
        self.doc_heads = doc_heads
        self.doc_types = doc_types
        self.doc_span = doc_span
        self.token_doc = token_doc
        self.token_que = token_que
        self.token_opt = token_opt
        self.opt_heads = opt_heads
        self.opt_types = opt_types
        self.opt_span = opt_span

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        l = [
            f"swag_id: {self.swag_id}",
            f"context_sentence: {self.context_sentence}",
            f"start_ending: {self.start_ending}",
            f"ending_0: {self.endings[0]}",
            f"ending_1: {self.endings[1]}",
            f"ending_2: {self.endings[2]}",
            f"ending_3: {self.endings[3]}",
        ]

        if self.label is not None:
            l.append(f"label: {self.label}")

        return ", ".join(l)


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

                 ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'doc_len': doc_len,
                'ques_len': ques_len,
                'option_len': option_len,
                'input_span_mask': input_span_mask
            }
            for _, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len, input_span_mask in
            choices_features
        ]
        self.label = label


def rea_race(path):
    with open(path, 'r', encoding='utf_8') as f:
        data_all = json.load(f)
        article = []
        question = []
        st = []
        ct1 = []
        ct2 = []
        ct3 = []
        ct4 = []
        y = []
        q_id = []
        for instance in data_all:

            ct1.append(' '.join(instance['options'][0]))
            ct2.append(' '.join(instance['options'][1]))
            ct3.append(' '.join(instance['options'][2]))
            ct4.append(' '.join(instance['options'][3]))
            question.append(' '.join(instance['question']))
            q_id.append(instance['q_id'])
            art = instance['article']
            l = []
            for i in art: l += i
            article.append(' '.join(l))
            y.append(instance['ground_truth'])

        return article, question, ct1, ct2, ct3, ct4, y, q_id


class SimpleNlp(object):
    def __init__(self):
        self.nlp = spacy.load('en', disable=['parser', 'tagger', 'ner', 'textcat'])
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))
        # self.nlp = nltk.data.load('tokenizers/punkt/english.pickle').tokenize

    def nlp(self, text):
        return self.nlp(text)


def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def read_race_examples(input_file, input_tag_file, is_training):
    article, question, ct1, ct2, ct3, ct4, y, q_id = rea_race(input_file)

    input_tag_data = []
    simple_nlp = SimpleNlp()

    with open(input_tag_file, "r", encoding='utf-8') as reader:
        for line in reader:
            input_tag_data.append(json.loads(line))
    qas_id_to_tag_idx_map = {}
    all_dqtag_data = []
    for idx, tag_data in enumerate(tqdm(input_tag_data, ncols=50, desc="tagging...")):
        qas_id = tag_data["qas_id"]
        qas_id_to_tag_idx_map[qas_id] = idx
        tag_rep = tag_data["tag_rep"]
        dqtag_data = {
            "qas_id": qas_id,
            "head_que": [int(i) for i in tag_rep["pred_head_que"]],
            "span_que": [eval(i) for i in tag_rep["hpsg_list_que"]],
            "type_que": tag_rep["pred_type_que"],
            "head_opt1": [int(i) for i in tag_rep["pred_head_opt1"]],
            "span_opt1": [eval(i) for i in tag_rep["hpsg_list_opt1"]],
            "type_opt1": tag_rep["pred_type_opt1"],
            "head_opt2": [int(i) for i in tag_rep["pred_head_opt2"]],
            "span_opt2": [eval(i) for i in tag_rep["hpsg_list_opt2"]],
            "type_opt2": tag_rep["pred_type_opt2"],
            "head_opt3": [int(i) for i in tag_rep["pred_head_opt3"]],
            "span_opt3": [eval(i) for i in tag_rep["hpsg_list_opt3"]],
            "type_opt3": tag_rep["pred_type_opt3"],
            "head_opt4": [int(i) for i in tag_rep["pred_head_opt4"]],
            "span_opt4": [eval(i) for i in tag_rep["hpsg_list_opt4"]],
            "type_opt4": tag_rep["pred_type_opt4"],
            "span_doc": [eval(i) for sen_span in tag_rep["hpsg_list_doc"] for i in sen_span],
            "type_doc": [i for sen in tag_rep["pred_type_doc"] for i in sen],
            "head_doc": [int(i) for sen_head in tag_rep["pred_head_doc"] for i in sen_head],
            "token_doc": [token for sen_token in tag_rep['doc_tokens'] for token in sen_token],
            "token_que": tag_rep['que_tokens'],
            "token_opt1": tag_rep['opt1_tokens'],
            "token_opt2": tag_rep['opt2_tokens'],
            "token_opt3": tag_rep['opt3_tokens'],
            "token_opt4": tag_rep['opt4_tokens']

        }
        all_dqtag_data.append(dqtag_data)

    examples = []
    for i, (s1, s2, s3, s4, s5, s6, s7, s8), in enumerate(
            tqdm(zip(article, question, ct1, ct2, ct3, ct4, y, q_id), total=len(q_id), ncols=50, desc="reading...")):
        dqtag = all_dqtag_data[qas_id_to_tag_idx_map[s8]]
        doc_tokens = dqtag["token_doc"]
        assert dqtag["qas_id"] == s8

        sen_texts = simple_nlp.nlp(s1)
        sen_list = []
        for sen_ix, sent in enumerate(sen_texts.sents):
            sent_tokens = []
            prev_is_whitespace = True
            for c in sent.string:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        sent_tokens.append(c)
                    else:
                        sent_tokens[-1] += c
                    prev_is_whitespace = False
            sen_list.append((sen_ix, sent_tokens))

        cnt_token = 0
        new_sen_list = []
        flag = False
        tmp_token = ""
        for sen_ix, sent_tokens in sen_list:
            new_sent_tokens = []
            for tok_ix, token in enumerate(sent_tokens):
                if tok_ix == 0 and flag:
                    token = tmp_token + token
                    flag = False
                    tmp_token = ""
                    assert token == doc_tokens[cnt_token]

                if token != doc_tokens[cnt_token]:
                    assert tok_ix == len(sent_tokens) - 1
                    tmp_token = token
                    flag = True
                else:
                    assert token == doc_tokens[cnt_token]
                    new_sent_tokens.append(token)
                    cnt_token += 1
            new_sen_list.append(new_sent_tokens)

        span_doc = dqtag["span_doc"]
        head_doc = dqtag["head_doc"]
        type_doc = dqtag["type_doc"]
        assert len(span_doc) == len(head_doc) == len(type_doc) == cnt_token

        # reconstruct into sentences
        new_span_doc = []
        new_head_doc = []
        new_type_doc = []
        cnt = 0
        for sent_tokens in new_sen_list:
            new_span_sen = []
            new_head_sen = []
            new_type_sen = []
            for _ in sent_tokens:
                new_span_sen.append(span_doc[cnt])
                new_head_sen.append(head_doc[cnt])
                new_type_sen.append(type_doc[cnt])
                cnt += 1
            new_span_doc.append(new_span_sen)
            new_head_doc.append(new_head_sen)
            new_type_doc.append(new_type_sen)

        examples.append(
            SwagExample(
                swag_id=s8,
                context_sentence=s1,
                start_ending=s2,  # in the swag dataset, the
                # common beginning of each
                # choice is stored in "sent2".
                ending_0=s3,
                ending_1=s4,
                ending_2=s5,
                ending_3=s6,
                label=s7 if is_training else None,
                que_heads=dqtag["head_que"],
                que_types=dqtag["type_que"],
                que_span=dqtag["span_que"],
                doc_heads=new_head_doc,
                doc_types=new_type_doc,
                doc_span=new_span_doc,
                token_doc=new_sen_list,
                token_que=dqtag["token_que"],
                token_opt=[dqtag["token_opt1"], dqtag["token_opt2"], dqtag["token_opt3"], dqtag["token_opt4"]],
                opt_heads=[dqtag["head_opt1"], dqtag["head_opt2"], dqtag["head_opt3"], dqtag["head_opt4"]],
                opt_types=[dqtag["type_opt1"], dqtag["type_opt2"], dqtag["type_opt3"], dqtag["type_opt4"]],
                opt_span=[dqtag["span_opt1"], dqtag["span_opt2"], dqtag["span_opt3"], dqtag["span_opt4"]]
            )
        )

    return examples


def get_sub_spans(que_tokens, que_types, tokenizer, que_span):
    que_org_to_split_map = {}
    pre_tok_len = 0
    sub_que_types = []
    sub_que_span = []
    query_tokens = []

    assert len(que_tokens) == len(que_span)
    for idx, que_token in enumerate(que_tokens):
        sub_que_type_list = [que_types[idx]]
        sub_que_tok = tokenizer.tokenize(que_token)
        query_tokens.extend(sub_que_tok)
        while len(sub_que_type_list) != len(sub_que_tok):
            sub_que_type_list.append("subword")
        sub_que_types.extend(sub_que_type_list)
        que_org_to_split_map[idx] = (pre_tok_len, len(sub_que_tok) + pre_tok_len - 1)
        pre_tok_len += len(sub_que_tok)

    for idx, (start_ix, end_ix) in enumerate(que_span):
        head_start, head_end = que_org_to_split_map[idx]
        # sub_start_idx and sub_end_idx of children of head node
        head_spans = [(que_org_to_split_map[start_ix - 1][0], que_org_to_split_map[end_ix - 1][1])]
        # all other head sub_tok point to first head sub_tok
        if head_start != head_end:
            head_spans.append((head_start + 1, head_end))
            sub_que_span.append(head_spans)

            for i in range(head_start + 1, head_end + 1):
                sub_que_span.append([(i, i)])
        else:
            sub_que_span.append(head_spans)

    assert len(sub_que_span) == len(query_tokens)

    return sub_que_span


def convert_examples_to_features(examples, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""

    # Swag is a multiple choice task. To perform this task using Bert,
    # we will use the formatting proposed in "Improving Language
    # Understanding by Generative Pre-Training" and suggested by
    # @jacobdevlin-google in this issue
    # https://github.com/google-research/bert/issues/38.
    #
    # Each choice will correspond to a sample on which we run the
    # inference. For a given Swag example, we will create the 4
    # following inputs:
    # - [CLS] context [SEP] choice_1 [SEP]
    # - [CLS] context [SEP] choice_2 [SEP]
    # - [CLS] context [SEP] choice_3 [SEP]
    # - [CLS] context [SEP] choice_4 [SEP]
    # The model will output a single value for each input. To get the
    # final decision of the model, we will run a softmax over these 4
    # outputs.
    features = []
    for example_index, example in enumerate(tqdm(examples, ncols=50, desc="converting...")):
        que_types = example.que_types
        opt_types = example.opt_types
        doc_types = example.doc_types
        que_span = example.que_span
        opt_span = example.opt_span
        org_que_token = example.token_que
        org_doc_token = example.token_doc
        org_opt_token = example.token_opt
        all_doc_span = example.doc_span
        sub_que_spans = get_sub_spans(org_que_token, que_types, tokenizer, que_span)
        sub_opt_spans = [get_sub_spans(org_op_token, op_types, tokenizer, op_span) for org_op_token,
                                                                                       op_types, op_span in
                         zip(org_opt_token, opt_types, opt_span)]

        all_doc_types = [i for sen_type in doc_types for i in sen_type]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        all_sub_doc_types = []
        doc_tokens = [i for sen_token in org_doc_token for i in sen_token]
        for (i, token) in enumerate(doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            sub_doc_list = [all_doc_types[i]]
            while len(sub_doc_list) != len(sub_tokens):
                sub_doc_list.append("subword")

            all_sub_doc_types.extend(sub_doc_list)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        assert len(all_sub_doc_types) == len(all_doc_tokens)

        doc_org_to_split_map = {}
        pre_tok_len = 0
        for idx, doc_token in enumerate(doc_tokens):
            sub_doc_tok = tokenizer.tokenize(doc_token)
            doc_org_to_split_map[idx] = (pre_tok_len, len(sub_doc_tok) + pre_tok_len - 1)
            pre_tok_len += len(sub_doc_tok)

        cnt_span = 0
        for sen_idx, sen_span in enumerate(all_doc_span):
            for idx, (start_ix, end_ix) in enumerate(sen_span):
                assert (start_ix <= len(sen_span) and end_ix <= len(sen_span))
                cnt_span += 1
        # if not is_training:
        assert cnt_span == len(doc_tokens)

        sub_doc_span = []
        pre_sen_len = 0
        for sen_idx, sen_span in enumerate(all_doc_span):
            sen_offset = pre_sen_len
            pre_sen_len += len(sen_span)
            for idx, (start_ix, end_ix) in enumerate(sen_span):
                head_start, head_end = doc_org_to_split_map[sen_offset + idx]
                # sub_start_idx and sub_end_idx of children of head node
                head_spans = [(doc_org_to_split_map[sen_offset + start_ix - 1][0],
                               doc_org_to_split_map[sen_offset + end_ix - 1][1])]
                # all other head sub_tok point to first head sub_tok
                if head_start != head_end:
                    head_spans.append((head_start + 1, head_end))
                    sub_doc_span.append(head_spans)

                    for i in range(head_start + 1, head_end + 1):
                        sub_doc_span.append([(i, i)])
                else:
                    sub_doc_span.append(head_spans)

        assert len(sub_doc_span) == len(all_doc_tokens)

        # making masks
        que_span_mask = np.zeros((len(sub_que_spans), len(sub_que_spans)))
        for idx, span_list in enumerate(sub_que_spans):
            for (start_ix, end_ix) in span_list:
                if start_ix != end_ix:
                    que_span_mask[start_ix:end_ix + 1, idx] = 1

        [sub_opt1_span, sub_opt2_span, sub_opt3_span, sub_opt4_span] = sub_opt_spans

        opt1_span_mask = np.zeros((len(sub_opt1_span), len(sub_opt1_span)))
        for idx, span_list in enumerate(sub_opt1_span):
            for (start_ix, end_ix) in span_list:
                if start_ix != end_ix:
                    opt1_span_mask[start_ix:end_ix + 1, idx] = 1

        opt2_span_mask = np.zeros((len(sub_opt2_span), len(sub_opt2_span)))
        for idx, span_list in enumerate(sub_opt2_span):
            for (start_ix, end_ix) in span_list:
                if start_ix != end_ix:
                    opt2_span_mask[start_ix:end_ix + 1, idx] = 1

        opt3_span_mask = np.zeros((len(sub_opt3_span), len(sub_opt3_span)))
        for idx, span_list in enumerate(sub_opt3_span):
            for (start_ix, end_ix) in span_list:
                if start_ix != end_ix:
                    opt3_span_mask[start_ix:end_ix + 1, idx] = 1

        opt4_span_mask = np.zeros((len(sub_opt4_span), len(sub_opt4_span)))
        for idx, span_list in enumerate(sub_opt4_span):
            for (start_ix, end_ix) in span_list:
                if start_ix != end_ix:
                    opt4_span_mask[start_ix:end_ix + 1, idx] = 1

        opt_span_mask = [opt1_span_mask, opt2_span_mask, opt3_span_mask, opt4_span_mask]

        doc_span_mask = np.zeros((len(sub_doc_span), len(sub_doc_span)))

        for idx, span_list in enumerate(sub_doc_span):
            for (start_ix, end_ix) in span_list:
                if start_ix != end_ix:
                    doc_span_mask[start_ix:end_ix + 1, idx] = 1

        context_tokens = tokenizer.tokenize(example.context_sentence)
        assert len(sub_doc_span) == len(context_tokens)

        start_ending_tokens = tokenizer.tokenize(example.start_ending)
        assert len(sub_que_spans) == len(start_ending_tokens)

        choices_features = []
        for ending_index, ending in enumerate(example.endings):
            # We create a copy of the context tokens in order to be
            # able to shrink it according to ending_tokens
            context_tokens_choice = context_tokens + start_ending_tokens

            context_que_span_mask = np.zeros((len(context_tokens_choice), len(context_tokens_choice)))
            # print("===", len(context_tokens_choice))
            # 0 count for [CLS] and select_doc_len+1 count for [SEP]
            context_que_span_mask[0:len(context_tokens), 0:len(context_tokens)] = doc_span_mask
            context_que_span_mask[len(context_tokens):, len(context_tokens):] = que_span_mask

            ending_tokens = tokenizer.tokenize(ending)

            assert len(sub_opt_spans[ending_index]) == len(ending_tokens)

            option_len = len(ending_tokens)
            ques_len = len(start_ending_tokens)

            # ending_tokens = start_ending_tokens + ending_tokens

            # Modifies `context_tokens_choice` and `ending_tokens` in
            # place so that the total length is less than the
            # specified length.  Account for [CLS], [SEP], [SEP] with
            # "- 3"
            # ending_tokens = start_ending_tokens + ending_tokens
            idxa, idxb = _truncate_seq_pair(context_tokens_choice, ending_tokens, max_seq_length - 3)

            input_span_mask = np.zeros((max_seq_length, max_seq_length))
            # 0 count for [CLS] and select_doc_len+1 count for [SEP]

            input_span_mask[1:len(context_tokens_choice) + 1, 1:len(context_tokens_choice) + 1] = context_que_span_mask[
                                                                                                  idxa:, idxa:]
            input_span_mask[len(context_tokens_choice) + 2:len(context_tokens_choice) + 2 + len(ending_tokens),
            len(context_tokens_choice) + 2:len(context_tokens_choice) + 2 + len(ending_tokens)] = opt_span_mask[
                                                                                                      ending_index][
                                                                                                  idxb:, idxb:]

            record_mask = []
            for i in range(max_seq_length):
                i_mask = []
                for j in range(max_seq_length):
                    if input_span_mask[i, j] == 1:
                        i_mask.append(j)
                record_mask.append(i_mask)

            doc_len = len(context_tokens_choice) - ques_len

            tokens = ["[CLS]"] + context_tokens_choice + ["[SEP]"] + ending_tokens + ["[SEP]"]
            segment_ids = [0] * (len(context_tokens_choice) + 2) + [1] * (len(ending_tokens) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert (doc_len + ques_len + option_len) <= max_seq_length

            choices_features.append(
                (tokens, input_ids, input_mask, segment_ids, doc_len, ques_len, option_len, record_mask))

        label = example.label

        features.append(
            InputFeatures(
                example_id=example.swag_id,
                choices_features=choices_features,
                label=label
            )
        )

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    idx_a = list(range(len(tokens_a)))
    idx_b = list(range(len(tokens_b)))
    while True:

        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
            idx_a.pop(0)
        else:
            tokens_b.pop(0)
            idx_b.pop(0)

    return idx_a[0], idx_b[0]


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)


def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default='bert-base-cased', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default='output',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--output_file",
                        # default='output_batch4_gpu4_large_qo_lamda10_fp16.txt',
                        default='output_batch4_gpu4_match_pa_ap_cat_lamda00_fp16.txt',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--model_name",
                        default='output_batch4_gpu4_match_pa_ap_cat_lamda00_fp16.bin',
                        type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--train_file",
                        default='data/race/race_sample.json',
                        type=str)
    parser.add_argument("--dev_file",
                        default='dev.json',
                        type=str)
    parser.add_argument("--test_file",
                        default='test.json',
                        type=str)
    parser.add_argument("--train_tag_file",
                        default='data/race/race_span_sample.json',
                        type=str)
    parser.add_argument("--dev_tag_file",
                        default='output_race_dev.json',
                        type=str)
    parser.add_argument("--test_tag_file",
                        default='output_race_test.json',
                        type=str)
    parser.add_argument("--test_middle_tag_file",
                        default='output_race_testmiddle.json',
                        type=str)
    parser.add_argument("--test_high_tag_file",
                        default='output_race_testhigh.json',
                        type=str)

    parser.add_argument("--sa_file",
                        default='sample_nsp_train_dev.json',
                        type=str)
    parser.add_argument("--test_middle",
                        default='testmiddle.json',
                        type=str)
    parser.add_argument("--test_high",
                        default='testhigh.json',
                        type=str)
    parser.add_argument("--mid_high",
                        default=True,
                        help="Whether to run training.")
    parser.add_argument('--n_gpu',
                        type=int, default=2,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-6,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=50.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_sa",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--load_ft",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--do_lower_case",
                        default=True,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=4,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()
    logger.info(args)
    output_eval_file = os.path.join(args.output_dir, args.output_file)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(output_eval_file, "w") as writer:
        writer.write("%s\t\n" % args)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = args.n_gpu
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = args.n_gpu
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = BertForMultipleChoiceSpanMask.from_pretrained(args.bert_model,
                                                          cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                              args.local_rank),
                                                          num_choices=4)

    train_examples = None
    num_train_steps = None


    if args.do_train:
        train_examples = read_race_examples(args.train_file, args.train_tag_file,
                                            is_training=True)
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)


    if args.fp16:
        model.half()

    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    global_step = 0
    best_accuracy = 0

    if args.do_train:
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        with open(output_eval_file, "a") as writer:
            writer.write("***** Running training *****\t\n")
            writer.write("  Num examples = %d\t\n" % len(train_examples))
            writer.write("  Batch size = %d\t\n" % args.train_batch_size)
            writer.write("  Num steps = %d\t\n" % num_train_steps)

        all_input_ids = torch.tensor(select_field(train_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(train_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(train_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in train_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label, all_example_index)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        TrainLoss = []

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, ncols=50, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, example_index = batch

                input_span_mask = np.zeros((input_ids.size(0), input_ids.size(1), input_ids.size(2), input_ids.size(2)))
                for batch_idx, ex_idx in enumerate(example_index):
                    train_feature = train_features[ex_idx.item()]
                    choice_features = train_feature.choices_features
                    for idx, choice_fea in enumerate(choice_features):
                        train_span_mask = choice_fea["input_span_mask"]
                        for i, i_mask in enumerate(train_span_mask):
                            for j in i_mask:
                                input_span_mask[batch_idx, idx, i, j] = 1
                input_span_mask = torch.tensor(input_span_mask, dtype=torch.long)

                loss = model(input_ids, segment_ids, input_mask, label_ids,
                             input_span_mask=input_span_mask)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            TrainLoss.append(tr_loss/nb_tr_steps)
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(args.output_dir, "epoch_" + str(epoch) + "_pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

        with open(os.path.join(args.output_dir, "train_loss.pkl"), 'wb') as f:
            pickle.dump(TrainLoss, f)

    if args.do_eval:

        with open(os.path.join(args.output_dir, "train_loss.pkl"), 'rb') as f:
            TrainLoss = pickle.load(f)

        eval_examples = read_race_examples(args.test_file, args.test_tag_file,
                                           is_training=True)
        total_eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length)

        eval_features = total_eval_features
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor(select_field(eval_features, 'input_ids'), dtype=torch.long)
        all_input_mask = torch.tensor(select_field(eval_features, 'input_mask'), dtype=torch.long)
        all_segment_ids = torch.tensor(select_field(eval_features, 'segment_ids'), dtype=torch.long)
        all_label = torch.tensor([f.label for f in eval_features], dtype=torch.long)
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label,
                                  all_example_index)
        eval_sampler = SequentialSampler(eval_data)

        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        output_eval_file = os.path.join(args.output_dir, "result.txt")

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            train_loss = TrainLoss[epoch]
            output_model_file = os.path.join(args.output_dir, "epoch_" + str(epoch) + "_pytorch_model.bin")
            model_state_dict = torch.load(output_model_file)
            model = BertForMultipleChoiceSpanMask.from_pretrained(args.bert_model, state_dict=model_state_dict,
                                                                  num_choices=4)
            model.to(device)
            logger.info("Start evaluating")

            print("=======================")
            print("test_total...")
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for input_ids, input_mask, segment_ids, label_ids, example_index in eval_dataloader:
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                input_span_mask = np.zeros((input_ids.size(0), input_ids.size(1), input_ids.size(2), input_ids.size(2)))
                for batch_idx, ex_idx in enumerate(example_index):
                    total_eval_feature = total_eval_features[ex_idx.item()]
                    choice_features = total_eval_feature.choices_features
                    for idx, choice_fea in enumerate(choice_features):
                        span_mask = choice_fea["input_span_mask"]
                        for i, i_mask in enumerate(span_mask):
                            for j in i_mask:
                                input_span_mask[batch_idx, idx, i, j] = 1
                input_span_mask = torch.tensor(input_span_mask, dtype=torch.long)

                with torch.no_grad():
                    tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids,
                                          input_span_mask=input_span_mask)
                    logits = model(input_ids, segment_ids, input_mask, input_span_mask=input_span_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                tmp_eval_accuracy = accuracy(logits, label_ids)

                eval_loss += tmp_eval_loss.mean().item()
                eval_accuracy += tmp_eval_accuracy

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            eval_accuracy = eval_accuracy / nb_eval_examples

            result = {'eval_loss': eval_loss,
                      'best_accuracy': best_accuracy,
                      'eval_accuracy': eval_accuracy,
                      'global_step': global_step,
                      'loss': train_loss}

            with open(output_eval_file, "a") as writer:
                logger.info("***** Eval results *****")
                writer.write("\t\n***** Eval results Epoch %d  %s *****\t\n" % (
                    epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\t" % (key, str(result[key])))
                writer.write("\t\n")


if __name__ == "__main__":
    main()