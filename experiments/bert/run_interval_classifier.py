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

"""
This is a modified version of 'run_classifier.py' from HuggingFace's package.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import json

from word2number import w2n 
from sutime import SUTime
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

#test test
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, minmax=None, nor_val=None, haed=1):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
            minmax: (for interval head)
            nor_val: (current answer's normalized value)
            head: 1 for original head and 2 for interval head
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.minmax = minmax
        self.nor_val = nor_val
        self.head = head


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id,  minmax, nor_val, head):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.minmax = minmax
        self.nor_val = nor_val
        self.head = head


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class IntervalProcessor(DataProcessor):

    def __init__(self, jars_path):
        super(IntervalProcessor, self).__init__()
        self.sutime = SUTime(jars=jars_path, mark_time_ranges=True, include_range=True)
        self.convert_map = {
            **dict.fromkeys(["nanoseconds", "nanosecond"], 1.e-9),
            **dict.fromkeys(["seconds", "second"], 1.0),
            **dict.fromkeys(["minutes", "minute"], 60.0),
            **dict.fromkeys(["hours", "hour"], 60.0 * 60.0),
            **dict.fromkeys(["days", "day"], 24.0 * 60.0 * 60.0),
            **dict.fromkeys(["weeks", "week"], 7.0 * 24.0 * 60.0 * 60.0),
            **dict.fromkeys(["months", "month"], 30.0 * 24.0 * 60.0 * 60.0),
            **dict.fromkeys(["seasons", "season"], 3.0 * 30.0 * 24.0 * 60.0 * 60.0),
            **dict.fromkeys(["years", "year"], 365.0 * 24.0 * 60.0 * 60.0),
            **dict.fromkeys(["decades", "decade"], 10.0 * 365.0 * 24.0 * 60.0 * 60.0),
            **dict.fromkeys(["centuries", "century"], 100.0 * 365.0 * 24.0 * 60.0 * 60.0)
        }
        self.ans_errors, self.exp_errors = [], []

    def normalize(self, list_candidate_answers, normalize_type, category="Event Duration"):
        if category == "Event Duration":
            def filter(ans):
                ans = ans.replace(" later", "")
                pre_select = ["season", "nanosecond"]
                if any(x in ans for x in pre_select):
                    return ans
                else:
                    try:
                        return self.sutime.parse(ans)[0]['text']
                    except:
                        self.ans_errors.append(ans)
                        return None

            def get_trivial_floats(tokens):
                try:
                    try:
                        return float(" ".join(tokens).replace("," ,""))
                    except:
                        return w2n.word_to_num(" ".join(tokens))
                except:
                    return None
    
            def get_surface_floats(tokens):
                if tokens[-1] in ["a", "an"]:
                    return 1.0
                if tokens[-1] == "several":
                    return 4.0
                if tokens[-1] == "many":
                    return 10.0
                if tokens[-1] == "some":
                    return 3.0
                if tokens[-1] == "few":
                    return 3.0
                if tokens[-1] == "couple":
                    return 3.0    
                if tokens[-1] == "tens" or " ".join(tokens[-2:]) == "tens of":
                    return 10.0
                if tokens[-1] == "hundreds" or " ".join(tokens[-2:]) == "hundreds of":
                    return 100.0
                if tokens[-1] == "thousands" or " ".join(tokens[-2:]) == "thousands of":
                    return 1000.0
                if tokens[-1] == "millions" or " ".join(tokens[-2:]) == "millions of":
                    return 1000000.0
                if tokens[-1] == "billions" or " ".join(tokens[-2:]) == "billions of":
                    return 1000000.0
                if " ".join(tokens[-2:]) in ["a few", "a couple"]:
                    return 3.0
                if " ".join(tokens[-3:]) == "a couple of":
                    return 2.0
                return None
    
            def quantity(tokens):
                try:
                    if not tokens:
                        return 1
                    if get_trivial_floats(tokens) is not None:
                        return get_trivial_floats(tokens)
                    if get_surface_floats(tokens) is not None:
                        return get_surface_floats(tokens)
                    string_comb = tokens[-1]
                    cur = w2n.word_to_num(string_comb)
                    for i in range(-2, max(-(len(tokens)) - 1, -6), -1):
                        status = True
                        try:
                            _ = w2n.word_to_num(tokens[i])
                        except:
                            status = False
                        if tokens[i] in ["-", "and"] or status:
                            if tokens[i] != "-":
                                string_comb = tokens[i] + " " + string_comb
                            update = w2n.word_to_num(string_comb)
                            if update is not None:
                                cur = update
                        else:
                            break
                    if cur is not None:
                        return float(cur)
                except Exception as e:
                    return None 
    
            def exp2num(exp):
                try:
                    tokens = exp.split()
                    num = self.convert_map[tokens[-1]] * quantity(tokens[:-1])
                    return num
                except:
                    self.exp_errors.append(exp)
                    return None

            temporal_expressions = [filter(ans) for ans in list_candidate_answers]
            #temporal_values = np.array([exp2num(exp[0]['text']) for exp in temporal_expressions])
            temporal_values, masks = [], [] 
            results = np.array([None] * len(temporal_expressions))
            for i, exp in enumerate(temporal_expressions):
                value = exp2num(exp)
                if value: 
                    masks.append(i)
                    temporal_values.append(value)

            ##################################################
            # When number of normalizable answers are scarce #
            ##################################################
            if len(masks) < 2:
                return results

            temporal_values = np.log(np.array(temporal_values))

        # if category == "Typical Time":

        #     def exp2min(exp):
        #         h, m = exp.split(":")
        #         return float(h) * 60 + float(m)

        #     temporal_values = np.array([exp2min(exp[0]["value"][-5:]) for exp in temporal_expressions])

        if normalize_type == "minmax":
            min_v = min(temporal_values)
            max_v = max(temporal_values)
            temporal_values = (temporal_values - min_v) / (max_v - min_v)
        
        elif normalize_type == "meanstd":
            mean_v = np.mean(temporal_values)
            std_v = np.std(temporal_values)
            temporal_values = (temporal_values - mean_v) / std_v

        results[masks] = temporal_values
        return results

    def get_train_examples(self, data_dir):
        f = open(os.path.join(data_dir, "dev_3783.tsv"), "r")
        lines = [x.strip() for x in f.readlines()]
        examples = self._create_examples(lines, "train")
        return examples
        # sorted_example = sort_by_questsions(examples)
        # {
        #     'question1' : {
        #         'answers' : [1, 10 , 100 ]
        #         'labels' : [0, 1, 0]
        #     }
        # }
        # for question in sorted_example.keys():
        #     new_example, label = mixup(sorted_example[question])

        # example_w_pseudo = generate_psuedo_labels(sorted_example)
        # return example_w_pseudo

    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test_9442.tsv"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def get_labels(self):
        # set as none as there is no distinctive labe
        return ["yes", "no"]

    def _create_examples(self, lines, type):
      cat_filter = ["Event Duration"]

      questions = {}


      for (i, line) in enumerate(lines):
          group = line.split("\t")
          guid = "%s-%s" % (type, i)
          text_a = group[0] + " " + group[1]
          text_b = group[2]
          label = group[3]
          cat = group[4]
          if cat not in cat_filter:
            continue
          if text_a in questions:
            questions[text_a][0].append(text_b)
            questions[text_a][1].append(label)
          else:
            questions[text_a] = [[text_b],[label]]

      examples = []

      cur = False
      cur_normalize_result = None
      cur_q = None
      cur_minmax = None
      skip_cur_qa_interval = False

      for (i, line) in enumerate(lines):
          group = line.split("\t")
          guid = "%s-%s" % (type, i)
          text_a = group[0] + " " + group[1]
          text_b = group[2]
          label = group[3]
          cat = group[4]

          if cat not in cat_filter:
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
          elif skip_cur_qa_interval == True and cur_q == text_a:
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
          else:
              # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label,  ))
            cur_candidate = questions[text_a][0]
            cur_candidate_label = questions[text_a][1]
            # cur_idx = cur_candidate.index(text_b) 

            if "yes" not in cur_candidate_label or cur_candidate_label.count("yes") < 2:
              examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
            else:
              print("cur_candidate:")
              print(cur_candidate)
              print(cur_candidate_label)
              print("cur_idx")
              if cur_q != text_a:
                cur = False
                skip_cur_qa_interval = False
                cur_q = text_a


              if cur == False:
                cur = True
                normalize_result = self.normalize(cur_candidate,normalize_type ="minmax",category=cat)

                
                # to filiter out candidates that are not normalizable
                cur_normalize_result = {}
                filter_cur_candidate = []
                filter_cur_candidate_label = []
                filter_cur_normalize_result = []
                for i, j in enumerate(zip(cur_candidate, normalize_result)):
                  if j[1] == None:
                    continue
                  else:
                    cur_normalize_result[j[0]] = j[1]
                    filter_cur_candidate.append(cur_candidate[i])
                    filter_cur_candidate_label.append(cur_candidate_label[i])
                    filter_cur_normalize_result.append(normalize_result[i])

              
                if text_b not in cur_normalize_result:
                  examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                
                else:
                  cur_ans_normalize_result = cur_normalize_result[text_b]
                  print(cur_normalize_result)

                  yes_idxs = [i for i,j in enumerate(filter_cur_candidate_label) if j == 'yes']
                  yes_ans =  [j for i,j in enumerate(filter_cur_candidate) if i in yes_idxs]
                  yes_val =  [j for i,j in enumerate(filter_cur_normalize_result) if i in yes_idxs]

                  if len(yes_val) < 2:
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                    skip_cur_qa_interval = True
                  
                  else:
                  # sort to get min and max
                    yes_dict = dict(zip(yes_ans, yes_val))
                    yes_dict = {k: v for k, v in sorted(yes_dict.items(), key=lambda item: item[1])}  
                    
                    sort_ans = list(yes_dict.values())

                    cur_minmax = [sort_ans[0],sort_ans[-1]]
                    print("cur_minmaz")
                    print(cur_minmax)
                    examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, minmax= cur_minmax, nor_val=cur_ans_normalize_result, head=2))

                


              else:
                if text_b not in cur_normalize_result:
                  examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
                else:
                  cur_ans_normalize_result = cur_normalize_result[text_b]
                  examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, minmax= cur_minmax, nor_val=cur_ans_normalize_result, head=2))
            # print("cur_candidate:")
            # print(cur_candidate)
            # print(cur_candidate_label)
            # print("cur_idx")
            # print(cur_idx)
            # print("cur_ans")
            # print(text_b)
            # print("cur_cat")
            # print(cat)
            # print(label)
            # print("normalize_result")
            # print(normalize_result)
            # print("cur_ans_normalize_result")
            # print(cur_ans_normalize_result)
            # print("Yes ans")
            # print(yes_ans)
            # print("Yes val")
            # print(yes_val)
            # if i == 30:
            #   break
      return examples


class TemporalProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        f = open(os.path.join(data_dir, "dev_3783.tsv"), "r")
        lines = [x.strip() for x in f.readlines()]
        examples = self._create_examples(lines, "train")
        return examples
        # sorted_example = sort_by_questsions(examples)
        # {
        #     'question1' : {
        #         'answers' : [1, 10 , 100 ]
        #         'labels' : [0, 1, 0]
        #     }
        # }
        # for question in sorted_example.keys():
        #     new_example, label = mixup(sorted_example[question])

        # example_w_pseudo = generate_psuedo_labels(sorted_example)
        # return example_w_pseudo
        
    def get_dev_examples(self, data_dir):
        f = open(os.path.join(data_dir, "test_9442.tsv"), "r")
        lines = [x.strip() for x in f.readlines()]
        return self._create_examples(lines, "dev")

    def get_labels(self):
        return ["yes", "no"]

    def _create_examples(self, lines, type):
        examples = []
        for (i, line) in enumerate(lines):
            group = line.split("\t")
            guid = "%s-%s" % (type, i)
            text_a = group[0] + " " + group[1]
            text_b = group[2]
            label = group[3]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        minmax = example.minmax 
        nor_val = example.nor_val
        head = example.head

        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #             [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #             "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              minmax = minmax,
                              nor_val = nor_val,
                              head = head))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    prints = []
    for i in outputs:
        if i == 0:
            prints.append("yes")
        else:
            prints.append("no")
    return np.sum(outputs == labels), prints

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
    #################################
    # New argument
    #################################
    parser.add_argument("--sutime_jars_path",
                        default=None,
                        type=str,
                        require=True,
                        help="The path to jars folder required by sutime library")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
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
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    #############
    parser.add_argument('--interval_model',
                        default=False,
                        action='store_true',
                        help="Whether to replace the classification model with interval prediction model")
    # parser.add_argument('--loss_scale',
    #                     type=float, default=128,
    #                     help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    # parser.add_argument('--loss_scale',
    #                     type=float, default=128,
    #                     help='Loss scaling, positive power of 2 values can improve fp16 convergence.')
    

    args = parser.parse_args()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "temporal": TemporalProcessor,
        "interval" : IntervalProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

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

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name](args.sutime_jars_path)
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        if args.interval_model:
            import pdb; pdb.set_trace()        
            pass
            # train_exaples = processor.
        else:
            train_examples = processor.get_train_examples(args.data_dir)
            num_train_steps = int(
                len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    model = BertForSequenceClassification.from_pretrained(args.bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)

    global_step = 0
    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.float)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.float)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.float)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        all_minmax = torch.tensor(np.array([f.minmax for f in features],dtype=float), dtype=torch.float)
        all_nor_val = torch.tensor(np.array([f.nor_val for f in features],dtype=float), dtype=torch.float)
        all_head = torch.tensor([f.head for f in features], dtype=torch.float)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_minmax, all_nor_val, all_head)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, minmax_s, nor_val_s, heads = batch
                print(minmax_s)
                print(nor_val_s)
                print(heads)
                ### up untill here Sean 11/09

                import pdb; pdb.set_trace()
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_dev_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        total_prints = []
        for input_ids, input_mask, segment_ids, label_ids in eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            tmp_eval_accuracy, prints = accuracy(logits, label_ids)
            total_prints += prints

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'global_step': global_step,
                  'loss': tr_loss/nb_tr_steps}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        output_print_file = os.path.join(args.output_dir, "eval_outputs.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
        with open(output_print_file, "w") as writer:
            for l in total_prints:
                writer.write(l + "\n")

if __name__ == "__main__":
    main()