import pickle
import os
import json
import prettytable as pt
import math
from torch.utils.data import TensorDataset
import numpy as np
import os
import random
import time
from tqdm import tqdm
import torch
import logging
logger = logging.getLogger(__name__)

REPLACE = '<REPLACE>'
REPLACE_OLD = '<REPLACE_OLD>'
REPLACE_NEW = '<REPLACE_NEW>'
REPLACE_END = '<REPLACE_END>'

INSERT = '<INSERT>'
INSERT_OLD = '<INSERT_OLD>'
INSERT_NEW = '<INSERT_NEW>'
INSERT_END = '<INSERT_END>'

DELETE = '<DELETE>'
DELETE_END = '<DELETE_END>'

KEEP = '<KEEP>'
KEEP_END = '<KEEP_END>'

def add_lang_by_task(target_str, task, sub_task):
    if task == 'summarize':
        target_str = '<en> ' + target_str
    elif task == 'refine':
        target_str = '<java> ' + target_str
    elif task == 'translate':
        if sub_task == 'java-cs':
            target_str = '<c_sharp> ' + target_str
        else:
            target_str = '<java> ' + target_str
    elif task == 'concode':
        target_str = '<java> ' + target_str
    elif task == 'defect':
        target_str = target_str
    return target_str


def convert_examples_to_features(item):
    example, example_index, tokenizer, args, stage = item

    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        if args.sub_task != 'none':
            source_str = "{} {}: {}".format(args.task, args.sub_task, example.source)
        else:
            source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    # https://blog.csdn.net/qq_33293040/article/details/105439750
    source_str = source_str.replace('</s>', '<unk>')
    source_ids = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    assert source_ids.count(tokenizer.eos_token_id) == 1
    if stage == 'test':
        target_ids = []
    else:
        target_str = example.target
        if args.add_lang_ids:
            target_str = add_lang_by_task(example.target, args.task, args.sub_task)
        if args.task in ['defect', 'clone']:
            if target_str == 0:
                target_str = 'false'
            elif target_str == 1:
                target_str = 'true'
            else:
                raise NameError
        target_str = target_str.replace('</s>', '<unk>')
        target_ids = tokenizer.encode(target_str, max_length=args.max_target_length, padding='max_length',
                                      truncation=True)
        assert target_ids.count(tokenizer.eos_token_id) == 1

    return InputFeatures(
        example_index,
        source_ids,
        target_ids,
        url=example.url
    )


def convert_clone_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
        target_str = "{}: {}".format(args.task, example.target)
    else:
        source_str = example.source
        target_str = example.target
    code1 = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    code2 = tokenizer.encode(target_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    source_ids = code1 + code2
    return CloneInputFeatures(example_index, source_ids, example.label, example.url1, example.url2)


def convert_defect_examples_to_features(item):
    example, example_index, tokenizer, args = item
    if args.model_type in ['t5', 'codet5'] and args.add_task_prefix:
        source_str = "{}: {}".format(args.task, example.source)
    else:
        source_str = example.source
    code = tokenizer.encode(source_str, max_length=args.max_source_length, padding='max_length', truncation=True)
    return DefectInputFeatures(example_index, code, example.target)


class CloneInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label,
                 url1,
                 url2
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label
        self.url1 = url1
        self.url2 = url2


class DefectInputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 label
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.label = label


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 url=None
                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.url = url


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 url=None,
                 task='',
                 sub_task=''
                 ):
        self.idx = idx
        self.source = source
        self.target = target
        self.url = url
        self.task = task
        self.sub_task = sub_task


class CloneExample(object):
    """A single training/test example."""

    def __init__(self,
                 code1,
                 code2,
                 label,
                 url1,
                 url2
                 ):
        self.source = code1
        self.target = code2
        self.label = label
        self.url1 = url1
        self.url2 = url2


def read_translate_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            src = line1.strip()
            trg = line2.strip()
            examples.append(
                Example(
                    idx=idx,
                    source=src,
                    target=trg,
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_refine_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0

    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_concode_examples(filename, data_num):
    """Read examples from filename."""
    examples = []

    with open(filename) as f:
        for idx, line in enumerate(f):
            x = json.loads(line)
            examples.append(
                Example(
                    idx=idx,
                    source=x["nl"].strip(),
                    target=x["code"].strip()
                )
            )
            idx += 1
            if idx == data_num:
                break
    return examples


def read_summarize_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['code_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['docstring_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_defect_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)

            code = ' '.join(js['func'].split())
            examples.append(
                Example(
                    idx=js['idx'],
                    source=code,
                    target=js['target']
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_clone_examples(filename, data_num):
    """Read examples from filename."""
    index_filename = filename
    url_to_code = {}
    with open('/'.join(index_filename.split('/')[:-1]) + '/data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            url_to_code[js['idx']] = code

    data = []
    with open(index_filename) as f:
        idx = 0
        for line in f:
            line = line.strip()
            url1, url2, label = line.split('\t')
            if url1 not in url_to_code or url2 not in url_to_code:
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append(CloneExample(url_to_code[url1], url_to_code[url2], label, url1, url2))
            idx += 1
            if idx == data_num:
                break
    return data


def load_and_cache_gen_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)

    examples = read_examples(filename, args.data_num, args.task)

    if is_sample:
        examples = random.sample(examples, min(args.n_debug_samples, len(examples)))
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    if os.path.exists(cache_fn) and not is_sample:
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 5k data for computing bleu from %s", filename)
        else:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
        features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        if split_tag == 'test' or only_src:
            data = TensorDataset(all_source_ids)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            data = TensorDataset(all_source_ids, all_target_ids)
        # if args.local_rank in [-1, 0] and not is_sample:
        #     torch.save(data, cache_fn)
    return examples, data


def read_commit_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = ' '.join(js['diff_tokens']).replace('\n', ' ')
            code = ' '.join(code.strip().split())
            nl = ' '.join(js['msg_tokens']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_plain_diff_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            # code = ' '.join(js['diff_tokens']).replace('\n', ' ')
            # code = ' '.join(code.strip().split())
            chunks = js["chunks"]
            plain_diff=""
            for chunk in chunks:
                plain_diff += " - " + " - ".join(chunk["old"]) + " + " + " + ".join(chunk["old"]) 

            nl = ' '.join(js['msg_token']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source= plain_diff,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_old_context_plain_diff_examples(filename, data_num, sep_token):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            # code = ' '.join(js['diff_tokens']).replace('\n', ' ')
            # code = ' '.join(code.strip().split())
            chunks = js["chunks"]
            plain_diff=""
            for chunk in chunks:
                plain_diff += " - " + " - ".join(chunk["old"]) + " + " + " + ".join(chunk["old"]) 
            old_verison = " ".join(js["old"])
            code_diff = old_verison + " " + sep_token +  " " + plain_diff
            nl = ' '.join(js['msg_token']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source= code_diff,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def read_medit_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            # code = ' '.join(js['diff_tokens']).replace('\n', ' ')
            # code = ' '.join(code.strip().split())
            chunks = js["chunks_diff"]
            medit= " "
            for chunk in chunks:
                medit += " ".join(chunk)

            nl = ' '.join(js['msg_token']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source= medit,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_contextual_medit_examples(filename, data_num):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            # code = ' '.join(js['diff_tokens']).replace('\n', ' ')
            # code = ' '.join(code.strip().split())
            code_diff = ' '.join( js["diff"])
            # chunks = js["chunks_diff"]
            # medit= " "
            # for chunk in chunks:
            #     medit += " ".join(chunk)

            nl = ' '.join(js['msg_token']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source= code_diff,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples


def read_old_context_medit_examples(filename, data_num, sep_token):
    """Read examples from filename."""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            # code = ' '.join(js['diff_tokens']).replace('\n', ' ')
            # code = ' '.join(code.strip().split())
            chunks = js["chunks_diff"]
            medit= " "
            for chunk in chunks:
                medit += " ".join(chunk)
            old_verison = " ".join(js["old"])
            code_diff = old_verison + " " + sep_token +  " " + medit

            nl = ' '.join(js['msg_token']).replace('\n', '')
            nl = ' '.join(nl.strip().split())
            examples.append(
                Example(
                    idx=idx,
                    source= code_diff,
                    target=nl,
                )
            )
            if idx + 1 == data_num:
                break
    return examples

def load_and_cache_commit_data(args, filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
    if is_sample:
        data_num = args.n_debug_samples
    else:
        data_num = args.data_num

    if args.diff_type in ["plain_diff_context"]:
        examples = read_commit_examples(filename, data_num)
    elif args.diff_type in ["plain_diff"]:
        examples =read_plain_diff_examples(filename,  data_num)
    elif args.diff_type in ["old-plain-diff"]:
        examples =read_old_context_plain_diff_examples(filename,  data_num, tokenizer.sep_token)
    elif args.diff_type in ["medit"]:
        examples =read_medit_examples(filename,  data_num)
    elif args.diff_type in ["old-medit"]:
        examples =read_old_context_medit_examples(filename,  data_num, tokenizer.sep_token)
    elif args.diff_type in ["contextual-medit"]:
        examples =read_contextual_medit_examples(filename,  data_num)
    else:
        raise RuntimeError("no such diff type")

    # if is_sample:
    #     # examples = random.sample(examples, min(args.n_debug_samples, len(examples)))
    #     examples = examples[:args.n_debug_samples]
    if split_tag == 'train':
        calc_stats(examples, tokenizer, is_tokenize=True)
    else:
        calc_stats(examples)
    # if os.path.exists(cache_fn) and not is_sample:
    #     logger.info("Load cache data from %s", cache_fn)
    #     data = torch.load(cache_fn)
    # else:
    if is_sample:
        logger.info("Sample some data for computing bleu from %s", filename)
    else:
        logger.info("Create cache data into %s", cache_fn)
    tuple_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(examples)]
    features = pool.map(convert_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
    all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
    if split_tag == 'test' or only_src:
        data = TensorDataset(all_source_ids)
    else:
        all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_target_ids)
        # if args.local_rank in [-1, 0] and not is_sample:
        #     torch.save(data, cache_fn)
    if args.debug:
        logger.info("*** Example ***")
        logger.info("idx: {}".format(examples[0].idx))

        logger.info("source_tokens: {}".format( examples[0].source ))
        logger.info("source_ids: {}".format(' '.join(map(str,  features[0].source_ids))))

        
        logger.info("target_tokens: {}".format(examples[0].target))
        logger.info("target_ids: {}".format(' '.join(map(str, features[0].target_ids))))
    
    return examples, data

def load_and_commit_data_with_retrieved_result(args, input_filename, retireved_filename, pool, tokenizer, split_tag, only_src=False, is_sample=False):
    # cache the data into args.cache_path except it is sampled
    # only_src: control whether to return only source ids for bleu evaluating (dev/test)
    # return: examples (Example object), data (TensorDataset)
    # data_tag = '_all' if args.data_num == -1 else '_%d' % args.data_num
    # cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + ('_src' if only_src else '') + data_tag)
    # if args.diff_type in ["plain_diff_context"]:
    #     examples = read_commit_examples(filename, args.data_num)
    # elif args.diff_type in ["plain_diff"]:
    #     examples =read_plain_diff_examples(filename,  args.data_num)
    # elif args.diff_type in ["old-plain-diff"]:
    #     examples =read_old_context_plain_diff_examples(filename,  args.data_num, tokenizer.sep_token)
    # elif args.diff_type in ["medit"]:
    #     examples =read_medit_examples(filename,  args.data_num)
    # elif args.diff_type in ["old-medit"]:
    #     examples =read_old_context_medit_examples(filename,  args.data_num, tokenizer.sep_token)
    if args.diff_type in ["contextual-medit"]:
        if is_sample:
            input_examples =read_contextual_medit_examples(input_filename,  args.n_debug_samples)
            retrieved_examples =read_contextual_medit_examples(retireved_filename,  args.n_debug_samples)
        else:
            if "dev" in split_tag:
                data_num = 2000
            else:
                data_num = args.data_num
            input_examples =read_contextual_medit_examples(input_filename,  data_num)
            retrieved_examples =read_contextual_medit_examples(retireved_filename, data_num)
    else:
        raise RuntimeError("no such diff type")

    if is_sample:
        input_examples = random.sample(input_examples, min(args.n_debug_samples, len(input_examples)))
        retrieved_examples = random.sample(retrieved_examples, min(args.n_debug_samples, len(input_examples)))
    # if split_tag == 'train':
    #     calc_stats(examples, tokenizer, is_tokenize=True)
    # else:
    #     calc_stats(examples)
    # if os.path.exists(cache_fn) and not is_sample:
    #     logger.info("Load cache data from %s", cache_fn)
    #     data = torch.load(cache_fn)
    # else:
    if is_sample:
        logger.info("Sample some data for computing bleu from %s", input_filename)
        logger.info("Sample some data for computing bleu from %s", retireved_filename)
  
    tuple_input_examples = [(example, idx, tokenizer, args, split_tag) for idx, example in enumerate(input_examples)]
    tuple_retrieved_examples = [(example, idx, tokenizer, args, "train") for idx, example in enumerate(retrieved_examples)]

    input_features = pool.map(convert_examples_to_features, tqdm(tuple_input_examples, total=len(tuple_input_examples)))
    retrieved_features = pool.map(convert_examples_to_features, tqdm(tuple_retrieved_examples, total=len(tuple_retrieved_examples)))

    all_input_source_ids = torch.tensor([f.source_ids for f in input_features], dtype=torch.long)

    all_retrieved_source_ids = torch.tensor([f.source_ids for f in retrieved_features], dtype=torch.long)
    all_retrieved_target_ids = torch.tensor([f.target_ids for f in retrieved_features], dtype=torch.long)

    if split_tag == 'test' or only_src:
        data = TensorDataset(all_input_source_ids, all_retrieved_source_ids,all_retrieved_target_ids)
    else:
        all_target_ids = torch.tensor([f.target_ids for f in input_features], dtype=torch.long)
        data = TensorDataset(all_input_source_ids, all_target_ids, all_retrieved_source_ids,all_retrieved_target_ids)
        # if args.local_rank in [-1, 0] and not is_sample:
        #     torch.save(data, cache_fn)
    if args.debug:
        logger.info("*** Example ***")
        logger.info("idx: {}".format(input_examples[0].idx))

        logger.info("source_tokens: {}".format( input_examples[0].source ))
        logger.info("source_ids: {}".format(' '.join(map(str,  input_features[0].source_ids))))

        logger.info("target_tokens: {}".format(input_examples[0].target))
        logger.info("target_ids: {}".format(' '.join(map(str, input_features[0].target_ids))))
    
        logger.info("retrieved_source_tokens: {}".format( retrieved_examples[0].source ))
        logger.info("retrieved_source_ids: {}".format(' '.join(map(str,  retrieved_features[0].source_ids))))

        logger.info("retrieved_target_tokens: {}".format(retrieved_examples[0].target))
        logger.info("retrieved_target_ids: {}".format(' '.join(map(str, retrieved_features[0].target_ids))))
    
    return input_examples, retrieved_examples, data

def load_and_cache_clone_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = '{}/{}.pt'.format(args.cache_path, split_tag + '_all' if args.data_num == -1 else '_%d' % args.data_num)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_clone_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def load_and_cache_defect_data(args, filename, pool, tokenizer, split_tag, is_sample=False):
    cache_fn = os.path.join(args.cache_path, split_tag)
    examples = read_examples(filename, args.data_num, args.task)
    if is_sample:
        examples = random.sample(examples, int(len(examples) * 0.1))

    calc_stats(examples, tokenizer, is_tokenize=True)
    if os.path.exists(cache_fn):
        logger.info("Load cache data from %s", cache_fn)
        data = torch.load(cache_fn)
    else:
        if is_sample:
            logger.info("Sample 10 percent of data from %s", filename)
        elif args.data_num == -1:
            logger.info("Create cache data into %s", cache_fn)
        tuple_examples = [(example, idx, tokenizer, args) for idx, example in enumerate(examples)]
        features = pool.map(convert_defect_examples_to_features, tqdm(tuple_examples, total=len(tuple_examples)))
        # features = [convert_clone_examples_to_features(x) for x in tuple_examples]
        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        data = TensorDataset(all_source_ids, all_labels)

        if args.local_rank in [-1, 0] and args.data_num == -1:
            torch.save(data, cache_fn)
    return examples, data


def get_filenames(data_root, task, sub_task, split=''):
    if task == 'concode':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.json'.format(data_dir)
        dev_fn = '{}/dev.json'.format(data_dir)
        test_fn = '{}/test.json'.format(data_dir)
    elif task == 'summarize':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    elif task == 'refine':
        data_dir = '{}/{}/{}'.format(data_root, task, sub_task)
        train_fn = '{}/train.buggy-fixed.buggy,{}/train.buggy-fixed.fixed'.format(data_dir, data_dir)
        dev_fn = '{}/valid.buggy-fixed.buggy,{}/valid.buggy-fixed.fixed'.format(data_dir, data_dir)
        test_fn = '{}/test.buggy-fixed.buggy,{}/test.buggy-fixed.fixed'.format(data_dir, data_dir)
    elif task == 'translate':
        data_dir = '{}/{}'.format(data_root, task)
        if sub_task == 'cs-java':
            train_fn = '{}/train.java-cs.txt.cs,{}/train.java-cs.txt.java'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.cs,{}/valid.java-cs.txt.java'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.cs,{}/test.java-cs.txt.java'.format(data_dir, data_dir)
        else:
            train_fn = '{}/train.java-cs.txt.java,{}/train.java-cs.txt.cs'.format(data_dir, data_dir)
            dev_fn = '{}/valid.java-cs.txt.java,{}/valid.java-cs.txt.cs'.format(data_dir, data_dir)
            test_fn = '{}/test.java-cs.txt.java,{}/test.java-cs.txt.cs'.format(data_dir, data_dir)
    elif task == 'clone':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.txt'.format(data_dir)
        dev_fn = '{}/valid.txt'.format(data_dir)
        test_fn = '{}/test.txt'.format(data_dir)
    elif task == 'defect':
        data_dir = '{}/{}'.format(data_root, task)
        train_fn = '{}/train.jsonl'.format(data_dir)
        dev_fn = '{}/valid.jsonl'.format(data_dir)
        test_fn = '{}/test.jsonl'.format(data_dir)
    if split == 'train':
        return train_fn
    elif split == 'dev':
        return dev_fn
    elif split == 'test':
        return test_fn
    else:
        return train_fn, dev_fn, test_fn


def read_examples(filename, data_num, task):
    read_example_dict = {
        'summarize': read_summarize_examples, 
        'refine': read_refine_examples,
        'translate': read_translate_examples,
        'concode': read_concode_examples,
        'clone': read_clone_examples,
        'defect': read_defect_examples,
    }
    return read_example_dict[task](filename, data_num)


def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def save_pickle_data(path_dir, filename, data):
    full_path = path_dir + '/' + filename
    print("Save dataset to: %s" % full_path)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    with open(full_path, 'wb') as output:
        pickle.dump(data, output,protocol=4)


def read_json_file(filename):
    with open(filename, 'r') as fp:
        data = fp.readlines()
    if len(data) == 1:
        data = json.loads(data[0])
    else:
        data = [json.loads(line) for line in data]
    return data

def save_json_data(data_dir, filename, data):
    os.makedirs(data_dir, exist_ok=True)
    file_name = os.path.join(data_dir, filename)
    with open(file_name, 'w') as output:
        if type(data) == list:
            if type(data[0]) in [str, list,dict]:
                for item in data:
                    output.write(json.dumps(item))
                    output.write('\n')

            else:
                json.dump(data, output)
        elif type(data) == dict:
            json.dump(data, output)
        else:
            raise RuntimeError('Unsupported type: %s' % type(data))
    print("saved dataset in " + file_name)

def percent_len(all_len,percentiles=None):
    if percentiles is None:
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ptiles_vers = list(np.percentile(all_len, np.array(percentiles)))
    ptiles_vers =[str(round(item,4)) for item in  ptiles_vers]
    tb = pt.PrettyTable()
    tb.field_names = ['mean'] + percentiles
    mean_value = round(np.mean(all_len), 1)
    tb.add_row([mean_value] + ptiles_vers)
    print(tb)
    latex_output = "& %.2f &"% float(mean_value)  + " &".join(ptiles_vers)
    print(latex_output)

def cal_r1_r5_r10(ranks):
    r1,r5,r10= 0,0,0
    data_len= len(ranks)
    for item in ranks:
        if item >=1:
            r1 +=1
            r5 += 1 
            r10 += 1
        elif item >=0.2:
            r5+= 1
            r10+=1
        elif item >=0.1:
            r10 +=1
    # print("& %.3f &%.3f &%.3f  "%(round(r1/data_len,4),  round(r5/data_len,4),   round(r10/data_len,4)))
    result = {"R@1":round(r1/data_len,3), "R@5": round(r5/data_len,3),  "R@10": round(r10/data_len,3)}
    return result

def time_format(time_cost):
    m, s = divmod(time_cost, 60)
    h, m = divmod(m, 60)
    # print("time_cost: %d" % (time_cost))
    return "%02d:%02d:%02d" % (h, m, s)

def array_split(original_data, core_num):
    data = []
    total_size = len(original_data)
    per_core_size = math.ceil(total_size / core_num)
    for i in range(core_num):
        lower_bound = i * per_core_size
        upper_bound = min((i + 1) * per_core_size, total_size)
        data.append(original_data[lower_bound:upper_bound])
    return data