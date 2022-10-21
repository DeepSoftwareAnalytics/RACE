# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
import torch.nn as nn
import torch.nn.functional as F
import os
import logging
import argparse
import math
import numpy as np
# from tqdm import tqdm
import sys
import multiprocessing
import time
if sys.stderr.isatty():
    from tqdm import tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from model import   ECMGModel 
from metric import smooth_bleu
import random
from util import  get_elapse_time, load_and_cache_commit_data, load_and_commit_data_with_retrieved_result, save_json_data


from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


sys.path.append("data/commit_msg")
from util import REPLACE, REPLACE_OLD, REPLACE_NEW,REPLACE_END,INSERT,INSERT_OLD,INSERT_NEW ,INSERT_END,DELETE,DELETE_END,KEEP,KEEP_END



def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag,beam_size=1):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            # if args.model_type == 'roberta':
            #     preds = model(source_ids=source_ids, source_mask=source_mask)

            #     top_preds = [pred[0].cpu().numpy() for pred in preds]
            if hasattr(model, 'module'):
                preds = model.module.generate(source_ids,
                                       attention_mask=source_mask,
                                       use_cache=True,
                                       num_beams=beam_size,
                                       early_stopping=args.task == 'summarize',
                                       max_length=args.max_target_length)
                
            else:
                preds = model.generate(source_ids,
                                attention_mask=source_mask,
                                use_cache=True,
                                num_beams=beam_size,
                                early_stopping=args.task == 'summarize',
                                max_length=args.max_target_length)
            top_preds = list(preds.cpu().numpy())
            pred_ids.extend(top_preds)

    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]


    return  pred_nls 

def parse_args():
    parser = argparse.ArgumentParser()
    # debug
    parser.add_argument('--is_cosine_space', action='store_true', help='is_cosine_space', required=False)
    parser.add_argument('--debug', action='store_true', help='debug mode', required=False)
    parser.add_argument('--n_debug_samples', type=int, default=100, required=False)
    parser.add_argument("--eval_frequency", default=1, type=int, required=False)

    parser.add_argument("--ECMG_type", default="shared_encoders", choices=["shared_encoders"], type=str, required=False, help="the type of ECMG model")
    parser.add_argument("--base_model_type", default="codet5", choices=["codet5","codet5Siamese","ECMG"], type=str, required=False, help="the type of base model, like codet5, siamese network")
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'], help="the type of pretrain model")

    parser.add_argument("--diff_type", type=str,default="contextual-medit",
                        choices=['plain_diff',"old-plain-diff", "medit", "old-medit", "contextual-palin_diff", "contextual-medit"], 
                        help="plain_diff: only plain diff text; old-plain-diff: before and after three lines + plain diff \
                        medit ：using medit to represent plain diff; \
                        old-medit: old verison + medit, oversion is the before and after three lines + old diff , \
                        contextual-palin_diff: .diff; contextual-medit: " )
    
    parser.add_argument("--task", type=str,default="summarize",
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone'])
    # parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='java')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int, help="DATA_NUM == -1 means all data")
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--patience", default=5, type=int)
    # parser.add_argument("--tokenizer_path", default="tokenizer/salesforce", type=str)
    parser.add_argument("--cache_path", type=str, default="cache/codesum/java",required=False)
    parser.add_argument("--summary_dir", type=str, default="saved_model/codesum/tmp")
    # parser.add_argument("--data_dir", type=str, default="data/summarize/java")
    # parser.add_argument("--res_dir", type=str,default="saved_model/codesum/tmp")
    # parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", action='store_true')
    parser.add_argument("--always_save_model", action='store_true')
    parser.add_argument("--do_eval_bleu", action='store_true', help="Whether to evaluate bleu on dev set.")

   
    parser.add_argument("--model_name_or_path", default="Salesforce/codet5-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base,Salesforce/codet5-small,Salesforce/codet5-base")
    parser.add_argument("--output_dir",  type=str, default="../saved_model/commit_msg_generation/java/tmp",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--load_finetuned_model_path", default=None, type=str,
                        help="Path to fine tuned trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default="../data/commit_msg/java/contextual_medits/train.jsonl", type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default="../data/commit_msg/java/contextual_medits/valid.jsonl", type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default="../data/commit_msg/java/contextual_medits/test.jsonl", type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--train_retireved_filename", default="../data/commit_msg/java/contextual_medits/train.jsonl", type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_retireved_filename", default="../data/commit_msg/java/contextual_medits/valid.jsonl", type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_retireved_filename", default="../data/commit_msg/java/contextual_medits/test.jsonl", type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="Salesforce/codet5-small", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument('--do_retrieval', action='store_true', help='retrieval mode', required=False)
    parser.add_argument('--run_codet5', action='store_true', help='run codet5', required=False)
    parser.add_argument("--retrieval_filename", default="../data/commit_msg/java/contextual_medits/train.jsonl", type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--retrieval_result_dir", default="../data/commit_msg/java/tmp/codet5_retrieval_result", type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    parser.add_argument("--retrieval_result_filename", default="train.jsonl", type=str, 
                        help="The test filename. Should contain the .jsonl files for this task.")  
    

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=3407,
                        help="random seed for initialization")
    args = parser.parse_args()
    return args

def build_or_load_gen_model(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model = model_class.from_pretrained(args.model_name_or_path)

    special_tokens_dict = {'additional_special_tokens': [REPLACE, REPLACE_OLD, REPLACE_NEW,REPLACE_END,INSERT,INSERT_OLD,INSERT_NEW ,INSERT_END,DELETE,DELETE_END,KEEP,KEEP_END]}
    logger.info("adding new token %s"%str(special_tokens_dict))
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if args.load_finetuned_model_path is not None:
        logger.info("Reload fine tuned model from {}".format(args.load_finetuned_model_path))
        model.load_state_dict(torch.load(args.load_finetuned_model_path))
    if args.base_model_type == "codet5":
        pass
    elif  args.base_model_type == "ECMG":
        model = ECMGModel(model, config, args,sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    else:
        raise RuntimeError
    
    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    return config, model, tokenizer

def set_dist(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        # Setup for distributed data parallel
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    cpu_cont = multiprocessing.cpu_count()
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, cpu count: %d",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), cpu_cont)
    args.device = device
    args.cpu_cont = cpu_cont


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def main(args):

    t0 = time.time()

    set_dist(args)
    set_seed(args)
    config, model, tokenizer = build_or_load_gen_model(args)

    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    # args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_commit_data(args, args.train_filename, pool, tokenizer, 'train', is_sample=args.debug)
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                # print(step)
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)
            
                if args.base_model_type == "codet5" and args.model_type == 'codet5':
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss
                    
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    if sys.stderr.isatty():
                        bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
                if (step+1)% args.eval_frequency ==0 and (not sys.stderr.isatty()):
                    logger.info("epoch {} loss {}".format(cur_epoch,train_loss))
            if args.do_eval:
                eval_examples, eval_data = load_and_cache_commit_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                    only_src=True,  is_sample=args.debug)

                pred_nls = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev')
                output_fn = os.path.join(args.output_dir, "dev.output")
                gold_fn = os.path.join(args.output_dir, "dev.gold")
                dev_accs, predictions = [], []
                with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
                    for pred_nl, gold in zip(pred_nls, eval_examples):
                        dev_accs.append(pred_nl.strip() == gold.target.strip())
                        predictions.append(str(gold.idx) + '\t' + pred_nl)
                        f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    
                (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
                dev_bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s "%("codenn_bleu",str(dev_bleu)))
                logger.info("  "+"*"*20) 

                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)  

                if dev_bleu>best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

            # logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        model = model.module if hasattr(model, 'module') else model
        if args.load_model_path is not None:
            logger.info("reload model from {}".format(args.load_model_path))
            model.load_state_dict(torch.load(args.load_model_path))
        eval_examples, eval_data = load_and_cache_commit_data(args, args.test_filename, pool, tokenizer, 'test',
                                                            only_src=True, is_sample=args.debug)

        pred_nls = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test',beam_size=args.beam_size)
        output_fn = os.path.join(args.output_dir, "test.output")
        gold_fn = os.path.join(args.output_dir, "test.gold")
        dev_accs, predictions = [], []
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                predictions.append(str(gold.idx) + '\t' + pred_nl)
                f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
            
        (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
        dev_bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        logger.info("  %s = %s "%("codenn_bleu",str(dev_bleu)))

    if args.do_retrieval:
        logger.info("  " + "***** retrievaling *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        model = model.module if hasattr(model, 'module') else model
        train_examples, train_data = load_and_cache_commit_data(args, args.train_filename, pool, tokenizer, 'train', is_sample=args.debug)
        train_sampler = SequentialSampler(train_data) 
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size= args.eval_batch_size,
                                      num_workers=4, pin_memory=True)

        eval_examples, eval_data = load_and_cache_commit_data(args, args.retrieval_filename, pool, tokenizer, 'train',
                                                            only_src=True, is_sample=args.debug)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
        model.eval()

        model = model.module.encoder if hasattr(model, "moddule") else model.encoder

        train_code_vecs=[]
        eval_code_vecs=[]
        logger.info("  Num examples of Corpus = %d", len(train_data))
        for batch in train_dataloader:
            with torch.no_grad():
                source_ids = batch[0].to(args.device)
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                # [bs, hid_dim]
                train_code_vec = model(input_ids=source_ids, attention_mask=source_mask).last_hidden_state #[bs, sequence_length, dim]
                train_code_vec = torch.mean(train_code_vec, dim=1) #[bs, dim]
                if args.is_cosine_space:
                    train_code_vec = F.normalize(train_code_vec, p=2, dim=1)
                    
                train_code_vecs.append( train_code_vec.cpu().numpy()) 

        logger.info("  Num examples to retrieve = %d", len(eval_data))
        for batch in eval_dataloader:
            with torch.no_grad():
                # batch = tuple(t.to(args.device) for t in batch)
                source_ids = batch[0].to(args.device)
                source_mask = source_ids.ne(tokenizer.pad_token_id) 
                # [bs, 1, hid_dim]
                eval_code_vec = model(input_ids=source_ids, attention_mask=source_mask).last_hidden_state
                eval_code_vec = torch.mean(eval_code_vec, dim=1)
                if args.is_cosine_space:
                    eval_code_vec = F.normalize(eval_code_vec, p=2, dim=1)
                eval_code_vecs.append( eval_code_vec.cpu().numpy()) 

        train_code_vecs = np.concatenate(train_code_vecs,0) # [num_of_train_samples, hid_dim]
        eval_code_vecs = np.concatenate(eval_code_vecs,0) # [num_of_eval_samples, hid_dim]

        scores=np.matmul(eval_code_vecs, train_code_vecs.T)
        sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]   # [num_of_eval_samples,num_of_train_samples]
        if "train" in args.retrieval_result_filename:
            logger.info("return 2nd ranked result")
            rank1_result = sort_ids[:,1] # [num_of_eval_samples]
        else:
            rank1_result = sort_ids[:,0] # [num_of_eval_samples]
        logger.info("ranked list %s"%str(rank1_result[:30]))
        retrieval_results = []
        pred_nls = []
        for idx in rank1_result:
            retrieval_result = train_examples[idx]
            retrieval_results.append({ "diff":retrieval_result.source.split(),
                                       "msg_token":retrieval_result.target.split()})
            pred_nls.append(retrieval_result.target)
       
        predictions =[]
        output_fn = os.path.join(args.retrieval_result_dir, "%s.retireval.output"%args.retrieval_result_filename)
        gold_fn = os.path.join(args.retrieval_result_dir, "%s.gold"%args.retrieval_result_filename)                               
        with open(output_fn, 'w', encoding="utf-8") as f, open(gold_fn, 'w', encoding="utf-8") as f1:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                predictions.append(str(gold.idx) + '\t' + pred_nl)
                f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
            
        (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
        dev_bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        logger.info("  %s = %s "%("codenn_bleu",str(dev_bleu)))
        logger.info(" save predict result in %s"%(output_fn ))

        save_json_data(args.retrieval_result_dir, args.retrieval_result_filename, retrieval_results)
        
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()

def eval_ecmg_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag,beam_size=1):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model.eval()
    pred_ids = []
    bleu, codebleu = 0.0, 0.0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
        # source_ids = batch[0].to(args.device)
        # source_mask = source_ids.ne(tokenizer.pad_token_id)

        batch = tuple(t.to(args.device) for t in batch)
        input_source_ids, retrieved_source_ids, retrieved_target_ids = batch

        source_mask = input_source_ids.ne(tokenizer.pad_token_id)

        retrieved_source_mask = retrieved_source_ids.ne(tokenizer.pad_token_id)
        retrieved_target_mask = retrieved_target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            # if args.model_type == 'roberta':
                # preds = model(source_ids=source_ids, source_mask=source_mask)

                # top_preds = [pred[0].cpu().numpy() for pred in preds]
            # if hasattr(model, 'module'):
            #     preds = model.module.generate(source_ids,
            #                            attention_mask=source_mask,
            #                            use_cache=True,
            #                            num_beams=beam_size,
            #                            early_stopping=args.task == 'summarize',
            #                            max_length=args.max_target_length)
                
            # else:
            #     preds = model.generate(source_ids,
            #                     attention_mask=source_mask,
            #                     use_cache=True,
            #                     num_beams=beam_size,
            #                     early_stopping=args.task == 'summarize',
            #                     max_length=args.max_target_length)
            # top_preds = list(preds.cpu().numpy())
            # pred_ids.extend(top_preds)
            # preds = model(source_ids=source_ids, source_mask=source_mask)
            preds = model(source_ids=input_source_ids, source_mask=source_mask,
                                    retrieved_source_ids=retrieved_source_ids, retrieved_source_mask=retrieved_source_mask, 
                                    retrieved_target_ids=retrieved_target_ids, retrieved_target_mask=retrieved_target_mask
                                    )
            top_preds = [pred[0].cpu().numpy() for pred in preds]
            pred_ids.extend(top_preds)
    pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]


    return  pred_nls 


def ECMG(args):
    t0 = time.time()
    set_dist(args)
    set_seed(args)
    # config, model, tokenizer = build_or_load_gen_model(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    model = model_class.from_pretrained(args.model_name_or_path)

    special_tokens_dict = {'additional_special_tokens': [REPLACE, REPLACE_OLD, REPLACE_NEW,REPLACE_END,INSERT,INSERT_OLD,INSERT_NEW ,INSERT_END,DELETE,DELETE_END,KEEP,KEEP_END]}
    logger.info("adding new token %s"%str(special_tokens_dict))
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if args.load_finetuned_model_path is not None:
        logger.info("Reload fine tuned model from {}".format(args.load_finetuned_model_path))
        model.load_state_dict(torch.load(args.load_finetuned_model_path))

    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    # model = ECMGModel(model,decoder, config, args,sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)
    model = ECMGModel(model, config, args,sos_id=tokenizer.cls_token_id,eos_id=tokenizer.sep_token_id)

    if args.load_model_path is not None:
        logger.info("Reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))


    model.to(args.device)
    if args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    pool = multiprocessing.Pool(args.cpu_cont)
    # args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_path, exist_ok=True)
    fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        if args.local_rank in [-1, 0] and args.data_num == -1:
            summary_fn = '{}/{}'.format(args.summary_dir, '/'.join(args.output_dir.split('/')[1:]))
            tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_input_examples, train_retrieved_examples, train_data = load_and_commit_data_with_retrieved_result(args, args.train_filename, args.train_retireved_filename, pool, tokenizer, 'train', is_sample=args.debug)
        train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                # print(step)
                batch = tuple(t.to(args.device) for t in batch)
                input_source_ids, input_target_ids, retrieved_source_ids, retrieved_target_ids = batch

                source_mask = input_source_ids.ne(tokenizer.pad_token_id)
                target_mask = input_target_ids.ne(tokenizer.pad_token_id)

                retrieved_source_mask = retrieved_source_ids.ne(tokenizer.pad_token_id)
                retrieved_target_mask = retrieved_target_ids.ne(tokenizer.pad_token_id)

                # if args.model_type == 'roberta':
                #     loss, _, _ = model(source_ids=input_source_ids, source_mask=source_mask,
                #                        target_ids=input_target_ids, target_mask=target_mask)
                # else:
                #     outputs = model(input_ids=input_source_ids, attention_mask=source_mask,
                #                     labels=input_target_ids, decoder_attention_mask=target_mask)
                #     loss = outputs.loss

                loss, _, _  = model(source_ids=input_source_ids, source_mask=source_mask,
                                    target_ids=input_target_ids, target_mask=target_mask,
                                    retrieved_source_ids=retrieved_source_ids, retrieved_source_mask=retrieved_source_mask, 
                                    retrieved_target_ids=retrieved_target_ids, retrieved_target_mask=retrieved_target_mask
                                    )
                # loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += input_source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    if sys.stderr.isatty():
                        bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
                if (step+1)% args.eval_frequency ==0 and (not sys.stderr.isatty()):
                    logger.info("epoch {} loss {}".format(cur_epoch,train_loss))
            if args.do_eval:
                # eval_examples, eval_data = load_and_cache_commit_data(args, args.dev_filename, pool, tokenizer, 'dev',
                #                                                     only_src=True,  is_sample=args.debug)
                eval_examples, eval_retrieved_examples, eval_data = load_and_commit_data_with_retrieved_result(args, args.dev_filename, args.dev_retireved_filename, pool, tokenizer, 'dev', only_src=True, is_sample=args.debug)


                pred_nls = eval_ecmg_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev')
                output_fn = os.path.join(args.output_dir, "dev.output")
                gold_fn = os.path.join(args.output_dir, "dev.gold")
                dev_accs, predictions = [], []
                with open(output_fn, 'w', encoding="utf-8") as f, open(gold_fn, 'w', encoding="utf-8") as f1:
                    for pred_nl, gold in zip(pred_nls, eval_examples):
                        dev_accs.append(pred_nl.strip() == gold.target.strip())
                        predictions.append(str(gold.idx) + '\t' + pred_nl)
                        f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                    
                (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
                dev_bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
                logger.info("  %s = %s "%("codenn_bleu",str(dev_bleu)))
                logger.info("  "+"*"*20) 
                logger.info(" save predict result in %s"%(output_fn ))
                #save last checkpoint
                last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                if not os.path.exists(last_output_dir):
                    os.makedirs(last_output_dir)
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)  

                if dev_bleu>best_bleu:
                    logger.info("  Best bleu:%s",dev_bleu)
                    logger.info("  "+"*"*20)
                    best_bleu=dev_bleu
                    # Save best checkpoint for best bleu
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

            # logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()
        if args.local_rank in [-1, 0] and args.data_num == -1:
            tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)
        model = model.module if hasattr(model, 'module') else model
        if args.load_model_path is not None:
            logger.info("reload model from {}".format(args.load_model_path))
            model.load_state_dict(torch.load(args.load_model_path))
        # eval_examples, eval_data = load_and_cache_commit_data(args, args.test_filename, pool, tokenizer, 'test',
        #                                                     only_src=True, is_sample=args.debug)
        eval_examples, eval_retrieved_examples, eval_data = load_and_commit_data_with_retrieved_result(args, args.test_filename, args.test_retireved_filename, pool, tokenizer, 'test', only_src=True, is_sample=args.debug)

        pred_nls = eval_ecmg_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test',beam_size=args.beam_size)
        output_fn = os.path.join(args.output_dir, "test.output")
        gold_fn = os.path.join(args.output_dir, "test.gold")
        dev_accs, predictions = [], []
        with open(output_fn, 'w', encoding="utf-8") as f, open(gold_fn, 'w', encoding="utf-8") as f1:
            for pred_nl, gold in zip(pred_nls, eval_examples):
                dev_accs.append(pred_nl.strip() == gold.target.strip())
                predictions.append(str(gold.idx) + '\t' + pred_nl)
                f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
            
        (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
        dev_bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        logger.info("  %s = %s "%("codenn_bleu",str(dev_bleu)))
        logger.info(" save predict result in %s"%(output_fn ))

    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    fa.write("Finish and take {}".format(get_elapse_time(t0)))
    fa.close()

if __name__ == "__main__":
    args = parse_args()
    logger.info(args)
    if args.run_codet5:
        main(args)
    else:
        ECMG(args)
