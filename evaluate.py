#!/usr/bin/env python
# !-*-coding:utf-8 -*-
import json
import numpy as np
import sys

sys.path.append("metric")
from metric.smooth_bleu import codenn_smooth_bleu
from metric.meteor.meteor import Meteor
from metric.rouge.rouge import Rouge
from metric.cider.cider import Cider
import warnings
import argparse
import logging
import prettytable as pt

warnings.filterwarnings('ignore')
logging.basicConfig(format='[%(asctime)s - %(levelname)s - %(name)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


def Commitbleus(refs, preds):

    r_str_list = []
    p_str_list = []
    for r, p in zip(refs, preds):
        if len(r[0]) == 0 or len(p) == 0:
            continue
        r_str_list.append([" ".join([str(token_id) for token_id in r[0]])])
        p_str_list.append(" ".join([str(token_id) for token_id in p]))
    try:
        bleu_list = codenn_smooth_bleu(r_str_list, p_str_list)
    except:
        bleu_list = [0, 0, 0, 0]
    codenn_bleu = bleu_list[0]

    B_Norm = round(codenn_bleu, 4)

    return B_Norm


def read_to_list(filename):
    f = open(filename, 'r',encoding="utf-8")
    res = []
    for row in f:
        # (rid, text) = row.split('\t')
        res.append(row.lower().split())
    return res

def metetor_rouge_cider(refs, preds):

    refs_dict = {}
    preds_dict = {}
    for i in range(len(preds)):
        preds_dict[i] = [" ".join(preds[i])]
        refs_dict[i] = [" ".join(refs[i][0])]
        
    score_Meteor, scores_Meteor = Meteor().compute_score(refs_dict, preds_dict)
    print("Meteor: ", round(score_Meteor*100,2))

    score_Rouge, scores_Rouge = Rouge().compute_score(refs_dict, preds_dict)
    print("Rouge-L: ", round(score_Rouge*100,2))

    score_Cider, scores_Cider = Cider().compute_score(refs_dict, preds_dict)
    print("Cider: ",round(score_Cider,2) )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refs_filename', type=str, default="../saved_model/tlcodesum/UNLC/ref.txt", required=False)
    parser.add_argument('--preds_filename', type=str, default="../saved_model/tlcodesum/UNLC/dlen500-clen30-dvoc30000-cvoc30000-bs-ddim64-cdim-rhs64-lr0_Medit_pred.txt", required=False)
    args = parser.parse_args()
    refs = read_to_list(args.refs_filename)
    refs = [[t] for t in refs]
    preds = read_to_list(args.preds_filename)
    bleus_score = Commitbleus(refs, preds)
    print("BLEU: %.2f"%bleus_score)
    metetor_rouge_cider(refs, preds)


if __name__ == '__main__':
    main()
