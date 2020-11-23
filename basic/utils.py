import editdistance
import numpy as np
import cv2
import torch

# Charset / labels conversion


def LM_str_to_ind(labels, str):
    return [labels.index(c) for c in str]


def LM_ind_to_str(labels, ind, oov_symbol=None):
    if oov_symbol is not None:
        res = []
        for i in ind:
            if i < len(labels):
                res.append(labels[i])
            else:
                res.append(oov_symbol)
    else:
        res = [labels[i] for i in ind]
    return "".join(res)


# OCR METRICS


def edit_cer_from_list(truth, pred):
    edit = 0
    for t, p in zip(truth, pred):
        edit += editdistance.eval(t, p)
    return edit


def edit_wer_from_list(truth, pred):
    edit = 0
    separation_marks = ["?", ".", ";", ",", "!", "\n"]
    for pred, gt in zip(pred, truth):
        for mark in separation_marks:
            gt.replace(mark, " {} ".format(mark))
            pred.replace(mark, " {} ".format(mark))
        gt = gt.split(" ")
        pred = pred.split(" ")
        while '' in gt:
            gt.remove('')
        while '' in pred:
            pred.remove('')
        edit += editdistance.eval(gt, pred)
    return edit


def nb_words_from_list(list_gt):
    separation_marks = ["?", ".", ";", ",", "!", "\n"]
    len_ = 0
    for gt in list_gt:
        for mark in separation_marks:
            gt.replace(mark, " {} ".format(mark))
        gt = gt.split(" ")
        while '' in gt:
            gt.remove('')
        len_ += len(gt)
    return len_


def nb_chars_from_list(list_gt):
    return sum([len(t) for t in list_gt])


def cer_from_list_str(str_gt, str_pred):
        len_ = 0
        edit = 0
        for pred, gt in zip(str_pred, str_gt):
            edit += editdistance.eval(gt, pred)
            len_ += len(gt)
        cer = edit / len_
        return cer


def wer_from_list_str(str_gt, str_pred):
    separation_marks = ["?", ".", ";", ",", "!", "\n"]
    len_ = 0
    edit = 0
    for pred, gt in zip(str_pred, str_gt):
        for mark in separation_marks:
            gt.replace(mark, " {} ".format(mark))
            pred.replace(mark, " {} ".format(mark))
        gt = gt.split(" ")
        pred = pred.split(" ")
        while '' in gt:
            gt.remove('')
        while '' in pred:
            pred.remove('')
        edit += editdistance.eval(gt, pred)
        len_ += len(gt)
    cer = edit / len_
    return cer


def cer_from_files(file_gt, file_pred):
        with open(file_pred, "r") as f_p:
            str_pred = f_p.readlines()
        with open(file_gt, "r") as f_gt:
            str_gt = f_gt.readlines()
        return cer_from_list_str(str_gt, str_pred)


def wer_from_files(file_gt, file_pred):
    with open(file_pred, "r") as f_p:
        str_pred = f_p.readlines()
    with open(file_gt, "r") as f_gt:
        str_gt = f_gt.readlines()
    return wer_from_list_str(str_gt, str_pred)
