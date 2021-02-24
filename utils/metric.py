#
#  MAKINAROCKS CONFIDENTIAL
#  ________________________
#
#  [2017] - [2020] MakinaRocks Co., Ltd.
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains
#  the property of MakinaRocks Co., Ltd. and its suppliers, if any.
#  The intellectual and technical concepts contained herein are
#  proprietary to MakinaRocks Co., Ltd. and its suppliers and may be
#  covered by U.S. and Foreign Patents, patents in process, and
#  are protected by trade secret or copyright law. Dissemination
#  of this information or reproduction of this material is
#  strictly forbidden unless prior written permission is obtained
#  from MakinaRocks Co., Ltd.
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils.normalize import Standardizer, Rotater, Truncater
import time


def get_norm(x, norm_type=2):
    return abs(x)**norm_type

def get_auc_roc(score, test_label, nap=False):
    try:
        fprs, tprs, threshold = metrics.roc_curve(test_label, score)
        if nap == True:
            print('auroc',metrics.auc(fprs, tprs))
            # print('=================fprs===============')
            # for i in fprs:
            #     print(i)
            #
            # print('=================tprs===============')
            # for i in tprs:
            #     print(i)
            # print('=================ends===============')
        return metrics.auc(fprs, tprs)
    except:
        return .0

def get_nap_auc_roc(score, test_label, nap=False):
    try:
        fprs, tprs, threshold = metrics.roc_curve(test_label, score)
        show = False
        if show:
            roc_auc = metrics.auc(fprs, tprs)
            plt.title('Receiver Operating Characteristic')
            plt.plot(fprs, tprs, 'b', label='AUC = %0.4f' % roc_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
        return metrics.auc(fprs, tprs)
    except:
        return .0
def get_threshold(precisions, recalls, threshold):
    show = False
    if show:
        plt.figure(figsize=(8, 5))
        plt.plot(threshold, precisions[1:], label='Precision')
        plt.plot(threshold, recalls[1:], label='Recall')
        plt.xlabel('Threshold')
        plt.ylabel('Precision/Recall')
        plt.legend()
        plt.show()
    # best position of threshold
    index_cnt = [cnt for cnt, (p, r) in enumerate(zip(precisions, recalls)) if p == r][0]
    print('precision: ', precisions[index_cnt], ', recall: ', recalls[index_cnt])

    # fixed Threshold
    threshold_fixed = threshold[index_cnt]
    print('threshold: ', threshold_fixed)
    return threshold_fixed

def get_confusion_matrix(score, test_label, threshold):
    score_label = []
    for i in score:
        if i >= threshold:
            score_label.append(True)
        else:
            score_label.append(False)

    tn, fp, fn, tp = metrics.confusion_matrix(test_label, score_label).ravel()
    print('Tn, Fp : '+str(tn)+', '+str(fp)+'\nFn, Tp : '+str(fn)+', '+str(tp))
    precision = tp/ (tp+fp)
    recall = tp/ (tp+fn)
    return precision, recall

def get_auc_prc(score, test_label):
    try:
        precisions, recalls, threshold = metrics.precision_recall_curve(test_label, score)
        # threshold = get_threshold(precisions, recalls, threshold)
        # precision, recall = get_confusion_matrix(score, test_label, threshold)
        show = False
        if show:
            pr_auc = metrics.auc(recalls, precisions)
            plt.title('Precision Recall Characteristic')
            plt.plot(recalls, precisions, 'b', label='AUC = %0.4f' % pr_auc)
            plt.legend(loc='lower right')
            plt.plot([0, 1], [0, 1], 'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('precisions')
            plt.xlabel('recalls')
            plt.show()
        return metrics.auc(recalls, precisions)
    except:
        return .0

def get_f1_score(valid_score, test_score, test_label, f1_quantiles=[.99]):

    f1_quantiles = 0.90 #added

    threshold = np.quantile(valid_score, f1_quantiles)
    predictions = test_score > threshold
    p = (predictions & test_label).sum() / float(predictions.sum())
    r = (predictions & test_label).sum() / float(test_label.sum())

    # print((predictions & test_label).sum(), predictions.sum(), test_label.sum())
    f1s = p * r * 2 / (p + r)
    # print(f1s)
    return f1s, threshold

def get_recon_loss(valid_diff, test_diff, test_label, f1_quantiles=[.99]):
    loss = (test_diff**2).mean(axis=1)
    loss_auc_roc = get_auc_roc(loss, test_label)
    loss_auc_prc = get_auc_prc(loss, test_label)
    loss_f1s, threshold = get_f1_score((valid_diff**2).mean(axis=1),
                            loss,
                            test_label,
                            f1_quantiles=f1_quantiles,
                            )
    precision, recall = get_confusion_matrix(loss, test_label, threshold)
    return loss, loss_auc_roc, loss_auc_prc, loss_f1s, precision, recall

def get_d_loss(train_diffs,
               valid_diffs,
               test_diffs,
               test_label,
               start_layer_index=0,
               end_layer_index=None,
               gpu_id=-1,
               norm_type=2,
               f1_quantiles=[.99]
               ):
    if end_layer_index is None:
        end_layer_index = len(test_diffs) + 1

    if start_layer_index > len(test_diffs) - 1:
        start_layer_index = len(test_diffs) - 1

    if end_layer_index - start_layer_index < 1:
        end_layer_index = start_layer_index + 1

    # valid_diffs[0].shape == 1282, 1728
    # [torch.from_numpy(i) for i in valid_diffs[start_layer_index:end_layer_index]][0].shape == 1282,1728
    # torch.cat([torch.from_numpy(i) for i in valid_diffs[start_layer_index:end_layer_index]], dim=-1).shape == 1282,5278
    valid_diffs = torch.cat([torch.from_numpy(i) for i in valid_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    test_diffs = torch.cat([torch.from_numpy(i) for i in test_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    # |test_diffs| = (batch_size, dim * 1+config.n_layers)

    d_loss = (test_diffs**2).mean(axis=1)
    d_loss_auc_roc = get_auc_roc(d_loss, test_label)
    d_loss_auc_prc = get_auc_prc(d_loss, test_label)
    d_loss_f1s, threshold = get_f1_score((valid_diffs**2).mean(axis=1),
                              d_loss,
                              test_label,
                              f1_quantiles=f1_quantiles
                             )
    precision, recall = get_confusion_matrix(d_loss, test_label, threshold)
    return d_loss, d_loss_auc_roc, d_loss_auc_prc, d_loss_f1s, precision, recall

def get_d_norm_loss(train_diffs,
                    valid_diffs,
                    test_diffs,
                    test_label,
                    config,
                    start_layer_index=0,
                    end_layer_index=None,
                    gpu_id=-1,
                    norm_type=2,
                    f1_quantiles=[.99]
                   ):

    if end_layer_index is None:
        end_layer_index = len(test_diffs) + 1

    if start_layer_index > len(test_diffs) - 1:
        start_layer_index = len(test_diffs) - 1

    if end_layer_index - start_layer_index < 1:
        end_layer_index = start_layer_index + 1

    train_diffs = torch.cat([torch.from_numpy(i) for i in train_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    torch.save(train_diffs, config.train_diffs)
    valid_diffs = torch.cat([torch.from_numpy(i) for i in valid_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()

    start_data = time.time()
    test_diffs = torch.cat([torch.from_numpy(i) for i in test_diffs[start_layer_index:end_layer_index]], dim=-1).numpy()
    sum = time.time() - start_data
    # |test_diffs| = (batch_size, dim * config.n_layers)

    rotater = Rotater()
    stndzer = Standardizer()

    rotater.fit(train_diffs, gpu_id=gpu_id)
    stndzer.fit(rotater.run(train_diffs, gpu_id=gpu_id))

    valid_rotateds = stndzer.run(rotater.run(valid_diffs, gpu_id=gpu_id))
    start_data = time.time()
    test_rotateds = stndzer.run(rotater.run(test_diffs, gpu_id=gpu_id))
    score = get_norm(test_rotateds, norm_type).mean(axis=1)
    sum += time.time() - start_data
    print('nap cal', sum)
    auc_roc = get_auc_roc(score, test_label, nap=True)


    auc_prc = get_auc_prc(score, test_label)

    f1_scores, threshold = get_f1_score(get_norm(valid_rotateds, norm_type).mean(axis=1),
                             score,
                             test_label,
                             f1_quantiles=f1_quantiles
                            )
    precision, recall = get_confusion_matrix(score, test_label, threshold)

    return score, auc_roc, auc_prc, f1_scores, precision, recall