import copy
import torch
from torch import nn
import numpy as np

def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg

def average_weights_contra(history):
    #copy the first client's weights
    w =  {client_id: history[client_id]['cshared_state_dict'] for client_id in history}
    reputation = {client_id: history[client_id]['reputation'] for client_id in history}
    learning_rate = {client_id: history[client_id]['learning_rate'] for client_id in history}
    sum_learning_rate = sum(learning_rate.values())        
    client = list(w.keys())[0]
    w_avg = {key: torch.zeros_like(w[client][key]) if 'num_batches_tracked' not in key else w[client][key] for key in w[client]}
    if sum_learning_rate == 0:
        return w_avg
    for k in w_avg.keys():  #the nn layer loop
        if 'num_batches_tracked' in k:
            continue
        for i in learning_rate:
            w_avg[k] += torch.mul(w[i][k], learning_rate[i])
        w_avg[k] /= sum_learning_rate
    return w_avg



def average_weights_contra_cloud(w, lr):
    #copy the first client's weights
    client = list(w.keys())[0]
    w_avg = {key: torch.zeros_like(w[client][key]) if 'num_batches_tracked' not in key else w[client][key] for key in w[client]}
    for k in w_avg.keys():  #the nn layer loop
        if 'num_batches_tracked' in k:
            continue
        for i in lr:
            w_avg[k] += w[i][k] * lr[i]
    return w_avg 


def average_weights_edge(history, s_num):
    #copy the first client's weights
    w =  {client_id: history[client_id]['cshared_state_dict'] for client_id in history}
    reputation = {client_id: history[client_id]['reputation'] for client_id in history}

    sample_num = list(s_num.values())
    total_sample_num = sum([s_num[id] * reputation[id] for id in s_num])
    client = list(w.keys())[0]
    w_avg = {key: 0 for key in w[client]}

    for k in w_avg.keys():  #the nn layer loop
        for i in s_num.keys():
            w_avg[k] += torch.mul(w[i][k], reputation[i] * s_num[i]/ total_sample_num)
    return w_avg


def median_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_median = copy.deepcopy(w[0])
    
    for k in w_median.keys():  #the nn layer loop
        tmp = [w[i][k] for i in range(len(w))]
        tmp = torch.stack(tmp).median(dim=0).values
        w_median[k] = tmp
    return w_median

