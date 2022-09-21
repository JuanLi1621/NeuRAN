#coding:utf-8
from itertools import permutations
from collections import Counter
import numpy as np

head_in_dict = {}
head_out_dict = {} #(h,r)
tail_in_dict = {}
tail_out_dict = {}
train_data = []

def load_data(data_path):
    hd_max, hd_tot, tl_max, tl_tot = 0, 0, 0, 0
    head_out_dict_ratio = {} #{h:{r1:p1, r2:p2...}...}
    tail_in_dict_ratio = {}
    with open(data_path) as fin:
        for line in fin.readlines()[1:]:
            triple = line.strip().split()
            train_data.append(tuple(int(i) for i in triple)) #(h,t,r)

    for i in train_data:
        if i[0] not in head_out_dict:
            head_out_dict[i[0]] = []
        head_out_dict[i[0]].append(i[2])
        if i[1] not in tail_in_dict:
            tail_in_dict[i[1]] = [] 
        tail_in_dict[i[1]].append(i[2])

    head_out_dict_nums = {k: dict(Counter(np.array(v))) for k, v in head_out_dict.items()}
    for k in head_out_dict_nums:
        if (len(head_out_dict_nums[k].keys()))>hd_max:
            hd_max = len(head_out_dict_nums[k].keys())
        hd_tot += len(head_out_dict_nums[k].keys())
        rels_dict = head_out_dict_nums[k] #{k1:nums_1, k2:nums_2,...}
        rels_dict_prob = sorted({k1: v1/sum(rels_dict.values()) for k1, v1 in rels_dict.items()}.items(), key=lambda x:x[0]) #ascending, k1, [(k1,nums_1), (k2,nums_2)]
        head_out_dict_ratio[k] = rels_dict_prob

    tail_in_dict_nums = {k: dict(Counter(np.array(v))) for k, v in tail_in_dict.items()}
    for k in tail_in_dict_nums:
        if (len(tail_in_dict_nums[k].keys()))>tl_max:
            tl_max = len(tail_in_dict_nums[k].keys())
        tl_tot += len(tail_in_dict_nums[k].keys())
        rels_t = tail_in_dict_nums[k]
        rels_t_prob = sorted({k2: v2/sum(rels_t.values()) for k2, v2 in rels_t.items()}.items(), key=lambda x:x[0])
        tail_in_dict_ratio[k] = rels_t_prob
    
    head_out_dict_ratio = sorted(head_out_dict_ratio.items(), key=lambda x:x[0]) #[(k1,[(v1,p1),(v2,p2),...]),(k2,[...]),...]
    tail_in_dict_ratio = sorted(tail_in_dict_ratio.items(), key=lambda x:x[0])
    
    return hd_max, hd_tot, tl_max, tl_tot, head_out_dict_ratio, tail_in_dict_ratio

hd_max, hd_tot, tl_max, tl_tot, head_out_dict_ratio, tail_in_dict_ratio = load_data('./train2id.txt') 
# print(hd_max, tl_max, hd_tot, tl_tot)

with open('./ori_att_h_out_ranked.txt','w') as fin:
    fin.write(str(hd_tot)+'\n'+str(hd_max)+'\n')
    for k in head_out_dict_ratio: 
        for i in k[1]: 
            fin.write(str(k[0])+' '+str(i[0])+' '+str(i[1])+'\n')

with open('./ori_att_t_in_ranked.txt', 'w') as fin:
    fin.write(str(tl_tot)+'\n'+str(tl_max)+'\n')
    for k in tail_in_dict_ratio:
        for i in k[1]:
            fin.write(str(k[0])+' '+str(i[0])+' '+str(i[1])+'\n')
        

