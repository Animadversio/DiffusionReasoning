import pickle
import numpy as np
from collections import defaultdict
rule_dir = '/n/home12/binxuwang/Github/DiffusionReasoning/rule_data/'

with open(rule_dir+'r_dict_M7_withR.pkl', 'rb') as file:
    r_dict_M7_withR = pickle.load(file)
    
with open(rule_dir+'r_dict_M10_withR.pkl', 'rb') as file:
    r_dict_M10_withR = pickle.load(file)
    
with open(rule_dir+'r_dict_num_withR.pkl', 'rb') as file:
    r_dict_num_withR = pickle.load(file)
    
with open(rule_dir+'r_dict_pos_withR.pkl', 'rb') as file:
    r_dict_pos_withR = pickle.load(file)


def get_key_first2(a_list):
    """
    convert a_list to key string
    e.g., a_list = [[1,2], [1]], key='12-1'
    """
    key = ''
    for x in a_list[0]: 
        key+=str(x)
    key += '-'
    for x in a_list[1]: 
        key+=str(x) 
    return key

def get_attr(attr): 
    """attr.shape # (num_panels, num_obs, obj)"""
    # step 1: check number of objects 
    x_num = [[] for i in range(3)]
    x_pos = [[] for i in range(3)]
    x_shape = [[] for i in range(3)]
    x_color = [[] for i in range(3)]
    x_size = [[] for i in range(3)]

    for i, attr_panel in enumerate(attr): 

        pos = list(np.where((attr_panel==-1).sum(1)==0)[0])
        x_pos[i] = pos 
        x_num[i] = [len(pos)]
        x_shape[i] = list(np.unique(attr_panel[pos,0]))
        x_color[i] = list(np.unique(attr_panel[pos,1]))
        x_size[i] = list(np.unique(attr_panel[pos,2]))
        
    return x_num, x_pos, x_shape, x_color, x_size 

def check_rule(x, r_dict_withR): 
    i_R = None
    key_first2 = get_key_first2(x) # '5-5'
    if key_first2 in r_dict_withR.keys(): 
        d = r_dict_withR[key_first2]
        if x[2] in d.values():
            i_R = list(d.keys())[list(d.values()).index(x[2])]
    return i_R 

def check_row_rule(attr): 
    """attr: (3, 9, 3), return list of rule """
    x_num, x_pos, x_shape, x_color, x_size = get_attr(attr)
    
    rule_list = []
    R_shape = check_rule(x_shape, r_dict_M7_withR)
    if R_shape is not None: 
        rule_list.append(R_shape)

    R_color = check_rule(x_color, r_dict_M10_withR)
    if R_color is not None: 
        rule_list.append(R_color+10)

    R_size = check_rule(x_size, r_dict_M10_withR)
    if R_size is not None: 
        rule_list.append(R_size+20)
        
    R_num = check_rule(x_num, r_dict_num_withR)
    if R_num is not None: 
        rule_list.append(R_num+30)
        
    R_pos = check_rule(x_pos, r_dict_pos_withR)
    if R_pos is not None: 
        rule_list.append(R_pos+37)
        
    return np.asarray(rule_list)


def check_rule_overlap(attr_list): 
    """
    Inputs: 
        attr_list: (3, 3, 9, 3), (3 rows, 3 panels, 9 pos, 3 attr)
    Outputs: 
        r3: list of rules appearing in all 3 rows 
        r2: list of rules appear in only 2 of the 3 rows 
    """
    rule_all = [check_row_rule(a) for a in attr_list] # rule_list for each row, e.g., [[0,1], [0,1], [0,2]]
    
    r_dict = defaultdict(int) # e.g., {0: 3, 1: 2, 2: 1}, key=rule_ind, value=number of occurance in 3 rows
    for rule in rule_all:
        for x in rule: 
            if x not in r_dict.keys(): 
                r_dict[x] = 0 
            r_dict[x] += 1 

    r3, r2 = [], []
    for k, v in r_dict.items():
        if v == 3:
            r3.append(k)
        elif v == 2:
            r2.append(k)
    
    return r3, r2, rule_all

def check_r3_r2_batch(attr_sample): 
    """
    Inputs: 
        attr_sample: e.g., (4000, 3, 3, 9, 3), (num_samples, 3 rows, 3 panels, 9 pos, 3 attrs)
    Outputs: 
        r3_all: list, rule that appear in all 3 rows for each sample 
        r2_all: list, rule that appear in 2 of 3 rows 
        
    TODO: multiple CPU cores? concurrent.futures
    """
    r3_all = []
    r2_all = []
    rule_collector = []
    for attr_list in attr_sample: 
        r3, r2, rule_all = check_rule_overlap(attr_list)

        r3_all.append(r3)
        r2_all.append(r2)
        rule_collector.append(rule_all)
        
    return r3_all, r2_all, rule_collector


import einops
def infer_rule_from_sample_batch(sample_batch):
    # if not int convert to int
    sample_batch = sample_batch.round().int()
    sample_batch = sample_batch.view(-1, 3, 3, 3, 9) 
    sample_batch = einops.rearrange(sample_batch, 
        "B attr row h (panel w) -> B row panel (h w) attr", 
        panel=3, w=3, h=3, attr=3)
    r3_list, r2_list, rule_col = check_r3_r2_batch(sample_batch)
    return r3_list, r2_list, rule_col