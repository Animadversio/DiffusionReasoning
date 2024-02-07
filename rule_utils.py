"""
Infer rules from attributes of an image. 

Shane Shang 
Jan. 2024
"""

import numpy as np
import torch
import einops

def get_XOR(x1, x2): 
    x3 = []
    for i in x1:
        if i not in x2: 
            x3.append(i)
    for j in x2: 
        if j not in x1: 
            x3.append(j)
    x3 = np.sort(x3)
    return list(x3)

def get_OR(x1, x2): 
    x3 = []
    for i in x1: 
        x3.append(i)
    for j in x2: 
        if j not in x3: 
            x3.append(j)
    x3 = np.sort(x3)
    return list(x3) 

def get_AND(x1, x2): 
    x3 = []
    for i in x1: 
        if i in x2: 
            x3.append(i)
    x3 = np.sort(x3)
    return list(x3) 

d_list = [0, -2, -1, 1, 2]

def check_R_unique(a_list):
    """a_list e.g., [1, 1, 1]"""
    a1, a2, a3 = a_list
    d1, d2 = a2-a1, a3-a2
    if d1 == d2: # constant or progression 
        if d1 in d_list: 
            i_R = d_list.index(d1)
        elif a3 == a1+a2: # arith pos 
            i_R = 5 
        elif a3 == a1-a2: # arith neg 
            i_R = 6 
        else: 
            i_R = np.nan
    else: 
        i_R = np.nan 
    return i_R 

def check_R_logical(attr_unique_row): 
    """attr_unique_row: list of len 3, unique attr values for each panel, e.g., [[0, 3], [1], [0, 5]]"""
    [x1, x2, x3] = attr_unique_row
    if x3 == get_XOR(x1, x2): 
        i_R = 7 
    elif x3 == get_XOR(x1, x2): 
        i_R = 8 
    elif x3 == get_AND(x1, x2): 
        i_R = 9 
    else: 
        i_R = np.nan 
    return i_R

def check_R(attr_unique_row): 

    if [len(x) for x in attr_unique_row] == [1, 1, 1]: # check for const, prog, arith 
        i_R = check_R_unique([x[0] for x in attr_unique_row])
    else: 
        i_R = check_R_logical(attr_unique_row)
    return i_R

def get_obj_list(attr):
    """
    for each panel, convert to list of objects: (shape, size, color, pos)
    Inputs: 
        attr: (3, 9, 3), # (num_panels, num_pos, num_attr)
    """
    attr_list_row = []
    pos_row = []

    for i_panel in range(3): 

        attr_panel = attr[i_panel] # torch.Size([9, 3])
        pos = ((attr_panel==-1).sum(1)==0).nonzero().squeeze(1)
        pos_row.append(list(pos.numpy()))

        attr_list = attr_panel[pos]
        attr_list_row.append(attr_list)
    return attr_list_row, pos_row

def get_rule_list(attr_list_row, pos_row): 
    """check for rules"""
    rule_list = []
    for i_a in range(3): # shape, constant, color
        attr_unique_row = [list(torch.unique(attr_list[:,i_a]).numpy()) for attr_list in attr_list_row]
        i_R = check_R(attr_unique_row)
        if not np.isnan(i_R): 
            rule_list.append(i_a*10+i_R)

    # check for number 
    num_row = [len(x) for x in attr_list_row]
    i_R = check_R_unique(num_row)
    if not np.isnan(i_R): 
        rule_list.append(30+i_R)
        
    # check for position 
    i_R = check_R_logical(pos_row)
    if not np.isnan(i_R): 
        rule_list.append(30+i_R)
        
    if len(rule_list) != 1: 
        rule = -1
    else: 
        rule = rule_list[0]
        
    return rule 

def get_rule_img(attr_tsr):
    """get rules list for an image of 3 rows"""
    row_num = attr_tsr.size(1)//3
    rule_all = []
    for i_row in range(row_num): 
        attr_row = attr_tsr[:,3*i_row:3*(i_row+1),:] # torch.Size([3, 3, 9])
        attr = einops.rearrange(attr_row, 'attr h (p w) -> p (h w) attr', h=3,w=3,p=3) # (3, 9, 3)
        attr_list_row, pos_row = get_obj_list(attr)
        rule = get_rule_list(attr_list_row, pos_row)
        rule_all.append(rule)
    return np.asarray(rule_all)

def check_consistent(rules):
    # if all 3 rows have the same rule, and it's not -1, return 1
    # if two rows have the same rule, and it's not -1, return 2
    # else, return 0
    rule_all = np.asarray(rules)
    rule_all = rule_all[rule_all != -1]
    if len(rule_all) == 0: # [-1, -1, -1]
        return 0
    elif len(np.unique(rule_all)) == 1 and len(rule_all) == 3: # three of a kind
        return 1
    # elif len(np.unique(rule_all)) == 1 and len(rule_all) == 2: # two of a kind, and the other is -1
    #     return 2
    elif len(np.unique(rule_all)) < len(rule_all): # two of a kind, [2, 1] or 2
        return 2
    else: 
        return 3 # anything else 
    