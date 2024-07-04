import numpy as np 
import random
import torch

#################### PGM ####################
pos_list = [[20,20], [20,60], [20,100], [60,20], [60,60], [60,100], [100,20], [100,60], [100,100]]
# d_PGM = torch.load('data/20240308_data/PGM_shape_size_color_normalized.pt') # torch.Size([7, 10, 10, 40, 40])
d_PGM = torch.load('/n/home12/binxuwang/Github/DiffusionReasoning/rule_data/20240308_data/PGM_shape_size_color_normalized.pt') # torch.Size([7, 10, 10, 40, 40])

def load_PGM_inputs(attr): 
    """attr: (3, 9, 3), (num_panel, num_pos, num_attr)"""
    inputs = -0.6891*torch.ones((3, 160, 160))
    for i_panel in range(3): 
        for i_pos in range(9): 
            if attr[i_panel, i_pos, 0] != -1: 
                i_shape, i_size, i_color = attr[i_panel, i_pos]
                x0, y0 = pos_list[i_pos]
                inputs[i_panel, x0:(x0+40), y0:(y0+40)] = d_PGM[int(i_shape), int(i_size), int(i_color)]
    return inputs 

def get_XOR(x1, x2):
    """x3 has items that only appear in one of x1 or x2"""
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
    """x3 is the union of x1+x2"""
    x3 = []
    for i in x1: 
        x3.append(i)
    for j in x2: 
        if j not in x3: 
            x3.append(j)
    x3 = np.sort(x3)
    return list(x3) 

def get_AND(x1, x2): 
    """x3 is the intersect of x1 and x2"""
    x3 = []
    for i in x1: 
        if i in x2: 
            x3.append(i)
    x3 = np.sort(x3)
    return list(x3) 

from itertools import combinations 
def get_list_combinations(M): 
    """get all combinations of M items"""
    list_combinations = list()
    L = list(np.arange(M))
    for n in range(1, M+1): 
        list_combinations += list(combinations(L, n))
    return [list(x) for x in list_combinations]

def get_r_list(M, n_max): 
    """"""
    # rel 0: constant 
    r_const_list = []
    for i in range(1,M):
        r_const_list.append([[i], [i], [i]])
    # take out [0,0,0] because ambigurous (also can count as arith)

    # rel 1: prog neg 2
    r_prog_neg2_list = []
    for i in range(4, M):
        if i!=6: 
            r_prog_neg2_list.append([[i], [i-2], [i-4]])

    # rel 2: prog neg 1
    r_prog_neg1_list = []
    for i in range(2, M):
        if i!=3:  
            r_prog_neg1_list.append([[i], [i-1], [i-2]])

    # rel 3: prog pos 1
    r_prog_pos1_list = []
    for i in range(M-2):
        if i!=1:  
            r_prog_pos1_list.append([[i], [i+1], [i+2]])

    # rel 4: prog pos 2
    r_prog_pos2_list = []
    for i in range(M-4):
        if i!=2: 
            r_prog_pos2_list.append([[i], [i+2], [i+4]])

    # rel 5: arith pos
    r_arith_pos_list = []
    for i in range(M): 
        for j in range(1,M-i): 
            if (i==0 and j==0) or (i==1 and j==2) or (i==2 and j==4):
                pass
            else: 
                r_arith_pos_list.append([[i], [j], [i+j]])

    # rel 6: arith neg 
    r_arith_neg_list = []
    for i in range(M): 
        for j in range(1,i+1): 
            if (i==0 and j==0) or (i==3 and j==2) or (i==6 and j==4): 
                pass
            else: 
                r_arith_neg_list.append([[i], [j], [i-j]])

    # rel 7,8,9: XOR, OR, AND
    list_combinations = get_list_combinations(M)

    r_XOR_list = []
    r_OR_list = []
    r_AND_list = []

    for x1 in list_combinations: 
        for x2 in list_combinations: 

            x_XOR = get_XOR(x1, x2)
            x_OR = get_OR(x1, x2)
            x_AND = get_AND(x1, x2)

            if len(x_AND)>0 and len(x_XOR)>0 and len(x_OR)<n_max:

                # check 
                if (x_XOR==x_OR) or (x_XOR==x_OR) or (x_XOR==x_OR):
                    print('Error: '+str(x1)+' -- '+str(x2))
                else: 
                    r_XOR_list.append([x1, x2, x_XOR])
                    r_OR_list.append([x1, x2, x_OR])
                    r_AND_list.append([x1, x2, x_AND])

    r_all = [r_const_list, r_prog_neg2_list, r_prog_neg1_list, r_prog_pos1_list, r_prog_pos2_list, 
            r_arith_pos_list, r_arith_neg_list, r_XOR_list, r_OR_list, r_AND_list]
    r_all = np.array(r_all, dtype=object)
    return r_all

def get_r_list_num(M=9): 
    """M=9, for x_num, single value rules only"""
    # rel 0: constant, take out [1, 1, 1], [9,9,9] because no pos rule 
    r_const_list = []
    for i in range(2,M):
        r_const_list.append([[i], [i], [i]])
        
    # rel 1: prog neg 2
    r_prog_neg2_list = []
    for i in range(5, M+1):
        if i!=6: 
            r_prog_neg2_list.append([[i], [i-2], [i-4]])
            
    # rel 2: prog neg 1
    r_prog_neg1_list = []
    for i in range(4, M+1):
        r_prog_neg1_list.append([[i], [i-1], [i-2]])
        
    # rel 3: prog pos 1
    r_prog_pos1_list = []
    for i in range(1, M-1):
        if i!=1:  
            r_prog_pos1_list.append([[i], [i+1], [i+2]])
            
    # rel 4: prog pos 2
    r_prog_pos2_list = []
    for i in range(1, M-3):
        if i!=2: 
            r_prog_pos2_list.append([[i], [i+2], [i+4]])
            
    # rel 5: arith pos
    r_arith_pos_list = []
    for i in range(1, M+1): 
        for j in range(1,M-i+1): 
            if (i==1 and j==1) or (i==1 and j==2) or (i==2 and j==4) or (i==3 and j==6):
                pass
            else: 
                r_arith_pos_list.append([[i], [j], [i+j]])
                
    # rel 6: arith neg 
    r_arith_neg_list = []
    for i in range(1, M+1): 
        for j in range(1,i): 
            if (i==3 and j==2) or (i==6 and j==4) or (i==9 and j==6): 
                pass
            else: 
                r_arith_neg_list.append([[i], [j], [i-j]])
                
    r_all = [r_const_list, r_prog_neg2_list, r_prog_neg1_list, r_prog_pos1_list, r_prog_pos2_list, 
            r_arith_pos_list, r_arith_neg_list]
    r_all = np.array(r_all, dtype=object)
    return r_all

def get_r_list_pos(M): 
    # rel 7,8,9: XOR, OR, AND
    list_combinations = get_list_combinations(M)

    r_XOR_list = []
    r_OR_list = []
    r_AND_list = []

    for x1 in list_combinations: 
        for x2 in list_combinations: 

            x_XOR = get_XOR(x1, x2)
            x_OR = get_OR(x1, x2)
            x_AND = get_AND(x1, x2)

            if len(x_AND)>0 and len(x_XOR)>0 and len(x_OR)<=9:

                # check 
                if (x_XOR==x_OR) or (x_XOR==x_OR) or (x_XOR==x_OR):
                    print('Error: '+str(x1)+' -- '+str(x2))
                else: 
                    r_XOR_list.append([x1, x2, x_XOR])
                    r_OR_list.append([x1, x2, x_OR])
                    r_AND_list.append([x1, x2, x_AND])

    r_all = [r_XOR_list, r_OR_list, r_AND_list]
    r_all = np.array(r_all, dtype=object)
    return r_all

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

def get_key_numattr(a_list): 
    """e,g., [[1], [1], [1]] --> '111'"""
    return str(len(a_list[0]))+str(len(a_list[1]))+str(len(a_list[2]))

"""
def get_r_dict(r_all): 
    e.g., [[[[1], [1], [1]]]] --> {'111': {'1-1': [[1]]}}
    num_attr, first2 panels

    r_dict = {}
    for r_list in r_all: 
        for x in r_list: 
            key_numattr = get_key_numattr(x)
            key_first2 = get_key_first2(x)
            if key_numattr not in r_dict.keys(): 
                r_dict[key_numattr] = {}
            if key_first2 not in r_dict[key_numattr].keys(): 
                r_dict[key_numattr][key_first2] = []
            r_dict[key_numattr][key_first2].append(x[2])
    return r_dict
"""

def get_r_dict(r_all): 
    """ e.g., [[1], [1], [1]] --> {'111': {'1-1': [[1]]}}
    num_attr, first2 panels
    """
    r_dict = {}
    for r_list in r_all: 
        for x in r_list: 
            key_first2 = get_key_first2(x)
            if key_first2 not in r_dict.keys(): 
                r_dict[key_first2] = []
            r_dict[key_first2].append(x[2])

    return r_dict

def get_r_dict_withR(r_all): 
    """ e.g., [[1], [1], [1]] --> {'111': {'1-1': [[1]]}}
    num_attr, first2 panels
    """
    r_dict = {}
    for i_R, r_list in enumerate(r_all): 
        for x in r_list: 
            key_first2 = get_key_first2(x)
            if key_first2 not in r_dict.keys(): 
                r_dict[key_first2] = {}
                
            r_dict[key_first2][i_R] = x[2]

    return r_dict

def get_r_last_all(r_all): 
    """
    key: num_attr in last panel
    e.g., [[[[1], [1], [1]]]] --> {1: [[1]]}
    """
    r_last = {}
    for r_list in r_all: 
        for x in r_list: 
            key = len(x[2])
            
            if key not in r_last.keys(): 
                r_last[key] = []
            if x[2] not in r_last[key]:
                r_last[key].append(x[2])
    return r_last 

def get_r_dict_num(r_all):
    """
    specifically for number, key: first2 panel num
    e.g., [[[[1], [1], [1]]]] --> {'22': [2]}
    """
    r_dict_num = {}
    for r_list in r_all: 
        for x in r_list: 
            key_first2 = str(x[0][0])+str(x[1][0])

            if key_first2 not in r_dict_num.keys(): 
                r_dict_num[key_first2] = []
            r_dict_num[key_first2].append(x[2][0])
    return r_dict_num

def get_r_dict_pos(r_all): 
    """
    e.g., r_dict_pos['12'][1]: [ [[0], [0, 1], [1]], ...], rows with num_obj '12' in first 2 panel and 1 in last 
    """
    r_dict_pos = {}
    for r_list in r_all: 
        for x in r_list: 
            key = str(len(x[0]))+str(len(x[1]))

            if key not in r_dict_pos.keys(): 
                r_dict_pos[key] = {}

            l = len(x[2])
            if l not in r_dict_pos[key].keys(): 
                r_dict_pos[key][l] = []

            r_dict_pos[key][l].append(x)
    return r_dict_pos

def sample_nan(r_all, r_dict, r_last):
    """
    sample an attribute value, first2 belong to some rule but third panel is changed
    additionally, the third panel belong to some rule row
    """
    a = random.choice(random.choice(r_all)) # start with a row of some rule 
    x = a.copy()
    key_first2 = get_key_first2(x) # e.g., '12-1'
    
    d = r_dict[key_first2]
    potential_last = r_last[len(x[2])]
    x_new_last = random.choice(potential_last)
    while x_new_last in d: 
        x_new_last = random.choice(potential_last)
    x[2] = x_new_last
    return x

def sample_x_with_num(x_num, r_all, r_dict, r_last):
    """
    sample x with less than x_num e.g., [[2], [2], [2]]
    """
    x = random.choice(random.choice(r_all))
    while len(x[0])>x_num[0][0] or len(x[1])>x_num[1][0] or len(x[2])>x_num[2][0]: 
        x = random.choice(random.choice(r_all))
        
    # now, replace the last 
    key_first2 = get_key_first2(x) # e.g., '12-1'
    d = r_dict[key_first2]
    potential_last = r_last[len(x[2])]
    x_new_last = random.choice(potential_last)
    while x_new_last in d: 
        x_new_last = random.choice(potential_last)
    x[2] = x_new_last
    return x

def fill_panel_attr(pos_id, value): 
    """
    Fill panel for a single attribute 
    
    Args: 
        pos_id: [0, 1, 2, 3, 4, 5, 6, 8] 
        value: list, unique attribute values to fill with 
    Outputs: 
        x_fill: array, (9, ), filled attribute values for each pos
    """
    value_fill = np.zeros((len(pos_id),))
    value_fill[:len(value)] = value 
    value_fill[len(value):] = np.random.choice(value, len(pos_id)-len(value), replace=True)
    np.random.shuffle(value_fill)
    
    x_fill = -1 * np.ones((9,))
    for i_pos, pos in enumerate(pos_id): 
        x_fill[pos] = value_fill[i_pos]
    
    return x_fill 

def fill_panel(pos_id, value_shape, value_size, value_color): 
    """
    Args: 
        num_pos: number of positions to fill, ~ 1-9
        value_attr: list of unique values in each panel
    Outputs: 
        x_all: (num_pos, 3)
    """
    x_all = np.nan * np.zeros((9, 3))
    x_all[:,0] =  fill_panel_attr(pos_id, value_shape)
    x_all[:,1] =  fill_panel_attr(pos_id, value_size)
    x_all[:,2] =  fill_panel_attr(pos_id, value_color)
    return x_all

def sample_nan_pos(x_num, r_dict_pos, r_dict_pos_withR): 
    key_first2 = str(x_num[0][0])+str(x_num[1][0])
    N3 = x_num[2][0] 
    if N3 in r_dict_pos[key_first2].keys(): 
        d_pos = r_dict_pos[key_first2][N3]
        x_pos = np.copy(random.choice(d_pos)) # randomly choose one
        d = r_dict_pos_withR[get_key_first2(x_pos)]
        x_pos_last_new = list(np.sort(np.random.choice(9, N3, replace=False)))
        while x_pos_last_new in d.values(): 
            x_pos_last_new = list(np.sort(np.random.choice(9, N3, replace=False)))
        x_pos[2] = list(x_pos_last_new)
    else: 
        k = random.choice(list(r_dict_pos[key_first2].keys()))
        x_pos = np.copy(random.choice(list(r_dict_pos[key_first2][k])))
        x_pos[2] = list(np.sort(np.random.choice(9, N3, replace=False)))
    return x_pos 