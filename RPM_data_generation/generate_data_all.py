# %%
import numpy as np 
import torch 
import os 

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device('cpu')

# %%
import pickle
import random 
import time 

# %%
from utils_dataset import sample_nan, fill_panel, sample_nan_pos

# %%
data_dir = 'data/20240308_data/'

# %%
r_all_M7 = np.load(data_dir+'r_list_M7.npy', allow_pickle=True)

r_all_M10 = np.load(data_dir+'r_list_M10.npy', allow_pickle=True)
with open(data_dir+'r_dict_M10.pkl', 'rb') as file:
    r_dict_M10 = pickle.load(file)
with open(data_dir+'r_last_M10.pkl', 'rb') as file:
    r_last_M10 = pickle.load(file)
    
with open(data_dir+'r_dict_pos.pkl', 'rb') as file:
    r_dict_pos = pickle.load(file)
    
with open(data_dir+'r_dict_num.pkl', 'rb') as file:
    r_dict_num = pickle.load(file)
    
with open(data_dir+'r_dict_pos_withR.pkl', 'rb') as file:
    r_dict_pos_withR = pickle.load(file)

r_dict_num_keylist = list(r_dict_num.keys())

# %%
num_samples_total = 12010

# %%
i_a = 0

for i_R in range(7): 
    
    start_time = time.time()
    
    r_list = r_all_M7[i_R]
    num_samples_per_r = int(np.ceil(num_samples_total/len(r_list)))

    attr_all = []

    for i_r in range(len(r_list)):

        for i_sample in range(num_samples_per_r):

            # 1) x_shape is determined by the rule 
            x_shape = r_all_M7[i_R][i_r] # e.g. [[1], [1], [1]]

            potential_list = []
            while len(potential_list) == 0: 
                # 2) sample x_size and x_color attribute values to be not any rule ~ 
                # to avoid shortcut by just looking at the first two panels, make first2 same as some rule 
                x_size = sample_nan(r_all_M10, r_dict_M10, r_last_M10)
                x_color = sample_nan(r_all_M10, r_dict_M10, r_last_M10)

                # get the number of unique attribues per panel. This is the min for x_num
                num_attr_unique = [max([len(x_shape[i]), len(x_size[i]), len(x_color[i])]) for i in range(3)]

                # 3) sample number 
                # the way we sample it is to choose one 1)>x_min, 2) first 2 panel belong to some row 3) last panel not 
                potential_first2 = [k for k in r_dict_num_keylist if int(k[0])>=num_attr_unique[0] and int(k[1])>=num_attr_unique[1]]
                if len(potential_first2) > 0: 
                    key_first2 = random.choice(potential_first2) # e.g., '77'
                    d = r_dict_num[key_first2]
                    potential_list = [i for i in range(num_attr_unique[2], 9) if i not in d]

            last = random.choice(potential_list) # could this give us no choice? 
            x_num = [[int(key_first2[0])], [int(key_first2[1])], [last]] # e.g., [7, 7, 4]
            x_pos = sample_nan_pos(x_num, r_dict_pos, r_dict_pos_withR) # 4) sample position 

            attr = np.asarray([fill_panel(x_pos[i], x_shape[i], x_size[i], x_color[i]) for i in range(3)])
            attr_all.append(attr)

            del x_shape, x_size, x_color, x_pos, x_num 

    dur = time.time() - start_time
    
    attr_all = np.asarray(attr_all)
    a = np.copy(attr_all).reshape((-1, 3*9*3))
    b = np.unique(a, axis=0)
    ind = np.random.choice(np.arange(len(attr_all)), 12000,replace=False)
    np.save(data_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur)+' -- '+str(len(b) == len(a)))

# %%
i_a = 0 
for i_R in range(7, 10): 
    
    r_list = r_all_M7[i_R]
    ind_select = np.random.choice(len(r_list), num_samples_total, replace=False)
    
    attr_all = []
    start_time = time.time()
    
    for i_sample, i_r in enumerate(ind_select):

        # 1) x_shape is determined by the rule 
        x_shape = r_all_M7[i_R][i_r] # e.g. [[1], [1], [1]]

        potential_list = []
        while len(potential_list) == 0: 
            # 2) sample x_size and x_color attribute values to be not any rule ~ 
            # to avoid shortcut by just looking at the first two panels, make first2 same as some rule 
            x_size = sample_nan(r_all_M10, r_dict_M10, r_last_M10)
            x_color = sample_nan(r_all_M10, r_dict_M10, r_last_M10)

            # get the number of unique attribues per panel. This is the min for x_num
            num_attr_unique = [max([len(x_shape[i]), len(x_size[i]), len(x_color[i])]) for i in range(3)]

            # 3) sample number 
            # the way we sample it is to choose one 1)>x_min, 2) first 2 panel belong to some row 3) last panel not 
            potential_first2 = [k for k in r_dict_num_keylist if int(k[0])>=num_attr_unique[0] and int(k[1])>=num_attr_unique[1]]
            if len(potential_first2) > 0: 
                key_first2 = random.choice(potential_first2) # e.g., '77'
                d = r_dict_num[key_first2]
                potential_list = [i for i in range(num_attr_unique[2], 9) if i not in d]

        last = random.choice(potential_list) # could this give us no choice? 
        x_num = [[int(key_first2[0])], [int(key_first2[1])], [last]] # e.g., [7, 7, 4]
        x_pos = sample_nan_pos(x_num, r_dict_pos, r_dict_pos_withR) # 4) sample position 

        attr = np.asarray([fill_panel(x_pos[i], x_shape[i], x_size[i], x_color[i]) for i in range(3)])
        attr_all.append(attr)

        del x_shape, x_size, x_color, x_pos, x_num 
    
    dur = time.time() - start_time
    
    attr_all = np.asarray(attr_all)
    a = np.copy(attr_all).reshape((-1, 3*9*3))
    b = np.unique(a, axis=0)
    ind = np.random.choice(np.arange(len(attr_all)), 12000,replace=False)
    np.save(data_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur)+' -- '+str(len(b) == len(a)))

# %%



