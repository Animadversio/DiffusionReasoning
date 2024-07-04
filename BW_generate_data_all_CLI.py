# %%
# %load_ext autoreload
# %autoreload 2

# %%
import sys
sys.path.append("/n/home12/binxuwang/Github/DiffusionReasoning/RPM_data_generation")
sys.path.append("/n/home12/binxuwang/Github/DiffusionReasoning")

# %%
import numpy as np 
import torch 
import os 
import pickle
import random 
import time 
from os.path import join
from tqdm import tqdm, trange
os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device('cpu')

# %%
from utils_dataset import sample_nan, fill_panel, sample_nan_pos, sample_x_with_num

# %%
data_dir = '/n/home12/binxuwang/Github/DiffusionReasoning/rule_data/20240308_data/'


# %%

# Load numpy arrays
# Load pickled dictionaries
r_all_M7 = np.load(join(data_dir, 'r_list_M7.npy'), allow_pickle=True)
r_dict_M7 = pickle.load(open(join(data_dir, 'r_dict_M7.pkl'), 'rb'))
r_last_M7 = pickle.load(open(join(data_dir, 'r_last_M7.pkl'), 'rb'))
r_all_M10 = np.load(join(data_dir, 'r_list_M10.npy'), allow_pickle=True)
r_dict_M10 = pickle.load(open(join(data_dir, 'r_dict_M10.pkl'), 'rb'))
r_last_M10 = pickle.load(open(join(data_dir, 'r_last_M10.pkl'), 'rb'))
r_list_pos = np.load(join(data_dir, 'r_list_pos.npy'), allow_pickle=True)
r_dict_pos = pickle.load(open(join(data_dir, 'r_dict_pos.pkl'), 'rb'))
r_list_num = np.load(join(data_dir, 'r_list_num.npy'), allow_pickle=True)
r_dict_num = pickle.load(open(join(data_dir, 'r_dict_num.pkl'), 'rb'))
r_dict_pos_withR = pickle.load(open(join(data_dir, 'r_dict_pos_withR.pkl'), 'rb'))

# Convert keys of r_dict_num to a list
r_dict_num_keylist = list(r_dict_num.keys())

# %%
num_samples_total = 1200500
num_samples_keep = 1200000
output_dir = '/n/holylfs06/LABS/kempner_fellow_binxuwang/Users/binxuwang/Datasets/RPM_dataset/RPM1000k'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(join(output_dir, 'aR'), exist_ok=True)

# %% [markdown]
# ### Attribute 0 Shape

# %%
i_a = 0

for i_R in range(7): 
    
    start_time = time.time()
    
    r_list = r_all_M7[i_R]
    num_samples_per_r = int(np.ceil(num_samples_total/len(r_list)))

    attr_all = []

    for i_r in range(len(r_list)):

        for i_sample in trange(num_samples_per_r):

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
    ind = np.random.choice(np.arange(len(attr_all)), num_samples_keep,replace=False)
    # np.save(output_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    np.save(join(output_dir, 'aR', f"{i_a}_{i_R}"), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur)+' -- '+str(len(b) == len(a)))

# %%
i_a = 0 
for i_R in range(7, 10): 
    
    r_list = r_all_M7[i_R]
    ind_select = np.random.choice(len(r_list), num_samples_total, replace=True) # changed to get more samples @July 3rd 2024
    
    attr_all = []
    start_time = time.time()
    
    for i_sample, i_r in tqdm(enumerate(ind_select)):

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
    ind = np.random.choice(np.arange(len(attr_all)), num_samples_keep,replace=False)
    # np.save(output_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    np.save(join(output_dir, 'aR', f"{i_a}_{i_R}"), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur)+' -- '+str(len(b) == len(a)))

# %% [markdown]
# ### Attribute 1 Size

# %%
i_a = 1

for i_R in range(7): 
    
    start_time = time.time()
    
    r_list = r_all_M10[i_R]
    num_samples_per_r = int(np.ceil(num_samples_total/len(r_list)))

    attr_all = []

    for i_r in range(len(r_list)):

        for i_sample in trange(num_samples_per_r):

            # 1) x_shape is determined by the rule 
            x_size = r_list[i_r] # e.g. [[1], [1], [1]]

            potential_list = []
            while len(potential_list) == 0: 
                # 2) sample x_size and x_color attribute values to be not any rule ~ 
                # to avoid shortcut by just looking at the first two panels, make first2 same as some rule 
                x_shape = sample_nan(r_all_M7, r_dict_M7, r_last_M7)
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
    ind = np.random.choice(np.arange(len(attr_all)), num_samples_keep,replace=False)
    # np.save(output_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    np.save(join(output_dir, 'aR', f"{i_a}_{i_R}"), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur))

# %%
i_a = 1 
for i_R in range(7, 10):
    
    r_list = r_all_M10[i_R]
    ind_select = np.random.choice(len(r_list), num_samples_total, replace=True) # changed to get more samples @July 3rd 2024

    attr_all = []

    for i_sample, i_r in tqdm(enumerate(ind_select)):

        start_time = time.time()

        # 1) x_size is determined by the rule 
        x_size = r_list[i_r] # e.g. [[1], [1], [1]]

        potential_list = []
        while len(potential_list) == 0: 
            # 2) sample x_size and x_color attribute values to be not any rule ~ 
            # to avoid shortcut by just looking at the first two panels, make first2 same as some rule 
            x_shape = sample_nan(r_all_M7, r_dict_M7, r_last_M7)
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
    ind = np.random.choice(np.arange(len(attr_all)), num_samples_keep,replace=False)
    # np.save(output_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    np.save(join(output_dir, 'aR', f"{i_a}_{i_R}"), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur)+' -- '+str(len(b) == len(a)))

# %% [markdown]
# ### Attribute 2 Color

# %%
i_a = 2
for i_R in range(7): 
    
    start_time = time.time()
    
    r_list = r_all_M10[i_R]
    num_samples_per_r = int(np.ceil(num_samples_total/len(r_list)))

    attr_all = []

    for i_r in range(len(r_list)):

        for i_sample in trange(num_samples_per_r):

            # 1) x_color is determined by the rule 
            x_color = r_list[i_r] # e.g. [[1], [1], [1]]

            potential_list = []
            while len(potential_list) == 0: 
                # 2) sample x_size and x_color attribute values to be not any rule ~ 
                # to avoid shortcut by just looking at the first two panels, make first2 same as some rule 
                x_shape = sample_nan(r_all_M7, r_dict_M7, r_last_M7)
                x_size = sample_nan(r_all_M10, r_dict_M10, r_last_M10)

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
    ind = np.random.choice(np.arange(len(attr_all)), num_samples_keep,replace=False)
    # np.save(output_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    np.save(join(output_dir, 'aR', f"{i_a}_{i_R}"), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur)+' -- '+str(len(b) == len(a)))

# %%
i_a = 2 
for i_R in range(7, 10):
    
    start_time = time.time()
    
    r_list = r_all_M10[i_R]
    ind_select = np.random.choice(len(r_list), num_samples_total, replace=True) # changed to get more samples @July 3rd 2024

    attr_all = []

    for i_sample, i_r in tqdm(enumerate(ind_select)):

        # 1) x_size is determined by the rule 
        x_color = r_list[i_r] # e.g. [[1], [1], [1]]

        potential_list = []
        while len(potential_list) == 0: 
            # 2) sample x_size and x_color attribute values to be not any rule ~ 
            # to avoid shortcut by just looking at the first two panels, make first2 same as some rule 
            x_shape = sample_nan(r_all_M7, r_dict_M7, r_last_M7)
            x_size = sample_nan(r_all_M10, r_dict_M10, r_last_M10)

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
    ind = np.random.choice(np.arange(len(attr_all)), num_samples_keep,replace=False)
    # np.save(output_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    np.save(join(output_dir, 'aR', f"{i_a}_{i_R}"), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur)+' -- '+str(len(b) == len(a)))

# %% [markdown]
# ### Attribute 3 Number

# %%
i_a = 3
for i_R in range(7): 
    
    r_list = r_list_num[i_R]
    num_samples_per_r = int(np.ceil(num_samples_total/len(r_list)))

    attr_all = []

    for i_r in range(len(r_list)):

        start_time = time.time()

        for i_sample in trange(num_samples_per_r):

            x_num = r_list[i_r] # e.g., [[2], [2], [2]]

            x_shape = sample_x_with_num(x_num, r_all_M7, r_dict_M7, r_last_M7)
            x_size = sample_x_with_num(x_num, r_all_M10, r_dict_M10, r_last_M10)
            x_color = sample_x_with_num(x_num, r_all_M10, r_dict_M10, r_last_M10)

            x_pos = sample_nan_pos(x_num, r_dict_pos, r_dict_pos_withR) # 4) sample position

            attr = np.asarray([fill_panel(x_pos[i], x_shape[i], x_size[i], x_color[i]) for i in range(3)])
            attr_all.append(attr)

            del x_shape, x_size, x_color, x_pos, x_num 

    dur = time.time() - start_time
    
    attr_all = np.asarray(attr_all)
    a = np.copy(attr_all).reshape((-1, 3*9*3))
    b = np.unique(a, axis=0)
    ind = np.random.choice(np.arange(len(attr_all)), num_samples_keep,replace=False)
    # np.save(output_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    np.save(join(output_dir, 'aR', f"{i_a}_{i_R}"), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur)+' -- '+str(len(b) == len(a)))

# %% [markdown]
# ### Attribute 4: number 

# %%
r_all_num_all = []
for r_list in r_list_num: 
    for x in r_list: 
        r_all_num_all.append(x)

# %%
i_a = 4
for i_R in range(7,10):
    
    start_time = time.time()
    
    r_list = r_list_pos[i_R-7]
    ind_select = np.random.choice(len(r_list), num_samples_total, replace=True) # changed to get more samples @July 3rd 2024

    attr_all = []

    for i_sample, i_r in tqdm(enumerate(ind_select)):

        x_pos = r_list[i_r]
        x_num = [[len(x)] for x in x_pos]

        if x_num in r_all_num_all: 
            print('Error')

        x_shape = sample_x_with_num(x_num, r_all_M7, r_dict_M7, r_last_M7)
        x_size = sample_x_with_num(x_num, r_all_M10, r_dict_M10, r_last_M10)
        x_color = sample_x_with_num(x_num, r_all_M10, r_dict_M10, r_last_M10)
        attr = np.asarray([fill_panel(x_pos[i], x_shape[i], x_size[i], x_color[i]) for i in range(3)])

        attr_all.append(attr)
        del x_shape, x_size, x_color, x_pos, x_num 

    dur = time.time() - start_time
    
    attr_all = np.asarray(attr_all)
    a = np.copy(attr_all).reshape((-1, 3*9*3))
    b = np.unique(a, axis=0)
    ind = np.random.choice(np.arange(len(attr_all)), num_samples_keep,replace=False)
    # np.save(output_dir+'aR/'+str(i_a)+'_'+str(i_R), attr_all[ind])
    np.save(join(output_dir, 'aR', f"{i_a}_{i_R}"), attr_all[ind])
    
    print(str(i_R)+' -- '+str(dur)+' -- '+str(len(b) == len(a)))

# %%



