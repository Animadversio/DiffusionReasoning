import os
from os.path import join
import random
from tqdm import tqdm, trange
import numpy as np 
import torch 
import torchvision
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

def load_raw_data(data_dir,):
    d_PGM = torch.load(join(data_dir, 'PGM_shape_size_color_normalized.pt')) # torch.Size([7, 10, 10, 40, 40])
    train_inputs = torch.load(join(data_dir, 'train_inputs.pt')) # [35, 10000, 3, 9, 3]
    print(train_inputs.shape)
    return d_PGM, train_inputs

def render_attribute_tsr(attr_tsr, offset=5, clip=True): 
    """attr_tsr: (3, n_row, n_col)"""
    # inputs = -0.6891*torch.ones((3, 120 + 2*offset, 120 + 2*offset))
    inputs = -0.6891*torch.ones((attr_tsr.shape[1] * 40 + 2*offset, attr_tsr.shape[2] * 40 + 2*offset))
    for i_x in range(attr_tsr.shape[1]): 
        for i_y in range(attr_tsr.shape[2]): 
            if attr_tsr[0, i_x, i_y] != -1: 
                i_shape, i_size, i_color = attr_tsr[:, i_x, i_y]
                x0, y0 = i_x * 40 + offset, i_y * 40 + offset
                if clip:
                    i_shape = min(7 - 1, i_shape)
                    i_size = min(10 - 1, i_size)
                    i_color = min(10 - 1, i_color)
                inputs[x0:(x0+40), y0:(y0+40)] = d_PGM[int(i_shape), int(i_size), int(i_color)]
    return inputs 

# pos_list = [[20,20], [20,60], [20,100], 
#             [60,20], [60,60], [60,100], 
#             [100,20], [100,60], [100,100]]


def load_PGM_inputs(attr, offset = 0 ): 
    """Map attributes to single channel images 
    attr: (3, 9, 3), (num_panel, num_pos, num_attr)
    
    Return:
        inputs: (3, 120 + 2*offset, 120 + 2*offset)
    """
    pos_list = [[x + offset, y + offset] for x in [0, 40, 80] for y in [0, 40, 80]]
    inputs = -0.6891*torch.ones((3, 120 + 2*offset, 120 + 2*offset))
    for i_panel in range(3): 
        for i_pos in range(9): 
            if attr[i_panel, i_pos, 0] != -1: 
                i_shape, i_size, i_color = attr[i_panel, i_pos]
                x0, y0 = pos_list[i_pos]
                inputs[i_panel, x0:(x0+40), y0:(y0+40)] = d_PGM[int(i_shape), int(i_size), int(i_color)]
    return inputs 

class dataset_PGM_single(Dataset): 
    def __init__(self, attr_list, offset=0): 
        """attr_list: [num_samples, 3, 9, 3]"""
        self.attr_list = attr_list 
        self.offset = offset     
        
    def __len__(self): 
        return len(self.attr_list)
    
    def __getitem__(self, idx): 
        """attr: [3, 9, 3]"""
        attr = self.attr_list[idx] 
        inputs = load_PGM_inputs(attr, offset=self.offset)
        return inputs, idx
    
    
def PGM_dataloader_class(i_class, train_inputs, batch_size=256,): 
    # example: 
    dataset_class0 = dataset_PGM_single(train_inputs[i_class]) 
    load_class0 = DataLoader(dataset_class0, batch_size=256, shuffle=False, pin_memory=True) 
    return dataset_class0, load_class0


def load_PGM_abstract(attr): 
    """attr: (3, 9, 3), (num_panel, num_pos, num_attr)"""
    attr = attr.to(int)
    attr = torch.cat(tuple(attr.view(3, 3, 3, 3)), dim=1) # [3, 3, 3, 3] -> [3, 9, 3]
    inputs = attr.permute(2, 0, 1) # num_attr, num_row=3, num_col (n panel x 3)
    return inputs 


def train_data2attr_tsr(train_input):
    """Turn train_input to attribute tensor CxHxW or BxCxHxW"""
    import einops
    if len(train_input.shape) == 3:
        attr_tsr = einops.rearrange(train_input, 'p (h w) attr -> attr h (p w)', h=3,w=3,p=3)
    elif len(train_input.shape) == 4:
        attr_tsr = einops.rearrange(train_input,  'B p (h w) attr -> B attr h (p w)', h=3,w=3,p=3)
    else: 
        raise ValueError('train_input should be either 3D or 4D tensor')
    attr_tsr = attr_tsr.to(int)
    return attr_tsr


def onehot2attr_tsr(samples, dim=10, threshold=0.4):
    """Turn one-hot tensor to attribute tensor
    Input:
        samples: shape (batch, 27, n_row, n_col)
    
    Return:
        attr_tsr: shape (batch, 3, n_row, n_col)
    """
    attr0_onehot, attr0 = samples[:,  0:7].max(dim=1)
    attr0[attr0_onehot < threshold] = -1
    attr1_onehot, attr1 = samples[:,  7:17].max(dim=1)
    attr1[attr1_onehot < threshold] = -1
    attr2_onehot, attr2 = samples[:, 17:27].max(dim=1)
    attr2[attr2_onehot < threshold] = -1
    attr_tsr = torch.stack((attr0, attr1, attr2), dim=1)
    return attr_tsr