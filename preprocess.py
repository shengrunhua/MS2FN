from scipy.io import loadmat
import h5py
import numpy as np
from tqdm import tqdm
import torch
import random
import math
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from skimage.segmentation import slic

def loadData(Data):
    # 读入数据
    if Data == 'IP':
        data = loadmat('./data/Indian_pines_corrected.mat')['indian_pines_corrected']
        labels_TE = loadmat('./data/IP_TE_gt.mat')['y']
        labels_TR = loadmat('./data/IP_TR_gt.mat')['y']
        class_num = 16
    elif Data == 'PU':
        data = loadmat('./data/PaviaU.mat')['paviaU']
        labels_TE = loadmat('./data/DS_PaviaU_gt_TE.mat')['y']
        labels_TR = loadmat('./data/DS_PaviaU_gt_TR.mat')['y']
        class_num = 9
    elif Data == 'HU':
        data = loadmat('./data/Houston2013.mat')['HSI']
        labels_TE = loadmat('./data/TSLabel.mat')['TSLabel']
        labels_TR = loadmat('./data/TRLabel.mat')['TRLabel']
        class_num = 15
    elif Data == 'LN01':
        data = loadmat('./data/LN/LN01/HSI/LN01_HSI.mat')['HSI']
        label = loadmat('./data/LN/LN01/HSI/LN01_gt.mat')['gt']
        class_num = 10
        class_numbers = [0 for i in range(int(class_num))]
        for items in list(label):
            for item in items:
                if item!=0:
                    class_numbers[item-1] += 1           
        class_numbers_train = [math.ceil(0.01 * class_numbers[i]) for i in range(class_num)]
        
        if os.path.exists(Data+'_TE.npy'):
                labels_TE=np.load(Data+'_TE.npy')
                labels_TR=np.load(Data+'_TR.npy')
                print('Dataset Loaded')
        else:
            labels_TE=np.copy(label)
            for i in range(len(class_numbers_train)):
                target_value = i+1
                replace_count=class_numbers_train[i]
                indices = np.where(labels_TE == target_value)
                random_indices = np.random.choice(range(len(indices[0])), replace_count, replace=False)
                for j in random_indices:
                    row, col = indices[0][j], indices[1][j]
                    labels_TE[row, col] = 0
            labels_TR=label-labels_TE
            np.save(Data+'_TE.npy',labels_TE)
            np.save(Data+'_TR.npy',labels_TR)
            print('Dataset Spilted')

    return data, labels_TE, labels_TR, class_num

def get_location(label):
    location=[]
    labels=[]
    H,W=label.shape
    for i in range(H):
        for j in range(W):
            if label[i,j]!=0:
                location.append([i,j])
                labels.append(label[i,j]-1)
    location=np.array(location)
    labels=np.array(labels)
    return location,labels

def patch_data(data,l,location):
    H,W,S=data.shape
    patch_data=[]
    if l==1:
        for idx in tqdm(list(location)):
            i,j=idx
            patch_data.append(data[i,j,:].reshape(-1))
    else:
        for idx in tqdm(list(location)):
            mask=np.float32(np.zeros([l, l, S]))
            i,j=idx
            up=i-int(l/2)
            down=i+int(l/2)
            left=j-int(l/2)
            right=j+int(l/2)
            up=0 if up<0 else up
            left=0 if left < 0 else left
            down=H-1 if down>H-1 else down
            right=W-1 if right>W-1 else right
            mask[int(l/2)-(i-up):int(l/2)+down-i+1,int(l/2)-(j-left):int(l/2)+right-j+1,:]=data[up:down+1,left:right+1,:]
            patch_data.append(mask)
    print('Data patched.')
    patch_data=np.float32(np.array(patch_data))
    patch_data=patch_data.reshape(-1,l**2,S)
    return patch_data


def dist_mask(data,device,mode='Pre'):
    if mode == 'Pre':
        data = torch.tensor(data)
        data = data.to(device)
    data_s = torch.sum(data, dim=2)
    index = torch.where(data_s == 0)
    tile_data = torch.unsqueeze(data, dim=1)
    next_data = torch.unsqueeze(data, dim=-2)
    minus = tile_data - next_data
    a = -torch.sum(minus**2, -1)
    dist = torch.exp(a/data.shape[2])
    dist = dist / torch.sum(dist, 2, keepdims=True)
    dist = dist + torch.eye(data.shape[1]).to(device)
    if mode == 'Pre':
        dist = dist.cpu()
    dist[index[0], index[1], :] = 0
    dist[index[0], :, index[1]] = 0
    
    return dist

def set_seed(seed):
    random.seed(seed) # python的随机性
    np.random.seed(seed) # np的随机性
    torch.manual_seed(seed) # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed_all(seed) # torch的GPU随机性，为所有GPU设置随机种子
    
def train_split(labels_TR,labels_TE,train_ratio):
    class_num=np.max(labels_TR)
    class_numbers = [0 for i in range(int(class_num))]
    for items in list(labels_TR):
        for item in items:
            if item!=0:
                class_numbers[item-1] += 1           
    class_numbers_train = [math.ceil(train_ratio * class_numbers[i]) for i in range(class_num)]
    label=labels_TR+labels_TE
    labels_TE=np.copy(label)
    for i in range(len(class_numbers_train)):
        target_value = i+1
        replace_count=class_numbers_train[i]
        indices = np.where(labels_TR == target_value)
        random_indices = np.random.choice(range(len(indices[0])), replace_count, replace=False)
        for j in random_indices:
            row, col = indices[0][j], indices[1][j]
            labels_TE[row, col] = 0
    labels_TR=label-labels_TE
    
    return labels_TR, labels_TE

def LDA_operation(data,label,mode='Normal'):
    if mode=='Normal':
        H,W,S=data.shape
        data=data.reshape(-1,S)
        label=label.reshape(-1)
        idx=np.where(label!=0)[0]
        x=data[idx]
        y=label[idx]
        lda=LDA()
        lda.fit(x,y-1)
        data=lda.transform(data)
    elif mode=='Special':
        H,W,S=data.shape
        data=data.reshape(-1,S)
        label=label.reshape(-1)
        lda=LDA()
        lda.fit(x,y)
        data=lda.transform(data)
    print('LDA降维完成')
    
    return data.reshape(H, W, -1)

def SLIC(data, scale, compactness):
    H, W, BAND = data.shape
    pixel_num = H * W
    segments = slic(data, n_segments = pixel_num / scale, compactness = compactness, max_num_iter = 20, convert2lab = False, sigma = 0, enforce_connectivity = True, min_size_factor = 0.3, max_size_factor = 2, slic_zero = False)
    
    max_pixel_num = 0
    for i in range(segments.max()):
        index = (segments == i+1)
        pixel_num = np.sum(index)
        if pixel_num > max_pixel_num:
            max_pixel_num = pixel_num
    
    return segments, max_pixel_num

def slic_data(data, segments, max_pixel_num, location):
    H, W, BAND = data.shape
    slic_data = []
    for idx in tqdm(list(location)):
        temp = np.float32(np.zeros([max_pixel_num, BAND]))
        i, j = idx
        num = segments[i, j]
        pixel_idx = np.where(segments == num)
        data_temp = data[pixel_idx]
        pixel = data[i, j].reshape(1, -1)
        matching_indices = np.where(np.all(data_temp == pixel, axis=1))[0][0]
        pixel_temp = data_temp[0, :]
        data_temp[0, :] = data_temp[matching_indices, :]
        data_temp[matching_indices, :] = pixel_temp
        temp[0:data_temp.shape[0], :] = data_temp
        slic_data.append(temp)
        
    print('Data sliced.')
    slic_data = np.float32(np.array(slic_data))
    
    return slic_data