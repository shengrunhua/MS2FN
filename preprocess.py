import numpy as np
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from skimage.segmentation import slic

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


def dist_mask(data,device):
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
    dist = dist.cpu()
    dist[index[0], index[1], :] = 0
    dist[index[0], :, index[1]] = 0
    
    return dist

def LDA_operation(data,label):
    H,W,S=data.shape
    data=data.reshape(-1,S)
    label=label.reshape(-1)
    idx=np.where(label!=0)[0]
    x=data[idx]
    y=label[idx]
    lda=LDA()
    lda.fit(x,y-1)
    data=lda.transform(data)
    print('LDA Finished.')
    
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
