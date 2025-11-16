import torch
from preprocess import loadData,get_location,patch_data,dist_mask,LDA_operation,slic_data,SLIC
from test import predict
import numpy as np
from show import show
from MS2FN import MS2FN
import math
from tqdm import tqdm
from sklearn import preprocessing

device = torch.device("cuda:0")

data_size1 = 5
data_size2 = 15
hidden = 48
scale1 = 25
scale2 = 225
compactness = 0.1

dataset_name = 'IP'

weight_pth = './best_MS2FN_IP_weights.pth'

IP_Color = ['#53AB48', '#89BA43', '#42845B', '#3A7A43', '#945332', '#69BCC8','#FFFFFF', '#C7B0C9', '#DA332C', '#772324', '#326AB3', '#E0DB54', '#D98E34', '#54307E', '#E3775B', '#9D5796']
cmap = IP_Color

data, labels_TE, labels_TR, class_num=loadData(dataset_name)

H, W, S = data.shape
data = np.reshape(data, [H * W, S])
minMax = preprocessing.StandardScaler()
data = minMax.fit_transform(data)
data = np.reshape(data, [H, W, S])

all_location=[]
for i in range(H):
    for j in range(W):
        all_location.append([i,j])
all_location=np.array(all_location)

data_lda = LDA_operation(data, labels_TR, mode='Normal')
_, _, C = data_lda.shape
segments1, max_pixel_num1 = SLIC(data_lda, scale1, compactness)
segments2, max_pixel_num2 = SLIC(data_lda, scale2, compactness)
print(max_pixel_num1, max_pixel_num2)

cal_batch = 64

model = MS2FN(S, data_size1, data_size2, class_num, hidden, max_pixel_num1, max_pixel_num2, device).to(device)

all_num = H * W
all_batch_num = math.ceil(all_num/cal_batch)
total_times = math.ceil(all_batch_num/500)
for times in range(total_times):
    if times != total_times-1:
        dist_all1 = np.zeros([500*cal_batch, data_size1**2, data_size1**2])
        dist_all2 = np.zeros([500*cal_batch, data_size2**2, data_size2**2])
        all_input_data1 = patch_data(data, data_size1, all_location[500*cal_batch*times:500*cal_batch*(times+1)])
        all_input_data2 = patch_data(data, data_size2, all_location[500*cal_batch*times:500*cal_batch*(times+1)])
        all_input_data_s1 = slic_data(data, segments1, max_pixel_num1, all_location[500*cal_batch*times:500*cal_batch*(times+1)])
        all_input_data_s2 = slic_data(data, segments2, max_pixel_num2, all_location[500*cal_batch*times:500*cal_batch*(times+1)])
        dist_all_s1 = np.zeros([500*cal_batch, max_pixel_num1, max_pixel_num1])
        dist_all_s2 = np.zeros([500*cal_batch, max_pixel_num2, max_pixel_num2])
        for i in tqdm(range(500)):
            dist_all1[i*cal_batch:(i+1)*cal_batch] = dist_mask(all_input_data1[i*cal_batch:(i+1)*cal_batch], device)
            dist_all2[i*cal_batch:(i+1)*cal_batch] = dist_mask(all_input_data2[i*cal_batch:(i+1)*cal_batch], device)
            dist_all_s1[i*cal_batch:(i+1)*cal_batch]=dist_mask(all_input_data_s1[i*cal_batch:(i+1)*cal_batch], device)
            dist_all_s2[i*cal_batch:(i+1)*cal_batch]=dist_mask(all_input_data_s2[i*cal_batch:(i+1)*cal_batch], device)
        pred = predict(all_input_data1, all_input_data2, all_input_data_s1, all_input_data_s2, dist_all1, dist_all2, dist_all_s1, dist_all_s2, 1, model, weight_pth, device)
        if times==0:
            all_pred = pred
        else:
            all_pred = np.concatenate((all_pred,pred),axis=0)        
    else:
        dist_all1 = np.zeros([all_num%(500*cal_batch), data_size1**2, data_size1**2])
        dist_all2 = np.zeros([all_num%(500*cal_batch), data_size2**2, data_size2**2])
        dist_all_s1 = np.zeros([all_num%(500*cal_batch), max_pixel_num1, max_pixel_num1])
        dist_all_s2 = np.zeros([all_num%(500*cal_batch), max_pixel_num2, max_pixel_num2])
        all_input_data1 = patch_data(data, data_size1, all_location[500*cal_batch*times:])
        all_input_data2 = patch_data(data, data_size2, all_location[500*cal_batch*times:])
        all_input_data_s1 = slic_data(data, segments1, max_pixel_num1, all_location[500*cal_batch*times:])
        all_input_data_s2 = slic_data(data, segments2, max_pixel_num2, all_location[500*cal_batch*times:])
        for i in tqdm(range(all_batch_num%500)):
            if i != all_batch_num%500-1:
                dist_all1[i*cal_batch:(i+1)*cal_batch] = dist_mask(all_input_data1[i*cal_batch:(i+1)*cal_batch], device)
                dist_all2[i*cal_batch:(i+1)*cal_batch] = dist_mask(all_input_data2[i*cal_batch:(i+1)*cal_batch], device)
                dist_all_s1[i*cal_batch:(i+1)*cal_batch]=dist_mask(all_input_data_s1[i*cal_batch:(i+1)*cal_batch], device)
                dist_all_s2[i*cal_batch:(i+1)*cal_batch]=dist_mask(all_input_data_s2[i*cal_batch:(i+1)*cal_batch], device)
            else:
                dist_all1[i*cal_batch:] = dist_mask(all_input_data1[i*cal_batch:], device)
                dist_all2[i*cal_batch:] = dist_mask(all_input_data2[i*cal_batch:], device)
                dist_all_s1[i*cal_batch:] = dist_mask(all_input_data_s1[i*cal_batch:], device)
                dist_all_s2[i*cal_batch:] = dist_mask(all_input_data_s2[i*cal_batch:], device)
        pred = predict(all_input_data1, all_input_data2, all_input_data_s1, all_input_data_s2, dist_all1, dist_all2, dist_all_s1, dist_all_s2, 1, model, weight_pth, device)
        if times == 0:
            all_pred = pred
        else:
            all_pred = np.concatenate((all_pred, pred), axis=0)

show(all_pred, (H,W), cmap, dataset_name, type(model).__name__)