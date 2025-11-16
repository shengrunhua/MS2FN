import torch
from preprocess import loadData,get_location,patch_data,dist_mask,set_seed,LDA_operation,slic_data,SLIC
import numpy as np
import math
from tqdm import tqdm
from MS2FN import MS2FN
from train import train
from test import test
from thop import profile
from sklearn import preprocessing

device = torch.device("cuda:0")

data_size1 = 5
data_size2 = 15
hidden = 48
scale1 = 25
scale2 = 225
compactness = 0.1
batch_size = 32

dataset_name = 'IP'
data, labels_TE, labels_TR, class_num = loadData(dataset_name)

# Normalize data
H,W,S=data.shape
data = np.reshape(data, [H * W, S])
minMax = preprocessing.StandardScaler()
data = minMax.fit_transform(data)
data = np.reshape(data, [H, W, S])

# Prepare training and test data
## patch data
train_location,train_label = get_location(labels_TR)
test_location,test_label = get_location(labels_TE)
train_data1 = patch_data(data,data_size1,train_location)
test_data1 = patch_data(data,data_size1,test_location)
train_data2 = patch_data(data,data_size2,train_location)
test_data2 = patch_data(data,data_size2,test_location)
## slic data
data_lda = LDA_operation(data, labels_TR, mode='Normal')
_, _, C = data_lda.shape
segments1, max_pixel_num1 = SLIC(data, scale1, compactness)
segments2, max_pixel_num2 = SLIC(data_lda, scale2, compactness)
train_data_s1 = slic_data(data, segments1, max_pixel_num1, train_location)
test_data_s1 = slic_data(data, segments1, max_pixel_num1, test_location)
train_data_s2 = slic_data(data, segments2, max_pixel_num2, train_location)
test_data_s2 = slic_data(data, segments2, max_pixel_num2, test_location)

# Calculate Laplacian matrix
## training data
cal_batch = 64
batch_num = math.ceil(train_data1.shape[0]/cal_batch)
dist_train1 = torch.zeros([train_data1.shape[0],train_data1.shape[1],train_data1.shape[1]])
dist_train2 = torch.zeros([train_data2.shape[0],train_data2.shape[1],train_data2.shape[1]])
dist_train_s1 = torch.zeros([train_data_s1.shape[0],train_data_s1.shape[1],train_data_s1.shape[1]])
dist_train_s2 = torch.zeros([train_data_s2.shape[0],train_data_s2.shape[1],train_data_s2.shape[1]])
for i in tqdm(range(batch_num)):
    if i != batch_num-1:
        dist_train1[i*cal_batch:(i+1)*cal_batch] = dist_mask(train_data1[i*cal_batch:(i+1)*cal_batch],device)
        dist_train2[i*cal_batch:(i+1)*cal_batch] = dist_mask(train_data2[i*cal_batch:(i+1)*cal_batch],device)
        dist_train_s1[i*cal_batch:(i+1)*cal_batch] = dist_mask(train_data_s1[i*cal_batch:(i+1)*cal_batch],device)
        dist_train_s2[i*cal_batch:(i+1)*cal_batch] = dist_mask(train_data_s2[i*cal_batch:(i+1)*cal_batch],device)
    else:
        dist_train1[i*cal_batch:] = dist_mask(train_data1[i*cal_batch:],device)
        dist_train2[i*cal_batch:] = dist_mask(train_data2[i*cal_batch:],device)
        dist_train_s1[i*cal_batch:] = dist_mask(train_data_s1[i*cal_batch:],device)
        dist_train_s2[i*cal_batch:] = dist_mask(train_data_s2[i*cal_batch:],device)
## test data
test_batch_num = math.ceil(test_data1.shape[0]/cal_batch)
dist_test1 = torch.zeros([test_data1.shape[0],test_data1.shape[1],test_data1.shape[1]])
dist_test2 = torch.zeros([test_data2.shape[0],test_data2.shape[1],test_data2.shape[1]])
dist_test_s1 = torch.zeros([test_data_s1.shape[0],test_data_s1.shape[1],test_data_s1.shape[1]])
dist_test_s2 = torch.zeros([test_data_s2.shape[0],test_data_s2.shape[1],test_data_s2.shape[1]])
for i in tqdm(range(test_batch_num)):
    if i != test_batch_num-1:
        dist_test1[i*cal_batch:(i+1)*cal_batch] = dist_mask(test_data1[i*cal_batch:(i+1)*cal_batch],device)
        dist_test2[i*cal_batch:(i+1)*cal_batch] = dist_mask(test_data2[i*cal_batch:(i+1)*cal_batch],device)
        dist_test_s1[i*cal_batch:(i+1)*cal_batch] = dist_mask(test_data_s1[i*cal_batch:(i+1)*cal_batch],device)
        dist_test_s2[i*cal_batch:(i+1)*cal_batch] = dist_mask(test_data_s2[i*cal_batch:(i+1)*cal_batch],device)
    else:
        dist_test1[i*cal_batch:] = dist_mask(test_data1[i*cal_batch:],device)
        dist_test2[i*cal_batch:] = dist_mask(test_data2[i*cal_batch:],device)
        dist_test_s1[i*cal_batch:] = dist_mask(test_data_s1[i*cal_batch:],device)
        dist_test_s2[i*cal_batch:] = dist_mask(test_data_s2[i*cal_batch:],device)

model = MS2FN(S, data_size1, data_size2, class_num, hidden, max_pixel_num1, max_pixel_num2, device).to(device)

# Calculate FLOPs and params
input1 = torch.from_numpy(train_data1)[:32].to(device)
input_s1 = torch.from_numpy(train_data_s1)[:32].to(device)
dist1 = dist_train1[:32].to(device)
dist_s1 = dist_train_s1[:32].to(device)
input2 = torch.from_numpy(train_data2)[:32].to(device)
input_s2 = torch.from_numpy(train_data_s2)[:32].to(device)
dist2 = dist_train2[:32].to(device)
dist_s2 = dist_train_s2[:32].to(device)
flops, params = profile(model, inputs=(input1, input2, input_s1, input_s2, dist1, dist2, dist_s1, dist_s2))

# Training
train(train_data1, train_data2, train_data_s1, train_data_s2, dist_train1, dist_train2, dist_train_s1, dist_train_s2, train_label, model, dataset_name, class_num, batch_size, device, 0)

# Test
_,OA,acc_class,AA,kappa = test(test_data1, test_data2, test_data_s1, test_data_s2, dist_test1, dist_test2, dist_test_s1, dist_test_s2, test_label, model, dataset_name, class_num, device)

# Save results
name = ['FLOPs', 'Total Parameters'] + [int(i) for i in range(1, len(acc_class) + 1)] + ['AA','OA','kappa*100']
number = [flops, params] + list(acc_class * 100) + [AA * 100, OA * 100, kappa * 100]
for i in range(len(number) - 2):
    number[i + 2] = round(number[i + 2], 2)
with open(type(model).__name__ + '_' + dataset_name + '_' + str(data_size1) + '_' + str(data_size2) + '_' + str(scale1) + '_' + str(scale2) + '_' + str(hidden) + '.txt', 'w') as f:
    tplt = '{:<20}\t{:<20}\t'
    for i in range(len(number)):
        f.write(tplt.format(name[i], number[i]))
        f.write('\n')