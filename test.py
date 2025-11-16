from functions import accuracy_fn,AA_fn,kappa_fn
import torch
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from tqdm import tqdm

def test(test_data1, test_data2, test_data_s1, test_data_s2, dist_test1, dist_test2, dist_test_s1, dist_test_s2, test_label, model, dataset_name, class_num, device):
    batch_size = 1
    
    model.load_state_dict(torch.load('best_' + type(model).__name__ + '_' + dataset_name + '_weights.pth'), strict = True)
    model.eval()
    
    test_data1 = torch.tensor(test_data1, dtype=torch.float32)
    test_data2 = torch.tensor(test_data2, dtype=torch.float32)
    test_data_s1 = torch.tensor(test_data_s1, dtype=torch.float32)
    test_data_s2 = torch.tensor(test_data_s2, dtype=torch.float32)
    test_label = torch.tensor(test_label, dtype=torch.float32)
    
    test_dataset = TensorDataset(test_data1, test_data2, test_data_s1, test_data_s2, dist_test1, dist_test2, dist_test_s1, dist_test_s2, test_label)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    corrects = np.zeros(class_num)
    totals = np.zeros(class_num)
    output = []
    true_label = []
    
    with torch.no_grad():
        for data1, data2, data_s1, data_s2, dist1, dist2, dist_s1, dist_s2, labels in tqdm(test_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            data_s1 = data_s1.to(device)
            data_s2 = data_s2.to(device)
            dist1 = dist1.to(device)
            dist2 = dist2.to(device)
            dist_s1 = dist_s1.to(device)
            dist_s2 = dist_s2.to(device)
            labels = labels.to(device)
            outputs = model(data1, data2, data_s1, data_s2, dist1, dist2, dist_s1, dist_s2)
            correct, total = AA_fn(outputs, labels)
            corrects += correct
            totals += total
            outputs = outputs.cpu().tolist()
            output += outputs
            true_label += labels.cpu().tolist()
    
    OA = corrects.sum() / totals.sum()
    output = np.array(output)
    kappa = kappa_fn(output, true_label)
    acc_class = np.divide(corrects, totals)
    AA = np.mean(acc_class)
    print(OA)
    
    return np.argmax(output, axis=-1), OA, acc_class, AA, kappa

def predict(data1, data2, data_s1, data_s2, dist1, dist2, dist_s1, dist_s2, batch_size, model, weight_pth, device):

    model.load_state_dict(torch.load(weight_pth), strict=False)
    model.eval()
    
    data1 = torch.tensor(data1, dtype=torch.float32)
    data2 = torch.tensor(data2, dtype=torch.float32)
    data_s1 = torch.tensor(data_s1, dtype=torch.float32)
    data_s2 = torch.tensor(data_s2, dtype=torch.float32)
    print('Data changed')
    dist1 = torch.tensor(dist1, dtype=torch.float32)
    dist2 = torch.tensor(dist2, dtype=torch.float32)
    dist_s1 = torch.tensor(dist_s1, dtype=torch.float32)
    dist_s2 = torch.tensor(dist_s2, dtype=torch.float32)
    print('Dist changed')
    dataset = TensorDataset(data1, data2, data_s1, data_s2, dist1, dist2, dist_s1, dist_s2)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    
    output = []
    
    with torch.no_grad():
        for data1, data2, data_s1, data_s2, dist1, dist2, dist_s1, dist_s2 in tqdm(loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            data_s1 = data_s1.to(device)
            data_s2 = data_s2.to(device)
            dist1 = dist1.to(device)
            dist2 = dist2.to(device)
            dist_s1 = dist_s1.to(device)
            dist_s2 = dist_s2.to(device)
            outputs = model(data1, data2, data_s1, data_s2, dist1, dist2, dist_s1, dist_s2)
            outputs = outputs.cpu().tolist()
            output += outputs

    output = np.argmax(np.array(output), axis=-1)
    print('Pixel predicted')

    return output