from functions import loss_fn,AA_fn,kappa_fn
import torch
from torch.utils.data import DataLoader,TensorDataset
from torchinfo import summary
import time
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

def train(train_data1, train_data2, train_data_s1, train_data_s2, dist_train1, dist_train2, dist_train_s1, dist_train_s2, train_label, model, dataset_name, class_num, batch_size, device, j):
    epoches = 200
    batch_size = batch_size
        
    last_number = int(len(train_data1) % batch_size)
    
    model.train()
    
    writer = SummaryWriter('./tensorboard/' + type(model).__name__ + dataset_name + 'train')
    
    train_data1 = torch.from_numpy(train_data1.astype(np.float32))
    train_data2 = torch.from_numpy(train_data2.astype(np.float32))
    train_data_s1 = torch.from_numpy(train_data_s1.astype(np.float32))
    train_data_s2 = torch.from_numpy(train_data_s2.astype(np.float32))
    train_label = torch.from_numpy(train_label.astype(np.float32))
    
    if j==0:
        summary(model, input_data = [train_data1[:batch_size].to(device), train_data2[:batch_size].to(device), train_data_s1[:batch_size].to(device), train_data_s2[:batch_size].to(device), dist_train1[:batch_size].to(device), dist_train2[:batch_size].to(device), dist_train_s1[:batch_size].to(device), dist_train_s2[:batch_size].to(device)], batch_dim = 0)
    
    train_dataset=TensorDataset(train_data1, train_data2, train_data_s1, train_data_s2, dist_train1, dist_train2, dist_train_s1, dist_train_s2, train_label)
    if last_number == 1:
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    else:
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    
    Loss=99999
    
    for epoch in range(epoches):
        e_t = time.time()
        losss = 0
        corrects = np.zeros(class_num)
        totals = np.zeros(class_num)

        for data1, data2, data_s1, data_s2, dist1, dist2, dist_s1, dist_s2, label in tqdm(train_loader):
            data1 = data1.to(device)
            data2 = data2.to(device)
            data_s1 = data_s1.to(device)
            data_s2 = data_s2.to(device)
            dist1 = dist1.to(device)
            dist2 = dist2.to(device)
            dist_s1 = dist_s1.to(device)
            dist_s2 = dist_s2.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            outputs = model(data1, data2, data_s1, data_s2, dist1, dist2, dist_s1, dist_s2)
            loss = loss_fn(outputs, label, class_num, device)
            correct, total = AA_fn(outputs, label)
            loss.backward()
            optimizer.step()

            losss += loss.item()
            corrects += correct
            totals += total
        
        epoch_loss = losss
        epoch_accuracy = corrects.sum() / totals.sum()

        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_accuracy, epoch)
        
        print(f"Epoch {epoch+1}/{epoches}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, Epoch_time:{time.time()-e_t:.4f}")
        
        if Loss > epoch_loss:
            Loss = epoch_loss
            OA = epoch_accuracy
            torch.save(model.state_dict(), 'best_' + type(model).__name__ + '_' + dataset_name + '_weights.pth')
            print('-------------save model---------------')

    print(f"Final_Loss:{Loss:.4f}, Final_Accuracy:{OA:.4f}")
    
    return Loss