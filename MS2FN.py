import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class mish(nn.Module):
    def __init__(self):
        super(mish, self).__init__()
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class MS2FN(nn.Module):
    def __init__(self, S, l1, l2, class_num, hidden, max_pixel_num1, max_pixel_num2, device):
        super(MS2FN, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.device = device
        
        self.spectral = nn.Sequential(nn.Conv2d(S, hidden * 2, (1, 1)),
                                    nn.BatchNorm2d(hidden * 2),
                                    mish(),
                                    nn.Conv2d(hidden * 2, hidden, (1, 1)),
                                    nn.BatchNorm2d(hidden),
                                    mish())
        
        self.gcn1 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(3)])
        self.gcn2 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(3)])
        self.gbn1 = nn.ModuleList([nn.BatchNorm1d(l1 ** 2) for i in range(3)])
        self.gbn2 = nn.ModuleList([nn.BatchNorm1d(l2 ** 2) for i in range(3)])
        
        self.sgcn1 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(3)])
        self.sgcn2 = nn.ModuleList([nn.Linear(hidden, hidden) for i in range(3)])
        self.sgbn1 = nn.ModuleList([nn.BatchNorm1d(max_pixel_num1) for i in range(3)])
        self.sgbn2 = nn.ModuleList([nn.BatchNorm1d(max_pixel_num2) for i in range(3)])
        
        self.cnn1 = nn.ModuleList([nn.Conv2d(hidden, hidden, kernel_size=3, padding=1) for i in range(3)])
        self.cbn1 = nn.ModuleList([nn.BatchNorm2d(hidden) for i in range(3)])
        self.cnn2 = nn.ModuleList([nn.Conv2d(hidden, hidden, kernel_size=7, padding=3) for i in range(3)])
        self.cbn2 = nn.ModuleList([nn.BatchNorm2d(hidden) for i in range(3)])
        
        self.fpc1 = nn.Conv1d(hidden, hidden, kernel_size=l1 ** 2, groups=hidden)
        self.fpc2 = nn.Conv1d(hidden, hidden, kernel_size=l2 ** 2, groups=hidden)
        
        self.fpb = nn.ModuleList([nn.BatchNorm1d(hidden) for i in range(2)])
        
        self.fusion = nn.Conv1d(1, 1, kernel_size=6, dilation=hidden)
        self.bnf = nn.BatchNorm1d(1)
        
        self.output = nn.Linear(hidden, class_num)
        
    def forward(self, x_p1, x_p2, x_s1, x_s2, dist_p1, dist_p2, dist_s1, dist_s2):
        x_p1 = x_p1.to(torch.float32)
        dist_p1 = dist_p1.to(torch.float32)
        x_p1 = x_p1.reshape(-1, self.l1, self.l1, x_p1.shape[2])
        x_p1 = torch.transpose(x_p1, 1, 3)
        x_p1 = self.spectral(x_p1)
        x_p1c = x_p1
        x_p1 = torch.transpose(x_p1, 1, 3)
        x_p1 = x_p1.reshape(-1, self.l1**2, x_p1.shape[3])
        for i in range(3):
            x_p1 = self.gcn1[i](x_p1)
            x_p1 = torch.bmm(dist_p1, x_p1)
            x_p1 = self.gbn1[i](x_p1)
            x_p1 = mish()(x_p1)
        
        x_p1_g = torch.bmm(dist_p1, x_p1)
        x_p1_g = x_p1_g[:, int((self.l1 ** 2 - 1) / 2), :]
        x_p1 = torch.transpose(x_p1, 1, 2)
        x_p1 = self.fpc1(x_p1)
        x_p1 = self.fpb[0](x_p1)
        x_p1 = torch.transpose(x_p1,1,2).squeeze(1)
        x_p1 = mish()(x_p1)
        x_p1 = (x_p1 + x_p1_g) / 2
        
        
        for i in range(3):
            x_p1c = self.cnn1[i](x_p1c)
            x_p1c = self.cbn1[i](x_p1c)
            x_p1c = mish()(x_p1c)
        
        x_p1c = torch.transpose(x_p1c, 1, 3)
        x_p1c = x_p1c.reshape(-1, self.l1**2, x_p1c.shape[3])
        x_p1c = torch.mean(x_p1c, dim=1)
        
        x_p2 = x_p2.to(torch.float32)
        dist_p2 = dist_p2.to(torch.float32)
        x_p2 = x_p2.reshape(-1, self.l2, self.l2, x_p2.shape[2])
        x_p2 = torch.transpose(x_p2, 1, 3)
        x_p2 = self.spectral(x_p2)
        x_p2c = x_p2
        x_p2 = torch.transpose(x_p2, 1, 3)
        x_p2 = x_p2.reshape(-1, self.l2**2, x_p2.shape[3])
        for i in range(3):
            x_p2 = self.gcn2[i](x_p2)
            x_p2 = torch.bmm(dist_p2, x_p2)
            x_p2 = self.gbn2[i](x_p2)
            x_p2 = mish()(x_p2)

        x_p2_g = torch.bmm(dist_p2, x_p2)
        x_p2_g = x_p2_g[:, int((self.l2 ** 2 - 1) / 2), :]
        x_p2 = torch.transpose(x_p2, 1, 2)
        x_p2 = self.fpc2(x_p2)
        x_p2 = self.fpb[1](x_p2)
        x_p2 = torch.transpose(x_p2,1,2).squeeze(1)
        x_p2 = mish()(x_p2)
        x_p2 = (x_p2 + x_p2_g) / 2
        
        for i in range(3):
            x_p2c = self.cnn2[i](x_p2c)
            x_p2c = self.cbn2[i](x_p2c)
            x_p2c = mish()(x_p2c)
        
        x_p2c = torch.transpose(x_p2c, 1, 3)
        x_p2c = x_p2c.reshape(-1, self.l2**2, x_p2c.shape[3])
        x_p2c = torch.mean(x_p2c, dim=1)
        
        x_s1 = x_s1.to(torch.float32)
        dist_s1 = dist_s1.to(torch.float32)
        x_s1 = torch.transpose(x_s1, 1, 2)
        x_s1 = x_s1.unsqueeze(2)
        x_s1 = self.spectral(x_s1)
        x_s1 = x_s1.squeeze(2)
        x_s1 = torch.transpose(x_s1, 1, 2)
        for i in range(3):
            x_s1 = self.sgcn1[i](x_s1)
            x_s1 = torch.bmm(dist_s1, x_s1)
            x_s1 = self.sgbn1[i](x_s1)
            x_s1 = mish()(x_s1)
        
        weight_s1 = sim(x_s1, self.device)
        x_s1p = torch.bmm(weight_s1, x_s1)
        x_s1 = x_s1p[:, 0, :]
        
        x_s2 = x_s2.to(torch.float32)
        dist_s2 = dist_s2.to(torch.float32)
        x_s2 = torch.transpose(x_s2, 1, 2)
        x_s2 = x_s2.unsqueeze(2)
        x_s2 = self.spectral(x_s2)
        x_s2 = x_s2.squeeze(2)
        x_s2 = torch.transpose(x_s2, 1, 2)
        for i in range(3):
            x_s2 = self.sgcn2[i](x_s2)
            x_s2 = torch.bmm(dist_s2, x_s2)
            x_s2 = self.sgbn2[i](x_s2)
            x_s2 = mish()(x_s2)
        
        weight_s2 = sim(x_s2, self.device)
        x_s2p = torch.bmm(weight_s2, x_s2)
        x_s2 = x_s2p[:, 0, :]

        x = torch.cat((x_p1, x_p1c, x_p2, x_p2c, x_s1, x_s2), dim=1)
        x = x.unsqueeze(1)
        x = self.fusion(x)
        x = self.bnf(x)
        x = mish()(x)
        x = x.squeeze(1)
        y = self.output(x)
        
        return y
    
def sim(data, device):
    data_s = torch.sum(data, dim=2)
    index = torch.where(data_s == 0)
    tile_data = torch.unsqueeze(data, dim=1)
    next_data = torch.unsqueeze(data, dim=-2)
    minus = tile_data - next_data
    a = -torch.sum(minus**2, -1)
    simx = torch.exp(a/data.shape[2])
    simx = simx / torch.sum(simx, 2, keepdims=True)
    simx = simx + torch.eye(data.shape[1]).to(device)
    simx[index[0], index[1], :] = 0
    simx[index[0], :, index[1]] = 0
    
    return simx
