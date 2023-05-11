from torch import nn
import torch.nn.functional as F
import torch

class MLP(nn.Module):
  def __init__(self,out_dim=2, in_channel=1,img_sz=32,hidden_dim=400):
    super().__init__()
    self.in_dim=in_channel*img_sz*img_sz
    self.liner_1 = nn.Linear(self.in_dim,hidden_dim)
    self.liner_2 = nn.Linear(hidden_dim, hidden_dim)
    self.liner_3 = nn.Linear(hidden_dim,out_dim)


  def forward(self, input):
    x = input.view(-1,self.in_dim) # 展开成一列
    x = F.relu(self.liner_1(x))
    x = F.relu(self.liner_2(x))
    x = self.liner_3(x)
    return x


class LeNet5(nn.Module):
  def __init__(self, out_dim=2, in_channel=3):
    super(LeNet5, self).__init__()
    self.conv1 = nn.Conv2d(in_channel, 6, 5)
    self.pool1 = nn.MaxPool2d(2, 2)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool2 = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, out_dim)

  def forward(self, x):
    x = self.pool1(F.relu(self.conv1(x)))
    x = self.pool2(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x
