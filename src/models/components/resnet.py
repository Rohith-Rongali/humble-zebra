from pytorch_lightning import Trainer
from torch import nn
import torch
import math
from scipy.linalg import hadamard

def ZerO_Init_on_matrix(matrix_tensor):
    # Algorithm 1 in the paper.
    
    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)
    
    if m <= n:
        init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2**(clog_m)
        init_matrix = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))
    
    return init_matrix

  

def ZerO_Init_on_conv(matrix_tensor):
    # Algorithm 2 in the paper.
    
    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)
    p = matrix_tensor.size(2)

    init_matrix = torch.zeros_like(matrix_tensor)
    c = math.floor(p/2)
    
    if m <= n:
        init_matrix[:,:,c,c] = torch.nn.init.eye_(torch.empty(m, n))
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2**(clog_m)
        init_matrix[:,:,c,c] = torch.nn.init.eye_(torch.empty(m, p)) @ (torch.tensor(hadamard(p)).float()/(2**(clog_m/2))) @ torch.nn.init.eye_(torch.empty(p, n))
          
    return init_matrix  

def stable_rank(matrix_tensor):
    # computes the sum of the singular values of a matrix divided by maximum singular value.
    # this is a measure of the rank of the matrix.
    
    u, s, v = torch.svd(matrix_tensor[0,:,:,:])
    return torch.sum(s)/torch.max(s)
   
  
class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_dim, dim, stride=1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)
    self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(dim)
    self.activ = nn.ReLU()

    self.shortcut = nn.Identity()
    if (in_dim!=dim):
        self.shortcut = nn.Sequential(nn.Conv2d(in_dim, dim, kernel_size=1, stride=stride, bias=False))
      
  def forward(self, x):
    out = self.activ(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out = out+self.shortcut(x)
    out = self.activ(out)
    return out


class Bottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_dim, dim, stride=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(in_dim, dim, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(dim)
    self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(dim)
    self.conv3 = nn.Conv2d(dim, self.expansion * dim, kernel_size=1, bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion*dim)
    self.activ = nn.ReLU()

    self.shortcut = nn.Identity()
    if (in_dim!=self.expansion*dim):
        self.shortcut = nn.Sequential(nn.Conv2d(in_dim, self.expansion*dim, kernel_size=1, stride=stride, bias=False))


  def forward(self, x):
    out = self.activ(self.bn1(self.conv1(x)))
    out = self.activ(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out = out + self.shortcut(x)
    out = self.activ(out)
    return out


class ResNet(nn.Module):
  CONFIGS = {
      "resnet18": (BasicBlock, [2, 2, 2, 2]),
      "resnet34": (BasicBlock, [3, 4, 6, 3]),
      "resnet50": (Bottleneck, [3, 4, 6, 3]),
      "resnet101": (Bottleneck, [3, 4, 23, 3]),
      "resnet152": (Bottleneck, [3, 8, 36, 3]),
  }
  def __init__(self,init):
    super(ResNet, self).__init__()
    block, num_blocks = self.CONFIGS["resnet18"]
    self.init = init
    self.in_dim = 64
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.activ = nn.ReLU()
    self.avg = nn.AdaptiveAvgPool2d((1, 1))
    self.linear = nn.Linear(512*block.expansion, 10)
    
    if self.init is not None:
        self.apply(self._init_weights)

    self.ranks2 = []
    self.ranks3 = []
    
    
  def _init_weights(self, m):
    
    if self.init == 'ZerO':
        if isinstance(m, nn.Linear):
            m.weight.data = ZerO_Init_on_matrix(m.weight.data)
            
        elif isinstance(m, nn.Conv2d):
            m.weight.data = ZerO_Init_on_conv(m.weight.data)
    
    elif self.init == 'Partial_Identity':
        if isinstance(m, nn.Linear):
            m.weight.data = Identity_Init_on_matrix(m.weight.data)
    
    elif self.init == 'Kaiming':
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            
    elif self.init == 'Xavier':
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight, gain = torch.nn.init.calculate_gain('relu'))
            
    if isinstance(m, nn.Linear) and m.bias is not None:
        nn.init.constant_(m.bias, 0)
        
    if isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, dim, num_blocks, stride):
    strides = [stride] + [1]*(num_blocks-1)    
    layers = []
    for stride in strides: 
        layer = block(self.in_dim, dim, stride)
        layers.append(layer)
        self.in_dim = dim*block.expansion
    return nn.Sequential(*layers)

  def forward(self, x):
    out = self.activ(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    rank2 = stable_rank(self.layer1[1].conv1.weight).detach().cpu()
    self.ranks2.append(rank2)    
    out = self.layer2(out)
    rank3 = stable_rank(self.layer2[0].conv1.weight).detach().cpu()
    self.ranks3.append(rank3)     
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avg(out)
    out = out.reshape(out.shape[0], -1)
    out = self.linear(out)
    return out


if __name__ == "__main__":
    _ = ResNet()

