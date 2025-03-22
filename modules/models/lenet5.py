import torch
import torch.nn as nn


class LeNet_5(nn.Module):

  def __init__(self, in_channels = 1):
    super().__init__()
 
    self.tanh = nn.Tanh()
    self.c1 = nn.Conv2d(in_channels,6,kernel_size=5, stride=1, padding=0)
    self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

    self.c3_0 = nn.Conv2d(3,1,kernel_size=5,stride=1)
    self.c3_1 = nn.Conv2d(3,1,kernel_size=5,stride=1)
    self.c3_2 = nn.Conv2d(3,1,kernel_size=5,stride=1)
    self.c3_3 = nn.Conv2d(3,1,kernel_size=5,stride=1)
    self.c3_4 = nn.Conv2d(3,1,kernel_size=5,stride=1)
    self.c3_5 = nn.Conv2d(3,1,kernel_size=5,stride=1)
    self.c3_6 = nn.Conv2d(4,1,kernel_size=5,stride=1)
    self.c3_7 = nn.Conv2d(4,1,kernel_size=5,stride=1)
    self.c3_8 = nn.Conv2d(4,1,kernel_size=5,stride=1)
    self.c3_9 = nn.Conv2d(4,1,kernel_size=5,stride=1)
    self.c3_10 = nn.Conv2d(4,1,kernel_size=5,stride=1)
    self.c3_11 = nn.Conv2d(4,1,kernel_size=5,stride=1)
    self.c3_12 = nn.Conv2d(4,1,kernel_size=5,stride=1)
    self.c3_13 = nn.Conv2d(4,1,kernel_size=5,stride=1)
    self.c3_14 = nn.Conv2d(4,1,kernel_size=5,stride=1)
    self.c3_15 = nn.Conv2d(6,1,kernel_size=5,stride=1)

    self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

    self.c5 = nn.Conv2d(16,120,kernel_size=5,stride=1)
    
    self.l1 = nn.Linear(120,84)
    self.l2 = nn.Linear(84,10)

    self.tanh = nn.Tanh()

    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
  

  def forward(self,x):

    dim = x.shape[0]
    out = self.c1(x)
    out = self.tanh(out)
    out = self.s2(out)

    basket = torch.zeros((dim,16,10,10)).to(self.device)
    
    basket[:,0,:,:] = self.c3_0(out[:,[0,1,2],:,:])[:,0,:,:]
    basket[:,1,:,:] = self.c3_1(out[:,[1,2,3],:,:])[:,0,:,:]
    basket[:,2,:,:] = self.c3_2(out[:,[2,3,4],:,:])[:,0,:,:]
    basket[:,3,:,:] = self.c3_3(out[:,[3,4,5],:,:])[:,0,:,:]
    basket[:,4,:,:] = self.c3_4(out[:,[0,4,5],:,:])[:,0,:,:]
    basket[:,5,:,:] = self.c3_5(out[:,[0,1,5],:,:])[:,0,:,:]
    basket[:,6,:,:] = self.c3_6(out[:,[0,1,2,3],:,:])[:,0,:,:]
    basket[:,7,:,:] = self.c3_7(out[:,[1,2,3,4],:,:])[:,0,:,:]
    basket[:,8,:,:] = self.c3_8(out[:,[2,3,4,5],:,:])[:,0,:,:]
    basket[:,9,:,:] = self.c3_9(out[:,[0,3,4,5],:,:])[:,0,:,:]
    basket[:,10,:,:] = self.c3_10(out[:,[0,1,4,5],:,:])[:,0,:,:]
    basket[:,11,:,:] = self.c3_11(out[:,[0,1,2,5],:,:])[:,0,:,:]
    basket[:,12,:,:] = self.c3_12(out[:,[0,1,3,4],:,:])[:,0,:,:]
    basket[:,13,:,:] = self.c3_13(out[:,[1,2,4,5],:,:])[:,0,:,:]
    basket[:,14,:,:] = self.c3_14(out[:,[0,2,3,5],:,:])[:,0,:,:]
    basket[:,15,:,:] = self.c3_15(out)[:,0,:,:]
    
    out = basket
    out = self.tanh(out)
    out = self.s4(out)
    out = self.c5(out)

    temp = torch.zeros(dim,120).to(self.device)
    

    for idx in range(dim):
      temp[idx,:] = out[idx,:,0,0]
    
    out = temp
    out = self.tanh(out)
    out = self.l1(out)
    out = self.tanh(out)
    out = self.l2(out)

    return out

  def __str__(self):
    return 'LeNet_5_vanilla'
