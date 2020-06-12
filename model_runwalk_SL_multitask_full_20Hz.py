import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import sys
import random
import time
import xlrd
class Multitask_cnn(nn.Module):
    def __init__(self):
        super(Multitask_cnn, self).__init__()
        
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=8,            # n_filters 
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after con2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(), # activation
            nn.BatchNorm2d(8),
#             nn.MaxPool2d(kernel_size=[1,4]),    # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(8, 16, 3, 1, 1),    
            nn.ReLU(),                      
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=[1,2]),                
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(16, 32, 3, 1, 1),   
            nn.ReLU(),                      
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=[1,2]),                
        )
        
        self.conv4 = nn.Sequential(         
            nn.Conv2d(32, 32, 3, 1, 1),   
            nn.ReLU(),                      
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=[1,2]),                
        )
        self.conv5 = nn.Sequential(         
            nn.Conv2d(32, 32, 3, 1, 1),   
            nn.ReLU(),                      
            nn.BatchNorm2d(32),
         
            nn.MaxPool2d(kernel_size=[1,2]),                
        )
        
        self.linear1 = nn.Sequential(         
            nn.Linear(768, 128),   
            nn.ReLU(),                      
            nn.BatchNorm1d(128) ,              
        )

        self.out_sl = nn.Sequential(         
            nn.Linear(192,1),            
        )
        
        self.out_rw = nn.Sequential(         
            nn.Linear(192,2),               
        )
        
 
        
        self.drop5 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)   
#         print(x.size())
#         x = self.linear1(x)
#         x = self.drop5(x)
        output_sl = self.out_sl(x)
        output_rw = self.out_rw(x)
#         output = self.out(x)

        return output_sl,output_rw,F.softmax(output_rw,dim=1)   # return x for visualization
