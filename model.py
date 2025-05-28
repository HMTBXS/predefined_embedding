 
import torch 
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder,TransformerEncoderLayer

import torch
import torch.nn as nn
import os,sys 
import time 

class MODEL(torch.nn.Module):
    def __init__(self,norm="None", num_head=16,num_trans_layers=[2,2,2],d_model=512,ch_in=768,ch_out=512,conv_blks=3,hidden_size=1152):
        super(MODEL, self).__init__()
        # Get a resnet50 backbone
        num_head=16
        self.encoder_ex=[]
        for i,it in enumerate(num_trans_layers):
            loc_encoder=TransformerEncoder(TransformerEncoderLayer(d_model=d_model,nhead=num_head),num_layers=it)
            setattr(self,"transblk%d"%(i+1),loc_encoder)
            self.encoder_ex.append(loc_encoder)

        self.conv1d0=nn.Conv1d(d_model,hidden_size,kernel_size=1,padding=0)
        self.relu=nn.ReLU()

        self.conv1d1=nn.Conv1d(ch_out,ch_in,kernel_size=1,padding=0)
        self.relu=nn.ReLU()

        self.norm=norm 
        if self.norm=="ln":
            self.norm_funa=nn.LayerNorm(normalized_shape=hidden_size)
            self.norm_funb=nn.LayerNorm(normalized_shape=d_model)
        
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):

        for encoder_ex in self.encoder_ex:
            x = encoder_ex(x)+x
        if self.norm!="None":
            x=self.norm_funb(x)

        x=self.conv1d1(x)
        x=self.relu(x)
        x=self.conv1d0(x.permute(0,2,1)).permute(0,2,1)
        # x=self.relu(x)
        return x
