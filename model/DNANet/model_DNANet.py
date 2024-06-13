import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from tools.swin import *
from tools.trident import *
from tools.common import *
from tools.fusion import *
from tools.common import _FCNHead

class DNANet(nn.Module):
    def __init__(self,inchans=1,channels=[16,32,64,128],layers=[2,2,2,2],dilations=[1,2,3]) -> None:
        super().__init__()
        self.encoder=TridentEncoder(inchans,channels,layers,dilations)
        self.conv1=nn.Sequential(
            nn.Conv2d(channels[0]*3,channels[0],1,1,0,bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU()
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(channels[1]*3,channels[1],1,1,0,bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU()
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(channels[2]*3,channels[2],1,1,0,bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU()
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(channels[3]*3,channels[3],1,1,0,bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.ReLU()
        )
        self.vit1=SwinViT(img_dim=256,in_channels=channels[0],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=8)
        self.vit2=SwinViT(img_dim=128,in_channels=channels[1],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=4)
        self.vit3=SwinViT(img_dim=64,in_channels=channels[2],embedding_dim=channels[2],
                          head_num=4, block_num=1, patch_dim=2)
        self.vit4=SwinTransformerBlock(c1=channels[3],c2=channels[3],num_heads=4,num_layers=1)
        self.botteneck=self.make_layer(ResidualBlock,
                                   3*channels[2]+channels[3]*2,
                                   channels[3],1)
        
        self.downsample2 = nn.Upsample(scale_factor=1/2, mode='bilinear', align_corners=True)
        self.downsample4 = nn.Upsample(scale_factor=1/4, mode='bilinear', align_corners=True)
        self.downsample8 = nn.Upsample(scale_factor=1/8, mode='bilinear', align_corners=True)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.fusion1=AsymBiChaFuseReduce(in_high_channels=channels[3],in_low_channels=channels[2],
                                         out_channels=channels[2])
        self.decoder3=self.make_layer(ResidualBlock,channels[2],channels[2],2)
        self.fusion2=AsymBiChaFuseReduce(in_high_channels=channels[2],in_low_channels=channels[1],
                                         out_channels=channels[1])
        self.decoder2=self.make_layer(ResidualBlock,channels[1],channels[1],2)
        self.fusion3=AsymBiChaFuseReduce(in_high_channels=channels[1],in_low_channels=channels[0],
                                         out_channels=channels[0])
        self.decoder1=self.make_layer(ResidualBlock,channels[0],channels[0],2)
        self.decoder0=self.make_layer(ResidualBlock,channels[0],channels[0],2)
        # self.head=_FCNHead(channels[0],1)
        self.newhead=nn.Sequential(
            nn.Conv2d(channels[0],1,1),
            nn.Sigmoid()
        )
        
    def make_layer(self,block,inchans,outchans,layers):
        layer=[]
        layer.append(block(inchans,outchans))
        for _ in range(layers-1):
            layer.append(block(outchans,outchans))
        return nn.Sequential(*layer)
    
    def forward(self,x):
        fm=self.encoder(x)
        x1=self.conv1(torch.cat([fm[0][0],fm[0][1],fm[0][2]],1))
        x2=self.conv2(torch.cat([fm[1][0],fm[1][1],fm[1][2]],1))
        x3=self.conv3(torch.cat([fm[2][0],fm[2][1],fm[2][2]],1))
        x4=self.conv4(torch.cat([fm[3][0],fm[3][1],fm[3][2]],1))
        # 修改后
        out=self.botteneck(torch.cat([self.vit1(x1),
                                      self.vit2(x2),
                                      self.vit3(x3),
                                      self.vit4(x4),x4],dim=1))
        out=self.up(out)
        out=self.fusion1(out,x3)
        out=self.decoder3(out)

        out=self.up(out)
        out=self.fusion2(out,x2)
        out=self.decoder2(out)

        out=self.up(out)
        out=self.fusion3(out,x1)
        out=self.decoder1(out)
        out=self.up(out)
        out=self.decoder0(out)
        out=self.newhead(out)
        return out
    

