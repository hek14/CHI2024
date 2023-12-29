import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import sys
import scipy.ndimage as ndi
from scipy.interpolate import interpn, griddata
import os
from glob import glob
from PIL import Image

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder
class DesNet(nn.Module):
    def __init__(self,):
        super(DesNet,self).__init__()
        self.encoder = smp.encoders.get_encoder("resnet18", in_channels=1, depth=5)
        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=(256, 128),
            n_blocks=2,
            use_batchnorm=True,
            center=False,
            attention_type=None
        )
        self.feature_head = nn.Sequential(nn.Conv2d(128, 64, 5, 2, 2), nn.ReLU())

    def forward(self,x):
        print("input shape: ",x.shape)
        features = self.encoder(x)
        x = self.decoder(*features)
        x = self.feature_head(x) # 1/16
        print("desc feature: ",x.shape, x.min(), x.max())
        return x

class MatchNet(nn.Module):
    def __init__(self,):
        super().__init__()
        self.desc_head = DesNet()
        # 512/16, 64/16, 320/16

        # orientation head
        self.ori_head = nn.Conv2d(64, 2, 3, 1, 1)

        # cross-attention part
        self.to_q = nn.Linear(64, 64, bias=False)
        self.to_k = nn.Linear(64, 64, bias=False)
        self.to_v = nn.Linear(64, 64, bias=False)

        self.multihead_attn = nn.MultiheadAttention(64, num_heads=8, batch_first=True)

        # final prediction
        self.fc = nn.Linear(4 * 20 * 64, 3) # 4*20: patch dense grid size, 64: channel

    def forward(self, full, patch):
        # get dense/flatten feature
        full_desc = self.desc_head(full)
        full_desc_dense = full_desc
        B, C = full_desc.shape[0], full_desc.shape[1]
        full_desc = full_desc.view(B, C, -1)
        full_desc = torch.permute(full_desc, (0, 2, 1))
        patch_desc = self.desc_head(patch)
        patch_desc_dense = patch_desc
        patch_desc = patch_desc.view(B, C, -1)
        patch_desc = torch.permute(patch_desc, (0, 2, 1))

        # ori head
        full_ori = self.ori_head(full_desc_dense)
        patch_ori = self.ori_head(patch_desc_dense)

        print("full_ori: ", full_ori.shape, patch_ori.shape)

        # attention and final pred
        query = self.to_q(patch_desc)
        key = self.to_k(full_desc)
        value = self.to_v(full_desc)
        print("qkv shape: ",query.shape, query.min(), query.max(), key.shape, key.min(), key.max(), value.shape, value.min(), value.max())
        
        # cross-attn
        attn_out, _ = self.multihead_attn(query, key, value)
        print("attn_out shape: ",attn_out.shape, attn_out.min(), attn_out.max())

        attn_out = attn_out.reshape(B, -1)
        y = self.fc(attn_out)
        
        return {
                "pred": y,
                "full_desc": full_desc_dense,
                "full_ori": full_ori,
                "patch_desc": patch_desc_dense,
                "patch_ori": patch_ori,
                }
        
        


if __name__ == "__main__":
    m = MatchNet()
    full = torch.randn(4, 1, 512, 512)
    patch = torch.randn(4, 1, 64, 320)
    output = m(full, patch)
    # for k,v in output:
    #     print(e.shape, e.min(), e.max())
