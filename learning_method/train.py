import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import sys
import scipy.ndimage as ndi
from scipy.interpolate import interpn, griddata
import os
from glob import glob
from PIL import Image
from model import MatchNet
from dataset import MatchDataset

if __name__ == "__main__":
    model = MatchNet().cuda()
    dataset = MatchDataset()
    dataloader = DataLoader(dataset, 64, shuffle = True)
    loss_node = nn.MSELoss().cuda()
    optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
    for epoch in range(10):
        for i, item in enumerate(dataloader):
            img, patch, patch_pts, params = item
            img = img.cuda()
            patch = patch.cuda()
            patch_pts = patch_pts.cuda()
            params = params.cuda()

            output = model(img, patch)
            y, img_desc, patch_desc = output

            img_desc_size = img_desc.shape[-2:] # 32, 32
            patch_desc_size = patch_desc.shape[-2:]

            B = img.shape[0]
            locs = patch_pts[:, 8::16, 8::16, :]
            locs = (locs - 8) / 16

            locs[..., 0] = (locs[..., 0] - img_desc_size[1] / 2) / (img_desc_size[1] / 2)
            locs[..., 1] = (locs[..., 1] - img_desc_size[0] / 2) / (img_desc_size[0] / 2)

            interp_feature = F.grid_sample(img_desc, locs)

            loss1 = loss_node(interp_feature, patch_desc)
            loss2 = loss_node(y, params)
            loss = loss1 + 500 * loss2

            x_error = torch.mean(torch.abs(y[:, 0] - params[:, 0]) * 512)
            y_error = torch.mean(torch.abs(y[:, 1] - params[:, 1]) * 512)
            theta_error = torch.mean(torch.abs(torch.rad2deg(y[:, 2]) - torch.rad2deg(params[:, 2])))
            print(f"loss: {loss.item()}, loss1: {loss1.item()}, x: {x_error.item()}, y: {y_error.item()}, theta: {theta_error.item()}")

            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f"Epoch:{epoch} end")
        if epoch % 5 == 0:
            print("save!!!")
            torch.save({"model":model.state_dict(),
                        "optim":optim.state_dict()}, f"./epoch_{epoch}.pth.tar")
