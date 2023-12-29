import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

import sys
import scipy.ndimage as ndi
from scipy.interpolate import interpn, griddata
import os
from glob import glob
from PIL import Image

def get_index(path:str):
    return path.split("/")[-1].split(".")[0]

def crop(img, seg):
    ys, xs = np.where(seg > 0)
    y_min, y_max = np.min(ys), np.max(ys)
    x_min, x_max = np.min(xs), np.max(xs)
    img = img[y_min:y_max, x_min:x_max]
    seg = seg[y_min:y_max, x_min:x_max]
    if(img.shape[0] < 512):
        delta = 512 - img.shape[0]
        img = np.pad(img, ((delta//2, delta - delta//2),(0,0)))
    else:
        h = img.shape[0]
        img = img[h//2 - 256: h//2 + 256, :]

    if(img.shape[1] < 512):
        delta = 512 - img.shape[1]
        img = np.pad(img, ((0,0),(delta//2, delta - delta//2)))
    else:
        w = img.shape[1]
        img = img[:, w//2 - 256: w//2 + 256]
    return img, seg

def rotate_img(img, seg, theta):
    return ndi.rotate(img, theta, reshape=False, cval=255), ndi.rotate(seg, theta, reshape=False, cval=0)

def random_rotate(img, seg):
    theta = np.random.randint(-45, 45)
    return *rotate_img(img, seg, theta), theta

def transform(center, theta):
    # order: (x, y)
    mat = np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))], 
                    [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
    def fn(points): # n, 2
        # 顺时针旋转theta
        points_centered = points - center[None]
        return np.matmul(mat, points_centered.T).T + center[None]
    return fn

def collate_fn_exclude_none(batch):
    new_batch = list(filter(lambda e: e is not None, batch))
    if len(new_batch) == 0:
       return None
    else:
       return default_collate(new_batch)

class MatchDataset(Dataset):
    def __init__(self,):
        super(MatchDataset,self).__init__()
        root1 = "/disk1/dyj/data/finger/NIST4"
        img1 = glob(os.path.join(root1, "image/*.bmp"))
        seg1 = glob(os.path.join(root1, "image_feature/estimation/seg/fingernet/*.png"))
        img1 = sorted(img1, key=get_index)
        seg1 = sorted(seg1, key=get_index)

        root2 = "/disk1/dyj/data/finger/NIST14"
        img2 = glob(os.path.join(root2, "image/*.bmp"))
        seg2 = glob(os.path.join(root2, "image_feature/estimation/seg/fingernet/*.png"))
        img2 = sorted(img2, key=get_index)
        seg2 = sorted(seg2, key=get_index)

        self.imgs = img1 + img2
        self.segs = seg1 + seg2


    def __getitem__(self,index):
        img = np.asarray(Image.open(self.imgs[index]))
        seg = np.asarray(Image.open(self.segs[index]))

        img, seg = crop(img, seg)
        h, w = img.shape
        center = np.array([w/2, h/2])
        if(h <= 200 or w <= 400):
            print("error, image too small")
            return None
        patch_h, patch_w = 64, 320
        ########################
        # sample the start point
        rotated_img, rotated_seg, theta = random_rotate(img, seg)
        pts_x = np.random.randint(30, w - patch_w)
        pts_y = np.random.randint(30, h - patch_h)
        patch = rotated_img[pts_y:pts_y+patch_h, pts_x:pts_x+patch_w]

        patch_pts = np.stack(np.meshgrid(np.arange(0, patch_w), np.arange(0, patch_h)), -1)
        patch_pts = patch_pts.reshape(-1, 2)
        patch_pts[:,0] += pts_x
        patch_pts[:,1] += pts_y

        ####################################################
        # get the location in the original coordinate system
        patch_pts = transform(center, theta)(patch_pts)
        patch_pts[patch_pts < 0] = 0
        xs = patch_pts[:, 0]
        x_mask = xs > w-1
        patch_pts[:, 0][x_mask] = w - 1
        ys = patch_pts[:, 1]
        y_mask = ys > h - 1
        patch_pts[:, 1][y_mask] = h - 1

        #######
        # debug
        img_sampled = interpn((np.arange(0, h), np.arange(0, w)), img, patch_pts[:, ::-1]) # should be in (y,x) order
        img_sampled = img_sampled.reshape(patch_h, patch_w)
        fig,axes = plt.subplots(3,2)
        axes[0,0].imshow(img,'gray')
        axes[0,0].scatter(patch_pts[::1000,0], patch_pts[::1000,1], color="red", s=5)
        axes[0,1].imshow(seg,'gray')
        axes[1,0].imshow(rotated_img,'gray')
        axes[1,1].imshow(rotated_seg,'gray')
        axes[2,0].imshow(img_sampled,'gray')
        axes[2,1].imshow(patch,'gray')
        plt.savefig("./debug.png")
        
        #################
        # post-processing
        img = (img - 127.5)/127.5
        patch = (patch - 127.5)/127.5
        img = img[None]
        patch = patch[None]

        return img.astype(np.float32), patch.astype(np.float32), patch_pts.reshape(patch_h, patch_w, 2).astype(np.float32), np.array([pts_x/w, pts_y/h, np.deg2rad(theta)]).astype(np.float32)

    def __len__(self,):
        return len(self.imgs)

if __name__ == "__main__":
    data = MatchDataset()
    dataloader = DataLoader(data, 1, shuffle = True)
    next(iter(dataloader))
