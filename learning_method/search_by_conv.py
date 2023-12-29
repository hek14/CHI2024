import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import cv2
import sys
sys.path.append("/home/heke/codes_med33/CHI2024/")
from fptools.fp_segmtation import segmentation_coherence

def localEqualHist(img):
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(32,32))
    dst = clahe.apply(img)
    return dst

def patchfy(img: torch.Tensor,  p1, p2):
    h,w = img.shape
    img = img.reshape(h//p1, p1, w//p2, p2)
    img = torch.einsum('hpwq->hwpq',img)
    img = img.flatten(0,1) # n p q
    # img = img.reshape(-1, p1*p2)
    return img

def normalize(x: torch.Tensor):
    x1 = (x - x.min())/(x.max() - x.min() + 1e-4)
    return x1 * 2 - 1

def show_all_weights(weights):
    number = min(weights.shape[0], 10)
    fig,axes = plt.subplots(1,number)
    for i in range(number):
        axes[i].imshow(weights[i, 0].numpy(), 'gray')
    plt.savefig("./weights.png")

template = np.array(Image.open("../template_hk_r_index.png"))
h_template, w_template = template.shape
template_t = localEqualHist(template)

seg = segmentation_coherence(template_t, convex=True)
h, w = np.where(seg > 0)
min_h, max_h = np.min(h), np.max(h)
min_w, max_w = np.min(w), np.max(w)
ignore = 20
template_crop = template_t[min_h+ignore:max_h-ignore, min_w+ignore:max_w-ignore]
t = torch.Tensor(template_crop)[None,None]
t = t.to(torch.float32)
t = normalize(t)

search = np.array(Image.open("../data2/199.png"))
h, w = search.shape
search = search[:, w//2:]
search = localEqualHist(search)
print("search size: ", h, w)

weights = torch.Tensor(search)
weights = normalize(weights)
weights = patchfy(weights, 36, 32)
weights = weights.to(torch.float32)
weights = weights.unsqueeze(1) # 20, 1, 36, 32
print("weights.shape: ",weights.shape)

show_all_weights(weights)

output = F.conv2d(t, weights)
output = output.permute(1, 0, 2, 3)
print(output.shape, output.min(), output.max()) # 1, 5, 72, 64
#
# # compute patch std of output: std = E((x - Ex)**2)
# k_sz = 31;
# weights_avg = torch.ones(1, 1, k_sz, k_sz).to(torch.float32)
# weights_avg /= k_sz**2
# mean_output = F.conv2d(output, weights_avg, padding = k_sz//2)
#
# std_output = (output - mean_output)**2
#
# c = std_output.shape[0]
# fig,axes = plt.subplots(2, c)
# for i in range(c):
#     axes[0, i].imshow(weights[i,0], 'gray')
#     axes[1, i].imshow(std_output[i,0], 'gray')
# plt.savefig('./weights.png')
