# coding=utf-8

from http.client import LENGTH_REQUIRED
import numpy as np
import cv2
import torch
from PIL import Image
from torch import nn
import torch.nn.functional
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random


#Numpy，透视，高斯，逐点
def cover1(ori_img, attack_img, ori_pts, test_pts,device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    attack_img: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    M = cv2.getPerspectiveTransform(test_pts,ori_pts)
    W = ori_img.shape[1]
    H = ori_img.shape[0]
    out_img = cv2.warpPerspective(attack_img, M, (W,H))
    out_img_blur = cv2.GaussianBlur(out_img, (3, 3), 0)
    # out_img_blur = out_img.copy()
    for i in range(ori_img.shape[0]):
        for j in range(ori_img.shape[1]):
            if not np.all(out_img[i,j,:]==0):
                ori_img[i,j,:] = out_img_blur[i,j,:] * 0.73
    return ori_img

#Numpy，透视，高斯，mask
def cover12(ori_img, attack_img, ori_pts, test_pts,device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    attack_img: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    M = cv2.getPerspectiveTransform(test_pts,ori_pts)
    W = ori_img.shape[1]
    H = ori_img.shape[0]
    out_img = cv2.warpPerspective(attack_img, M, (W,H))
    out_img_blur = cv2.GaussianBlur(out_img, (3, 3), 0)
    # out_img_blur = out_img.copy()

    mask0=np.ones_like(attack_img)
    mask=cv2.warpPerspective(mask0, M, (W,H))
    ori_img=np.multiply(ori_img,1-mask)+out_img_blur*0.73

    return ori_img

#patch*mask，Numpy，透视，高斯，mask
def cover12_mask(ori_img, attack_img, mask, ori_pts, test_pts,device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    attack_img: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    M = cv2.getPerspectiveTransform(test_pts,ori_pts)
    W = ori_img.shape[1]
    H = ori_img.shape[0]
    out_img = cv2.warpPerspective(attack_img, M, (W,H))
    out_img_blur = cv2.GaussianBlur(out_img, (3, 3), 0)
    # out_img_blur = out_img.copy()

    mask=cv2.warpPerspective(mask, M, (W,H))
    mask0 = mask/255.0
    ori_img=np.multiply(ori_img,1-mask0)+np.multiply(mask0,out_img_blur*0.73)

    return ori_img

#Numpy，resize，无高斯，区域替代
def cover11(ori_img, attack_img, ori_pts, test_pts,device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    attack_img: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''

    x1=int((ori_pts[0][0]+ori_pts[3][0])/2)
    y1=int((ori_pts[0][1]+ori_pts[1][1])/2)
    x2=int((ori_pts[1][0]+ori_pts[2][0])/2)
    y2=int((ori_pts[2][1]+ori_pts[3][1])/2)
    #print(ori_pts)

    #resize
    h=y2-y1+1
    w=x2-x1+1
    out_img=cv2.resize(attack_img,(w,h),interpolation=cv2.INTER_NEAREST)
    ori_img[y1:y2+1,x1:x2+1,:]=out_img*0.73
    return ori_img

#Torch，透视，无高斯，逐点
def cover2(ori_img, newimg, ori_pts, test_pts,device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    newimg: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''

    #透视变换
    out_img = Torch_trans(newimg, ori_pts,device)
    # 高斯滤波
    #out_img_blur = cv2.GaussianBlur(out_img, (3, 3), 0)
    # out_img_blur = out_img.copy()

    for i in range(ori_img.shape[0]):
        print(i)
        for j in range(ori_img.shape[1]):
            if not torch.all(out_img[i, j, :] ==0):
                ori_img[i, j, :] = out_img[i, j, :] * 0.73

    return ori_img

#Torch，resize，无高斯，区域替代
def cover(ori_img, texture, ori_pts, test_pts,device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    texture: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    texture = torch.clamp(texture, 0, 255)
    x1=int((ori_pts[0][0]+ori_pts[3][0])/2)
    y1=int((ori_pts[0][1]+ori_pts[1][1])/2)
    x2=int((ori_pts[1][0]+ori_pts[2][0])/2)
    y2=int((ori_pts[2][1]+ori_pts[3][1])/2)

    #resize
    h=y2-y1+1
    w=x2-x1+1
    out_img=nn.functional.interpolate(texture.unsqueeze(0).unsqueeze(0), (h, w, 3), mode='trilinear', align_corners=True)
    out_img=out_img.reshape(h, w, 3)

    # 高斯滤波
    '''out_img = out_img.permute((2, 0, 1))
    out_img=out_img.unsqueeze(0).unsqueeze(0)
    conv1 = nn.Conv3d(3, 3, (1, 3, 3))
    print(conv1)
    w1 = torch.as_tensor(np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]]*3).reshape(3,3, 1, 3, 3)).to(device).float()

    conv1.weight = nn.Parameter(w1)
    out_img1 = conv1(out_img).permute((1, 2, 0))'''

    ori_img[y1:y2+1,x1:x2+1,:]=out_img*0.73

    return ori_img

#Torch，透视，高斯，mask
def cover0(ori_img, texture, ori_pts, test_pts, device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    texture: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    #透视变换
    texture = torch.clamp(texture, 0, 255)
    print(torch.min(texture), torch.max(texture))
    texture= texture.permute((2,0,1))
    texture = TF.perspective( texture, [test_pts[0], test_pts[3], test_pts[1], test_pts[2]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    #newimg = newimg.permute((1,2,0))

    # 高斯滤波
    put_img=TF.gaussian_blur( texture, 3, 1)
    put_img =  texture.permute((1,2,0))

    #贴图
    mask0=Variable(torch.ones(3,1080,1920).to(device))
    mask=TF.perspective(mask0, [[0,0], [0,1079], [1919,0], [1919,1079]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    mask=mask.permute((1,2,0))
    ori_img=torch.mul(ori_img,1-mask)+put_img[0:ori_img.shape[0],0:ori_img.shape[1],:]*0.73
    return ori_img

#pacth*mask，Torch，透视，高斯，mask
def cover0_mask(ori_img, texture, mask, ori_pts, test_pts, device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    texture: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    #透视变换
    texture=torch.clamp(texture,0, 255)
    texture=texture.permute((2,0,1))
    texture = TF.perspective(texture, [test_pts[0], test_pts[3], test_pts[1], test_pts[2]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])

    # 高斯滤波
    put_img=TF.gaussian_blur(texture, 3, 1)
    put_img = put_img.permute((1,2,0))

    #贴图
    mask0=mask.permute((2,0,1))
    mask0=TF.perspective(mask0, [[0,0], [0,1259], [2789,0], [2789,1259]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    mask0=mask0.permute((1,2,0))/255.0
    ori_img=torch.mul(1-mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],ori_img)+torch.mul(mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],put_img[0:ori_img.shape[0],0:ori_img.shape[1],:]*0.73)
    return ori_img


def cover02_mask_turb(ori_img, texture, mask, ori_pts, test_pts, frame_index, scene_index):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    texture: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    rand_ori_x = random.randint(-frame_index // 60, frame_index // 60)
    rand_ori_y = random.randint(-frame_index // 110, frame_index // 110)
    ori_pts[:,0]+=rand_ori_x
    ori_pts[:,1]+=rand_ori_y
    brightness = random.randint(0, 10)

    #高斯滤波
    texture=torch.clamp(texture,0, 255)
    texture=torch.mul(texture,mask/255)      #
    texture=texture.permute((2,0,1))
    texture=TF.gaussian_blur(texture, 9, 1)

    #透视变换
    put_img = TF.perspective(texture, [test_pts[0], test_pts[3], test_pts[1], test_pts[2]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])

    #再模糊
    put_img = TF.gaussian_blur(put_img, 3, 1)
    put_img = put_img.permute((1, 2, 0))

    #贴图
    mask0=mask.permute((2,0,1))
    mask0=TF.perspective(mask0, [[0,0], [0,1259], [2789,0], [2789,1259]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    mask0=mask0.permute((1,2,0))/255.0

    put_img=put_img*0.73
    if scene_index == 4:
        put_img += 100-frame_index//4.3
    put_img+=brightness
    put_img=torch.clamp(put_img,0,255)

    ori_img=torch.mul(1-mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],ori_img)+torch.mul(mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],put_img[0:ori_img.shape[0],0:ori_img.shape[1],:])
    return ori_img



#pacth*mask，Torch，透视，高斯，mask
def cover0_mask_robust(ori_img, texture, mask, ori_pts, test_pts, frame_index, scene_index):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    texture: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    # 坐标扰动
    ori_pts[:,0] += random.randint(-frame_index // 60, frame_index // 60)
    ori_pts[:,1] += random.randint(-frame_index // 110, frame_index // 110)

    texture=torch.clamp(texture,0, 255)
    texture=texture.permute((2,0,1))

    # 高斯滤波
    # sigma = round(np.random.random() * 3 + 0.1, 3) 
    # kernel = np.random.random()
    # if kernel < 0.2:
    #     kernel = 3
    # elif kernel < 0.5:
    #     kernel = 5
    # else:
    #     kernel = 7
    # texture=TF.gaussian_blur(texture, kernel, sigma)
    texture=TF.gaussian_blur(texture, 9, 1)

    # 透视变换
    texture = TF.perspective(texture, [test_pts[0], test_pts[3], test_pts[1], test_pts[2]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    
    # 再模糊
    put_img = TF.gaussian_blur(texture, 3, 1)
    
    # 随机brightness/contrast/saturation/hue变换
    put_img = put_img / 255
    put_img = transforms.ColorJitter(brightness=0.3, contrast=0.1, saturation=0.0, hue=0.0)(put_img)
    put_img *= 255

    # # 随机旋转
    # put_img = transforms.RandomRotation(degrees=15)(put_img)

    # # 随机crop and resize
    # _, h, w = put_img.shape
    # put_img = transforms.RandomResizedCrop(size=(h,w), scale=(0.9,1))(put_img)


    #贴图
    put_img = put_img.permute((1,2,0))
    put_img *= 0.73

    if scene_index == 4:  # 大雾场景
        put_img += 100-frame_index//4.3
    put_img=torch.clamp(put_img,0,255)

    mask0=mask.permute((2,0,1))
    mask0=TF.perspective(mask0, [[0,0], [0,1259], [2789,0], [2789,1259]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    mask0=mask0.permute((1,2,0))/255.0
    ori_img=torch.mul(1-mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],ori_img)+torch.mul(mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],put_img[0:ori_img.shape[0],0:ori_img.shape[1],:])
    return ori_img


#pacth*mask，Torch，透视，高斯，mask，调序
def cover01_mask(ori_img, texture, mask, ori_pts, test_pts, device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    texture: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    #高斯滤波
    texture=torch.clamp(texture,0, 255)
    texture=texture.permute((2,0,1))
    texture=TF.gaussian_blur(texture, 9, 1)

    #透视变换
    put_img = TF.perspective(texture, [test_pts[0], test_pts[3], test_pts[1], test_pts[2]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    put_img = put_img.permute((1, 2, 0))

    #贴图
    mask0=mask.permute((2,0,1))
    mask0=TF.perspective(mask0, [[0,0], [0,1259], [2789,0], [2789,1259]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    mask0=mask0.permute((1,2,0))/255.0
    ori_img=torch.mul(1-mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],ori_img)+torch.mul(mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],put_img[0:ori_img.shape[0],0:ori_img.shape[1],:]*0.73)
    return ori_img


#pacth*mask，Torch，透视，高斯，mask，调序，再模糊
def cover02_mask(ori_img, texture, mask, ori_pts, test_pts, device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    texture: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    #高斯滤波
    texture=torch.clamp(texture,0, 255)
    texture=texture.permute((2,0,1))
    texture=TF.gaussian_blur(texture, 9, 1)

    #透视变换
    put_img = TF.perspective(texture, [test_pts[0], test_pts[3], test_pts[1], test_pts[2]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])

    #再模糊
    put_img = TF.gaussian_blur(put_img, 3, 1)
    put_img = put_img.permute((1, 2, 0))

    #贴图
    mask0=mask.permute((2,0,1))
    mask0=TF.perspective(mask0, [[0,0], [0,1259], [2789,0], [2789,1259]],
                         [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    mask0=mask0.permute((1,2,0))/255.0
    ori_img=torch.mul(1-mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],ori_img)+torch.mul(mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],put_img[0:ori_img.shape[0],0:ori_img.shape[1],:]*0.73)
    return ori_img

def EOT_stage1(ori_img, texture, mask, ori_pts, test_pts, device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    texture: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    #透视变换
    texture=torch.clamp(texture,0, 255)
    texture=texture.permute((2,0,1))
    texture = TF.perspective(texture, [test_pts[0], test_pts[3], test_pts[1], test_pts[2]],[ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])

    #高斯滤波
    put_img=TF.gaussian_blur(texture, 3, 1)
    put_img = put_img.permute((1,2,0))

    #贴图
    mask0=mask.permute((2,0,1))
    mask0=TF.perspective(mask0, [[0,0], [0,1259], [2789,0], [2789,1259]],[ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
    mask0=mask0.permute((1,2,0))/255.0
    ori_img=torch.mul(1-mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],ori_img)+torch.mul(mask0[0:ori_img.shape[0],0:ori_img.shape[1],:],put_img[0:ori_img.shape[0],0:ori_img.shape[1],:]*0.73)
    return ori_img


def EOT_stage2(ori_img, texture, mask, mode, prameters, ori_pts, test_pts, device):
    '''
    ori_img: 未放置对抗补丁的原五个场景中帧图片
    texture: 对抗补丁
    ori_pts: 原帧图片中车厢四个顶点的像素坐标(dtype=np.float32)
    test_pts: 对抗补丁对应四个顶点的坐标(dtype=np.float32)
    return: 覆盖对抗样本之后的帧图片
    '''
    if mode==0:
        #仿射变换
        texture0 = torch.zeros_like(ori_img).to(device).float()  #patch扩充至原图大小
        texture0[0:texture.shape[0], 0:texture.shape[1], :] = torch.clamp(texture, 0, 255)
        texture0 = texture0.permute((2, 0, 1))
        texture0 = TF.affine(texture0,prameters[0],[prameters[1],prameters[2]],prameters[3],[prameters[4],prameters[5]])
        put_img = texture0.permute((1, 2, 0))

        #贴图
        mask0 = torch.zeros_like(ori_img).to(device).float()      #mask扩充至原图大小
        mask0[0:mask.shape[0],0:mask.shape[1],:] += mask
        mask0=mask0.permute((2,0,1))
        mask0=TF.affine(mask0,prameters[0],[prameters[1],prameters[2]],prameters[3],[prameters[4],prameters[5]])
        mask0=mask0.permute((1,2,0))/255.0
        ori_img=torch.mul(1-mask0,ori_img)+torch.mul(mask0,put_img*0.73)

    elif mode==1:
        #透视变换
        texture0 = torch.zeros_like(ori_img).to(device).float()     #patch扩充至原图大小
        texture0[0:texture.shape[0],0:texture.shape[1],:] = torch.clamp(texture,0, 255)
        texture0=texture0.permute((2,0,1))
        texture0 = TF.perspective(texture0, [test_pts[0], test_pts[3], test_pts[1], test_pts[2]],
                             [ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
        put_img = texture0.permute((1,2,0))

        #贴图
        mask0 = torch.zeros_like(ori_img).to(device).float()      #mask扩充至原图大小
        mask0[0:mask.shape[0],0:mask.shape[1],:] += mask
        mask0=mask0.permute((2,0,1))
        mask0=TF.perspective(mask0, [[0,0], [0,359], [639,0], [639,359]],[ori_pts[0], ori_pts[3], ori_pts[1], ori_pts[2]])
        mask0=mask0.permute((1,2,0))/255.0
        ori_img=torch.mul(1-mask0,ori_img)+torch.mul(mask0,put_img*0.73)

    return ori_img


#torch仿射变换
def Torch_trans(newimg,ori_pts,device):
    newimg = newimg.permute((2, 0, 1))
    theta = torch.tensor([
        [2, 0, 0],
        [0, 2, 0]
    ], dtype=torch.float)

    grid = F.affine_grid(theta.unsqueeze(0), newimg.unsqueeze(0).size()).to(device)
    output = F.grid_sample(newimg.unsqueeze(0), grid)
    new_img_torch = output[0]
    new_img_torch = new_img_torch.permute((1, 2, 0))
    return new_img_torch

#生成透视矩阵
def WarpPerspectiveMatric(src, dst):
    assert src.shape[0] == dst.shape[0] and src.shape[0] >= 4
    nums = src.shape[0]
    A = np.zeros((2 * nums, 8))  # A*warpMatric=B
    B = np.zeros((2 * nums, 1))
    for i in range(0, nums):
        A_i = src[i, :]
        B_i = dst[i, :]
        A[2 * i, :] = [A_i[0], A_i[1], 1,
                       0, 0, 0,
                       -A_i[0] * B_i[0], -A_i[1] * B_i[0]]
        B[2 * i] = B_i[0]

        A[2 * i + 1, :] = [0, 0, 0,
                           A_i[0], A_i[1], 1,
                           -A_i[0] * B_i[1], -A_i[1] * B_i[1]]
        B[2 * i + 1] = B_i[1]

    A = np.mat(A)
    warpMatric = A.I * B  # 求出a_11, a_12, a_13, a_21, a_22, a_23, a_31, a_32

    # 之后为结果的后处理
    warpMatric = np.array(warpMatric).T[0]
    warpMatric = np.insert(warpMatric, warpMatric.shape[0], values=1.0, axis=0)  # 插入a_33 = 1
    warpMatric = warpMatric.reshape((3, 3))
    return warpMatric


if __name__ == '__main__':
    device='cuda'
    ori_img = cv2.imread('./images/img01_none190_.jpg')
    ori_img = Variable(torch.from_numpy(np.array(ori_img)).to(device).float())
    attack_img0 = cv2.imread('./patch/texture.png')
    mask = cv2.imread('./mask/mask22.png')

    attack_img = Variable(torch.from_numpy(np.array(attack_img0)).to(device).float())
    mask = Variable(torch.from_numpy(np.array(mask)).to(device).float())

    #attack_img = np.ones((1260, 2790, 3), dtype=np.uint8)*255
    pts = np.load('./pts/data01.npy')
    ori_pts = np.around(pts[:,190].reshape(4,2)).astype(np.float32)
    test_pts = np.array([[0,0],[2790-1,0],[2790-1,1260-1],[0,1260-1]], dtype=np.float32)

    cv2.imwrite('/home/zjh/aisec/ori_yolov3/data/images/res_1_190.png', np.uint8(cover02_mask(ori_img, attack_img, mask, ori_pts, test_pts,device).cpu()))

    #for i in range(len(pts[1])):
    #    ori_pts=np.around(pts[:,i].reshape(4,2))
    #    print(np.array([abs(ori_pts[0][1] - ori_pts[1][1]), abs(ori_pts[1][0] - ori_pts[2][0]), abs(ori_pts[2][1] - ori_pts[3][1]), abs(ori_pts[0][0] - ori_pts[3][0])]))
