# YOLOv3 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov3.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""




import argparse
import os
import sys
import time
from pathlib import Path
import time
import cv2
import torch
import torchvision
from torch import nn
from data.warp import cover0_mask,cover0_mask_robust
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from utils.augmentations import letterbox


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression1,print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import matplotlib.pyplot as plt

def run(weights=ROOT / 'yolov3.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=640,  # inference size (pixels)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name

        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    

    def l1_norm(tensor):
        return torch.sum(torch.abs(tensor))
    
    def l2_norm(tensor):
        return torch.sqrt(torch.sum(torch.pow(tensor, 2)))

    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images, True
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    print('source:', source)

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    #device=torch.device('cpu')
    model = DetectMultiBackend(weights, device=device, dnn=dnn).to(device).eval()  #.eval()

    stride, names, pt, jit, onnx = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    #dt, seen = [0.0, 0.0, 0.0], 0

    #参数
    batch_size=12
    learning_rate=0.5
    max_epoches = 1000  # 最大迭代轮数
    thresh=0.24
    batch_num=int(len(dataset)/batch_size)
    print("batch_num:%d\n"%batch_num)

    #类别下标
    #three=[7,10,12]
    target = 14  # 14->bird
    target = np.eye(80,80)[target]
    other=[i+5 for i in range(80)]
    del other[7],other[5],other[2]

    #导入mask
    mask=cv2.imread('./data/mask/mask22.png')
    # print(type(mask))
    # print(type(np.array(mask)))
    mask=Variable(torch.from_numpy(np.array(mask)).to(device).float())

    #定义优化的patch
    ori_modifier = cv2.imread('./data/result/patch_125.png')
    modifier=Variable(torch.from_numpy(ori_modifier).to(device).float())
    # ori_patch = cv2.imread('./data/result/patch_80.png')
    # ori_patch = torch.from_numpy(ori_patch).to(device).float()  # shape=(1260,2790,3)
    #modifier*=255

    #设置为不保存梯度值
    for param in model.parameters():
       param.requires_grad = False

    # 图像数据的扰动量梯度可以获取
    modifier.requires_grad = False

    #定义优化器 仅优化modifier
    optimizer = torch.optim.Adam([modifier],lr=learning_rate)

    #迭代
    loss_plot=[[] for _ in range(batch_num)]
    iteration_plot=[]
    epoch_num = 0
    epoch_loss_plot=[]
    epoch_plot=[]
    success_plot = []
    success_plot_two = []

    loss = Variable(torch.tensor(0).to(device).float())

    for epoch in range(1,max_epoches+1):
        print("--------------- Epoch:"+str(epoch)+" ---------------")
        flag = 0             #攻击帧数
        target_flag = 0      #定向攻击帧数
        flag_two = 0
        count = 0          #总帧数
        iteration = 0
        epoch_loss = 0
        epoch_num += 1
        start_time = time.time()

        for path, _, im0s, __, ___ in dataset:
            # print('path:', path)  # path: C:\Users\zhengjunhao\Desktop\AISEC\yolov3\data\images\img02_none112.jpg
            # print('im0s.shape:',im0s.shape)  # (1080, 1920, 3) 原图大小
            # print('len(dataset)', len(dataset)) # source路径下图片数量
            # time.sleep(1000)
            flag0 = 1
            flag1 = 0
            flag0_two = 1
            count+=1

            # img02_none112.jpg: 2场景/112帧
            if (count-1)%batch_size==0:
                iteration += 1
                optimizer.zero_grad()  # 梯度清零
                loss=Variable(torch.tensor(0).to(device).float())

            scene_index = int(path.split("_")[0][-1])  # [1,5]
            frame_index = int(path.split("_")[1][4:])  # [0,230]

            #%背景图加patch
            im1s = Variable(torch.from_numpy(np.array(im0s)).to(device).float())
            pts = np.load('./data/pts/data0' + str(scene_index) + '.npy')  # 场景的所有坐标8*231
            ori_pts = np.around(pts[:, frame_index].reshape(4, 2)).astype(np.float32)  # 背景图对应的坐标

            test_pts = np.array([[0, 0], [2790 - 1, 0], [2790 - 1, 1260 - 1], [0, 1260 - 1]], dtype=np.float32)  #patch对应的坐标
            im11=cover0_mask(im1s, modifier, mask, ori_pts, test_pts, device)   # (1080, 1920, 3)
            out=im11.cpu().detach().numpy().copy()

            # 加patch后的图像预处理 #
            #法一padding
            # im1 = nn.functional.interpolate(im11.unsqueeze(0).unsqueeze(0), (360, 640,3),mode='trilinear', align_corners=True)
            # im1 = im1.reshape(360, 640, 3)
            im1 = im11.permute((2, 0, 1))   # (3, 1080, 1920)
            aug = torchvision.transforms.Resize([360,640])
            im1 = aug(im1)  # (3, 360, 640)
            im1 = torchvision.transforms.Pad([0, 12, 0, 12], fill=114, padding_mode="constant")(im1)  #(3, 384, 640)
            im=im1+0
            im[0, :, :] = im1[2, :, :]   #RGB转换
            im[2, :, :] = im1[0, :, :]

            im = im.half() if half else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim ，(1,3,H,W)

            # #法二resize
            # im1_two = nn.functional.interpolate(im11.unsqueeze(0).unsqueeze(0), (384, 640, 3), mode='trilinear', align_corners=True)
            # im1_two = im1_two.reshape(384, 640, 3)
            # im1_two = im1_two.permute((2, 0, 1))  # (3, 384, 640)
            # im_two=im1_two+0
            # im_two[0, :, :] = im1_two[2, :, :]   #RGB转换
            # im_two[2, :, :] = im1_two[0, :, :]

            # im_two = im_two.half() if half else im_two.float()  # uint8 to fp16/32
            # im_two /= 255  # 0 - 255 to 0.0 - 1.0
            # if len(im_two.shape) == 3:
            #     im_two= im_two[None]  # expand for batch dim ，(1,3,H,W)


            # Inference
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)     #(1,num_box,85), 85为box(4)+box_conf(1)+cls_conf(80)
            pred1 = non_max_suppression1(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            pred=pred[pred[...,4]>thresh]   #取检测框概率>thresh的结果

            # 定义损失函数 #
            # target_attack: 14-bird
            # if len(pred) == 0:
            #     loss_cls = torch.as_tensor(0)
            # else:
            #     target1 = torch.from_numpy(np.array([target for i in range(pred.shape[0])])).to(device).float()
            #     dif = Variable(pred[:, 5:] - target1).to(device).float()
            #     for i in range(pred.shape[0]):
            #         dif[i,:] *= pred[i,4]
            #     loss_r1 = l1_norm(dif)  # l1范数
            #     if pred1[0].shape[0] == 0:
            #         loss_r2 = torch.as_tensor(0)
            #     else:
            #         target_box = ori_pts[[0,2]]
            #         target_box[:, 0] = target_box[:, 0]
            #         target_box[:, 1] = target_box[:, 1]
            #         target_box1 = np.array([target_box.reshape(1,-1)[0] for i in range(pred1[0].shape[0])])
            #         target_box1 = torch.from_numpy(target_box1).to(device).float()
            #         pred_box = pred1[0][:,:4]
            #         loss_r2 = l2_norm(pred_box - target_box1)
            #     loss_cls = loss_r1 + 0.02 * loss_r2  # dif_conf + dif_bbox
            #     # print('loss_r1: ', loss_r1)
            #     # print('loss_r2: ', loss_r2)
            # # #法一
            # # if len(pred) == 0:
            # #     loss_cls = torch.as_tensor(0)
            # # else:
            # #    dif = Variable(torch.ones(len(pred)).to(device).float())
            # #     for i in range(len(pred)):
            # #         dif[i] = (torch.max(pred[i, [T+5 for T in range(80)]], 0) - torch.max(pred[i, other], 0))*pred[i,4]      #+0.2
            # #     if len(pred)>=4:
            # #         loss_cls=torch.sum(dif.sort(descending=True)[[0,1,2,3]])
            # #     else:
            # #         loss_cls = torch.sum(dif.sort(descending=True)[[z for z in range(len(pred))]])

            # # #法二
            # # if len(pred_two) == 0:
            # #     loss_cls_two = torch.as_tensor(0)
            # # else:
            # #     dif_two = Variable(torch.ones(len(pred_two)).to(device).float())
            # #     for i in range(len(pred_two)):
            # #         dif_two[i] = (torch.max(pred_two[i, [T+5 for T in range(80)]], 0).values - torch.max(pred_two[i, other], 0).values)*pred_two[i,4]      #+0.2
            # #     if len(pred_two)>=4:
            # #         loss_cls_two=torch.sum(dif_two.sort(descending=True).values[[0,1,2,3]])
            # #     else:
            # #         loss_cls_two = torch.sum(dif_two.sort(descending=True).values[[z for z in range(len(pred_two))]])

            # # loss+=loss_cls+loss_cls_two
            # loss+=loss_cls

            # Process predictions #
            #法一
            for i, det in enumerate(pred1):  # 每张图片  len(pred)=1
                if webcam:  # batch_size >= 1
                    print("hh")
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                else:
                    p, im0, frame = path, out.copy(), getattr(dataset, 'frame', 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # 结果

                            # 含有三种目标类别
                            if names[c] == 'truck' or names[c] == 'bus' or names[c] == 'car':  # names[2]='car',names[5]='bus',names[7]='truck'
                                flag0 = 0
                
                # 攻击成功
                if flag0 == 1:
                    # print('success scene = ', scene_index)
                    # print('success frame = ', frame_index)
                    flag += 1


            # #法二
            # for i_two, det_two in enumerate(pred1_two):  # 每张图片  len(pred)=1
            #     im0_two=out.copy()
            #     if len(det_two):
            #         # Rescale boxes from img_size to im0 size
            #         det_two[:, :4] = scale_coords(im_two.shape[2:], det_two[:, :4], im0_two.shape).round()

            #         # Write results
            #         for *xyxy_two, conf_two, cls_two in reversed(det_two):
            #             c_two = int(cls_two)  # integer class
            #             # 含有三种目标类别
            #             if names[c_two] == 'truck' or names[c_two] == 'bus' or names[c_two] == 'car':  # names[2]='car',names[5]='bus',names[7]='truck'
            #                 flag0_two = 0

            #     # 攻击成功
            #     if flag0_two == 1:
            #         flag_two += 1
                    
            # print("cuda:", str(torch.cuda.memory_allocated() / 1e9) + "G")  # 内存使用情况
            # if count % batch_size == 0:
            #     # 通过loss反向传递并优化modifier
            #     torch.autograd.set_detect_anomaly(True)   #自动求导的异常侦测
            #     loss.requires_grad_(True)
            #     loss.backward(retain_graph=True)
            #     optimizer.step()

            #     loss_iter=loss.cpu().detach().numpy()
            #     loss_plot[(iteration-1)%batch_num].append(loss_iter)
            #     epoch_loss+=loss_iter
            #     iteration_plot.append(iteration)
            #     #print("Batch_loss: %.3f" % loss_iter)

        end_time = time.time()
        print('Epoch_time: ', end_time - start_time)
        print('Success_attack: ', flag / 1155.0)
        # print('Success_target_attack: ', target_flag / 1155.0)
        print("Total_loss: %.3f" % epoch_loss)
        if (epoch_num%20)==0:
            cv2.imwrite("./data/result/patch_"+str(epoch_num)+".png", np.uint8(torch.clamp(modifier,0,255).cpu().detach()))
            np.save("./data/result/patch_" + str(epoch_num) + ".npy", modifier.cpu().detach().numpy())

    #绘制loss曲线#
    #epoch-loss
    epoch_loss_plot.append(epoch_loss)
    epoch_plot.append(epoch_num)
    fig2 = plt.figure()
    fig3 = fig2.add_subplot(1, 1, 1)
    fig3.plot(epoch_plot, epoch_loss_plot, color='r', linewidth=1)
    fig3.set(xlabel='epoch', ylabel='loss', title='Attack-loss: learn=%.3f, thresh=%.2f' % (learning_rate, thresh))
    plt.legend(['loss'], fontsize=10)
    plt.savefig("./data/loss/Loss_total.png")
    plt.close()

    #batch-loss
    fig0=plt.figure()
    fig1=fig0.add_subplot(1,1,1)
    for k in range(batch_num):
        fig1.plot(epoch_plot,loss_plot[k],linewidth=0.8)
    fig1.set(xlabel='epoch', ylabel='loss', title='Attack-loss: learn=%.3f, thresh=%.2f'%(learning_rate,thresh))
    plt.savefig("./data/loss/Loss_batch.png")
    plt.close()

    success_plot.append(float(flag / count))
    success_plot_two.append(float(flag_two / count))
    print("Success-padding={}/{},   Ratio={:.4f} ".format(flag, count, flag / count))
    print("Success-resize={}/{},   Ratio={:.4f} ".format(flag_two, count, flag_two / count))
    print()

    #成功率
    fig4 = plt.figure()
    fig5 = fig4.add_subplot(1, 1, 1)
    fig5.plot(epoch_plot, success_plot, color='fuchsia', linewidth=1)
    fig5.plot(epoch_plot, success_plot_two, color='deepskyblue', linewidth=1)
    fig5.legend(["padding","resize"],fontsize=10)
    fig5.set(xlabel='epoch', ylabel='rate', title='Rate of Success')
    plt.savefig("./data/loss/Success-Rate.png")
    plt.close()


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov3.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')     #data/images，video
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')   #0.25
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'data/result', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
    print("finished")
