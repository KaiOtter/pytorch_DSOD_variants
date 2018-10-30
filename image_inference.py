from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from models.other.ssd_mobile import build_mobile_ssd
from layers.functions.deteciton import Detect
from PIL import Image, ImageDraw, ImageFont

import os
import numpy as np
import cv2
import colorsys
import random


def inference_net(net, cuda, images, num_cls=21, top_k=100, im_size=300,
                  conf_thresh=0.05, nms_thresh=0.45, extend=None):
    VOC_CLASSES = (  # always index 0
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor')
    # get colors
    hsv_tuples = [(x / len(VOC_CLASSES), 1., 1.)
                  for x in range(len(VOC_CLASSES))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.

    # set detection
    # __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh)
    detector = Detect(num_cls, 0, top_k, conf_thresh, nms_thresh)

    for img_path in images:
        # to rgb and normal
        img = cv2.imread(img_path)
        ori_img = img.copy()
        h, w, channels = img.shape
        # to rgb
        img = cv2.resize(img, (im_size, im_size)).astype(np.float32)
        img -= dataset_mean
        img = img.astype(np.float32)
        img = img[:, :, (2, 1, 0)]
        img = torch.from_numpy(img).permute(2, 0, 1)

        ori_img = Image.fromarray(ori_img)
        font = ImageFont.truetype(
            font='font/FiraMono-Medium.otf',
            size=np.floor(3e-2 * ori_img.size[1] + 0.5).astype('int32'))
        thickness = (ori_img.size[0] + ori_img.size[1]) // 300

        x = Variable(img.unsqueeze(0))
        if cuda:
            x = x.cuda()
        output = net(x)
        detections = detector(output[0], output[1], output[2]).data

        # all detections are collected into:
        #    (score, x1, y1, x2, y2)
        # skip j = 0, because it's the background class
        for j in range(1, detections.size(1)):
            dets = detections[0, j, :]
            mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
            dets = torch.masked_select(dets, mask).view(-1, 5)
            if dets.size(0) == 0:
                # if dets.dim() == 0:
                continue
            boxes = dets[:, 1:]
            boxes[:, 0] *= w
            boxes[:, 2] *= w
            boxes[:, 1] *= h
            boxes[:, 3] *= h
            scores = dets[:, 0].data
            cls_name = VOC_CLASSES[j-1]
            label = []
            for s in scores:
                t = '{} {:.2f}'.format(cls_name, s)
                label.append(t)

            for idx, box in enumerate(boxes):
                draw = ImageDraw.Draw(ori_img)
                label_size = draw.textsize(label[idx], font)
                box = box.cpu().numpy()
                left, top, right, bottom = box
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(h, np.floor(bottom + 0.5).astype('int32'))
                right = min(w, np.floor(right + 0.5).astype('int32'))

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])
                    # My kingdom for a good redistributable image drawing library.
                for i in range(thickness):
                    draw.rectangle(
                        [left + i, top + i, right - i, bottom - i],
                        outline=colors[j-1])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[j-1])
                draw.text(text_origin, label[idx], fill=(0, 0, 0), font=font)
                del draw
            save = img_path[:-4] + extend + img_path[-4:]
            cv2.imwrite(save, np.array(ori_img))
            print("%s done" % img_path)
    print('Finsh Evaluating detections')


if __name__ == '__main__':
    cuda = True
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset_mean = (127, 127, 127)
    # the extend part of names of output imgs
    save_extend = "SSD"

    # load net
    # num_classes = len(labelmap) + 1                      # +1 for background
    labelmap = ['cow'] # names of classes
    clsmap = dict(zip(labelmap, range(len(labelmap))))
    num_classes = len(labelmap) + 1
    net = build_mobile_ssd(num_classes)            # initialize SSD
    net.load_state_dict(torch.load("xxx"), strict=True)
    net.eval()
    print('Finished loading model!')

    # load image data
    test_folder = "xxx"  #path to the directory
    images = []
    for im in os.listdir(test_folder):
        if len(im.split('.')[0]) == 6:
            images.append(os.path.join(test_folder, im))

    net = net.cuda()
    cudnn.benchmark = True

    # evaluation
    inference_net(net, cuda, images, num_classes, top_k=200, im_size=300,
                  conf_thresh=0.3, nms_thresh=0.45, extend=save_extend)
