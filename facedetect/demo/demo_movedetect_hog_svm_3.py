'''
参考资料https://developer.aliyun.com/article/851131
'''
# import the necessary packages
from __future__ import print_function
import numpy as np
import argparse
import cv2
import os

def nms(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")
image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    # 如果高和宽为None则直接返回
    if width is None and height is None:
        return image
    # 检查宽是否是None
    if width is None:
        # 计算高度的比例并并按照比例计算宽度
        r = height / float(h)
        dim = (int(w * r), height)
    # 高为None
    else:
        # 计算宽度比例，并计算高度
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


# 读取摄像头
cap = cv2.VideoCapture('20220827-093000-100000.mp4')
# 视频的帧率FPS
fps = cap.get(cv2.CAP_PROP_FPS)
# 视频的总帧数
total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# 视屏读取速度，1为1倍速，8为8倍速
speed = 8

# 使用opencv的hog特征进行行人检测
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for i in range(int(total_frame)):
    ret, frame = cap.read()
    if not ret:
        break
    if i % (int(fps)*speed) == 0:
        # 加载图像并调整其大小以
        # （1）减少检测时间
        # （2）提高检测精度
        #frame = resize(frame, width=min(800, frame.shape[1]))
        orig = frame.copy()
        #print(frame)
        # detect people in the image
        '''
        4.winStride(可选)
        HoG检测窗口移动时的步长(水平及竖直)。
        winStride和scale都是比较重要的参数，需要合理的设置。一个合适参数能够大大提升检测精确度，同时也不会使检测时间太长。
        5.padding(可选)
        在原图外围添加像素，作者在原文中提到，适当的pad可以提高检测的准确率（可能pad后能检测到边角的目标？）
        常见的pad size 有(8, 8), (16, 16), (24, 24), (32, 32).
        6.scale(可选)
        scale参数可以具体控制金字塔的层数，参数越小，层数越多，检测时间也长。 一下分别是1.01  1.5 1.03 时检测到的目标。 通常scale在1.01-1.5这个区间
        '''
        (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),padding=(8, 8), scale=1.05)
        # draw the original bounding boxes
        #print(rects)
        for (x, y, w, h) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 使用相当大的重叠阈值对边界框应用非极大值抑制，以尝试保持仍然是人的重叠框
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = nms(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        # show the output images
        cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", frame)
        if ord('q') == cv2.waitKey(1):
            break

# 释放内存
cv2.destroyAllWindows()
# 释放摄像头
cap.release()
