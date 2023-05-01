'''
参考资料：
https://zhuanlan.zhihu.com/p/161530919
https://www.bilibili.com/medialist/play/watchlater/BV1Lq4y1Z7dm
'''
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import imutils
import cv2
import os
import shutil
from sys import flags
import time
from datetime import datetime
import hashlib
from tkinter import Frame
from types import GeneratorType
'''
https://www.bilibili.com/video/BV19k4y1d7aT?spm_id_from=333.999.0.0&vd_source=e8fd31f897b7522a3f026329abd94f25
https://github.com/chunhuizhang/bilibili_vlogs/blob/master/dip/hog_det/pedestrian_det.py
'''

gl_resize=400
gl_scale=1.01

def hog_clf(descriptor_type='default'):
    '''
    参数解释如下
    https://blog.csdn.net/qq_26898461/article/details/46786285
    '''
    if descriptor_type == 'daimler':
        #经测试，这组参数检测的误差非常大，不是人物的东西也检测出来了
        winSize = (48, 96)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        # 1.初始化 HOG 描述符
        hog = cv2.HOGDescriptor(
            winSize, blockSize, blockStride, cellSize, nbins)
        # 2.设置 SVM 为预训练好的行人检测器
        hog.setSVMDetector(cv2.HOGDescriptor_getDaimlerPeopleDetector())
        return hog
    else:
        '''
        原始
        winSize = (64, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        '''
        winSize = (64, 128)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        # 1.初始化 HOG 描述符
        hog = cv2.HOGDescriptor(
            winSize, blockSize, blockStride, cellSize, nbins)
        # 2.设置 SVM 为预训练好的行人检测器
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        return hog


def detect_image(hog, image):
    # image = cv2.imread(image_path)
    image = imutils.resize(image, width=min(gl_resize, image.shape[1]))
    orig = image.copy()

    # detect people in the image
    # 3.检测行人对应的矩形框以及权重值
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
                                            padding=(8, 8), scale=gl_scale, useMeanshiftGrouping=False)

    # draw the original bounding boxes
    # 4.遍历矩形框绘制图像
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # show some information on the number of bounding boxes
    # print("[INFO] {} original boxes, {} after suppression".format(
    #     len(rects), len(pick)))
    return image,len(pick)


def detect_images(hog, images_path):
    # loop over the image paths
    for image_path in paths.list_images(images_path):
        # load the image and resize it to (1) reduce detection time
        # and (2) improve detection accuracy
        orig = cv2.imread(image_path)
        image = detect_image(hog, orig)

        # show the output images
        cv2.imshow("Before NMS", orig)
        cv2.imshow("After NMS", image)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


def detect_video(hog, video_path):
    #统计检测个数
    detectedTotle = 0
    #读取视频
    cap = cv2.VideoCapture(video_path)
    # 视频的帧率FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    # 视频的总帧数
    total_frame = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    # 视屏读取速度，1为1倍速，8为8倍速
    speed = 8
    for i in range(int(total_frame)):
        ret = cap.grab()
        if not ret:
            break
        if i % (int(fps)*speed) == 0:
            ret, frame = cap.retrieve()
            if ret:
                detected, detectedCount = detect_image(hog, frame)
                cv2.imshow("capture", detected)
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break                

                # 如果找到人脸图片
                if detectedCount > 0:
                    #统计只是为了测试视频检测的准确率，正常情况下可可注视掉
                    #detectedTotle = detectedTotle + detectedCount
                    #正常情况下只要检测到就返回
                    #释放视频
                    cap.release()
                    return detectedCount
            else:
                print("Error retrieving frame from movie!")
                cap.release()
                return detectedCount
                break
    
    #打印只是为了测试视频检测的准确率，正常情况下可可注视掉
    #print(f'{video_path}在{speed}倍速,resize{gl_resize},scale{gl_scale}下共检测到行人移动{detectedTotle}次')
    #释放视频
    cap.release()
    return detectedCount


# MD5值
def getMD5(path):
    f = open(path, 'rb')
    d5 = hashlib.md5()  # 生成一个hash的对象
    with open(path, 'rb') as f:
        while True:
            content = f.read(40960)
            if not content:
                break
            d5.update(content)   # 每次读取一部分，然后添加到hash对象里
    # print('MD5 : %s' % d5.hexdigest())
    return d5.hexdigest()        # 打印16进制的hash值

# 装饰器，计时用的


def timer(func):   # 高阶函数：以函数作为参数
    def deco(*args, **kwargs):    # 嵌套函数，在函数内部以 def 声明一个函数,接受 被装饰函数的所有参数
        time1 = time.time()
        func(*args, **kwargs)
        time2 = time.time()
        print('Elapsed %ss\n' % round(time2-time1, 2))
    return deco    # 注意，返回的函数没有加括号！所以返回的是一个内存地址，而不是函数的返回值


@timer
def compare(baseFolder, targetFolder, content_compare='y'):
    '''
    :param baseFolder:   基础文件夹，将基础文件夹中的文件按照相应的目录结构同步到目标文件夹中。
    :param targetFolder: 目标文件夹
    :content_compare: 是否比对两个文件的内容，默认比对，防止文件内容有更改。参数值如果不是'y',则不比对内容，只判断目标文件夹是否有同名文件，有就跳过，没有就复制过去。
    '''
    # 统计移动文件的个数
    filesCount = 0
    hog = hog_clf()
    #hog = hog_clf('daimler')
    for path, subpath, files in os.walk(baseFolder):
        for fileName in files:
            # 如果不是mp4类型则跳过
            if not ".mp4" in fileName:
                continue
            # 获取文件的绝对路径
            fullSrcFile = os.path.join(path, fileName)
            # 获取文件的创建时间 get create time
            ctime = os.path.getctime(fullSrcFile)
            #print("创建时间是："+time.ctime(ctime))
            mtime = os.path.getmtime(fullSrcFile)
            #print("修改时间是："+time.ctime(mtime))
            # 获取文件创建时间的小时和分钟
            #hourmin = datetime.fromtimestamp(ctime+43200).strftime('%H:%M')
            hourmin = datetime.fromtimestamp(ctime).strftime(('%H:%M'))            
            # 如果是六点半之前或晚上八点以后的视频跳过
            if hourmin >= '19:30' or hourmin < '06:30':
                continue
            # 文件的后半截路径,即文件的相对路径
            relativePath = fullSrcFile.replace(baseFolder, '')
            # 去掉路径前面的\
            if relativePath[:1] == '\\':
                relativePath = relativePath[1:]

            # 替换目录末尾的\
            if targetFolder[:1] == '\\':
                targetFolder = targetFolder[1:]

            # 判断目标文件夹是否有相应的文件
            targetFile = os.path.join(targetFolder, relativePath)
            # 目标文件夹路径
            targetFileFolder = os.path.dirname(targetFile)
            #　如果目标文件夹不存在此文件
            if not os.path.exists(targetFile):
                # 测试视频
                # 在4倍速,resize400,scale1.01共检测到行人移动6次，经过实验得出这个的准确率比价高，主要针对只出现一次的行人的视频
                # 在4倍速,resize400,scale1.25共检测到行人移动3次
                # 在8倍速,resize400,scale1.01下共检测到行人移动5次，看样子是这个最优
                # 在8倍速,resize800,scale1.01下共检测到行人移动16次，这个最准但是耗时太久
                # 在8倍速,resize800,scale1.25下共检测到行人移动4次
                # detect_video(hog, '20220827-093000-100000.mp4')
                # 读取视频文件
                isDetected = detect_video(hog, fullSrcFile)
                if isDetected > 0:
                    # 目标文件夹路径
                    os.makedirs(targetFileFolder, exist_ok=True)
                    # 移动文件
                    shutil.move(fullSrcFile, targetFileFolder)
                    print(
                        f'{fullSrcFile}   -------moved------->   {targetFile}')
            else:  # 如果目标文件夹已经存在某文件
                # 如果要对比两个文件的内容
                if content_compare == 'y':
                    # 先对比两个文件的修改时间（谁的时间越大，代表谁的内容越新）
                    baseTime = os.stat(fullSrcFile).st_mtime
                    targetTime = os.stat(targetFile).st_mtime
                    if baseTime-targetTime > 0:
                        # 比了时间，再比一下MD5。如果MD5也不同，就将这个时间最新的文件复制过去
                        baseMD5 = getMD5(fullSrcFile)
                        targetMD5 = getMD5(targetFile)
                        if baseMD5 != targetMD5:
                            os.unlink(targetFile)
                            shutil.copy(fullSrcFile, targetFolder)
                            print(
                                f'\n     {fullSrcFile}   -------------->   {targetFile}')
                        else:
                            # MD5相同，而目标文件夹中的时间又小，就将修改时间改大，防止下次运行此脚本时再对比一遍MD5，浪费时间
                            # 修改文件的访问和修改时间，改成当前系统时间
                            os.utime(targetFile)
            filesCount += 1
    print("\r%s:  Has scanned %s files. " %
            (baseFolder, filesCount), end='')
        
    cv2.destroyAllWindows()



if __name__ == '__main__':
    print('----------Moving the files of BaseFolder to TargetFolder.----------\n')
    # base = input('Input base folder:')
    # target = input('Input target folder:')
    # 注意最后以/结尾
    base = '/Volumes/Yellow/Test/'
    target = '/Volumes/Yellow/qvrpro'
    # 当有相同文件时，是否对比文件内容，把最新的同步过去，适用于经常变动的文件，如脚本，文档
    # content_compare = input('Compare content? y/n: ').lower()
    content_compare = 'y'
    if base == '' or target == '':
        print('base folder or target folder is empty')
    else:
        compare(base, target, content_compare)
