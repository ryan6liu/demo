'''
参考资料：
https://zhuanlan.zhihu.com/p/161530919
https://www.bilibili.com/medialist/play/watchlater/BV1Lq4y1Z7dm
'''
import os
import shutil
from sys import flags
import time
from datetime import datetime
import hashlib
from tkinter import Frame
from types import GeneratorType
import cv2 as cv

def face_detect(img):
    face_detect = cv.CascadeClassifier('opencv-4.6.0/data/haarcascades/haarcascade_frontalface_default.xml')
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # 人脸检测
    # 参数1.表示的是要检测的灰度图像
    # 参数2.表示图像缩放系数
    # 参数3.目标大小，人脸最小不得小于5peix
    face = face_detect.detectMultiScale(gray, 1.3, 5)
    #face = face_detect.detectMultiScale(gray)
    # 绘制方框
    # 参数1：需要绘制的数据
    # 参数2，绘制的起始坐标
    # 参数3，绘制的高度和宽度
    # 参数4，颜色
    for x,y,w,h in face:
        cv.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
    cv.imshow('result',img)
    return face

# MD5值
def getMD5(path):
    f=open(path,'rb')
    d5 = hashlib.md5()      #生成一个hash的对象
    with open(path,'rb') as f:
        while True:
            content = f.read(40960)
            if not content:
                break
            d5.update(content)   # 每次读取一部分，然后添加到hash对象里
    # print('MD5 : %s' % d5.hexdigest())
    return d5.hexdigest()        # 打印16进制的hash值
 
# 装饰器，计时用的
def timer(func):   # 高阶函数：以函数作为参数
    def deco(*args,**kwargs):    # 嵌套函数，在函数内部以 def 声明一个函数,接受 被装饰函数的所有参数
        time1 = time.time()
        func(*args,**kwargs)
        time2 = time.time()
        print('Elapsed %ss\n' % round(time2-time1,2))
    return deco    # 注意，返回的函数没有加括号！所以返回的是一个内存地址，而不是函数的返回值
 
@timer
def compare(baseFolder,targetFolder,content_compare='y'):
    '''
    :param baseFolder:   基础文件夹，将基础文件夹中的文件按照相应的目录结构同步到目标文件夹中。
    :param targetFolder: 目标文件夹
    :content_compare: 是否比对两个文件的内容，默认比对，防止文件内容有更改。参数值如果不是'y',则不比对内容，只判断目标文件夹是否有同名文件，有就跳过，没有就复制过去。
    '''
    n = 0
    for path,subpath,files in os.walk(baseFolder):
        for fileName in files:
            # 如果不是mp4类型则跳过
            if not ".mp4" in fileName:
                continue
            # 获取文件的绝对路径
            fullPath = os.path.join(path,fileName)
            #获取文件的创建时间 get create time
            ctime = os.path.getctime(fullPath)
            print("创建时间是："+time.ctime(ctime))
            mtime = os.path.getmtime(fullPath)
            print("修改时间是："+time.ctime(mtime))
            # 获取文件创建时间的小时很分钟
            hourmin = datetime.fromtimestamp(ctime+43200).strftime('%H:%M')
            # 如果是六点半之前或晚上八点以后的视频跳过
            if print(hourmin >= '20:00' or hourmin <'06:30'):
                break
            # 文件的后半截路径,即文件的相对路径
            relativePath = fullPath.replace(baseFolder,'')
            # 去掉路径前面的\
            if relativePath[:1] == '\\':
                relativePath = relativePath[1:]
 
            # 替换目录末尾的\
            if targetFolder[:1] == '\\':
                targetFolder = targetFolder[1:]
 
            # 判断目标文件夹是否有相应的文件
            targetFile = os.path.join(targetFolder,relativePath)
             
            # 目标文件夹路径
            targetFolder = os.path.dirname(targetFile)
             
            #　如果目标文件夹不存在此文件
            if not os.path.exists(targetFile):
              
                # 读取视频文件
                cap = cv.VideoCapture(fullPath)
                #测试视频
                #cap = cv.VideoCapture('20220827-073000-080000.mp4')   
                # 视频的帧率FPS 
                fps = cap.get(cv.CAP_PROP_FPS)            
                # 视频的总帧数
                total_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)	
                for i in range(int(total_frame)):
                    ret = cap.grab()
                    if not ret:
                        break
                    if i % int(fps) == 0:
                        ret, frame = cap.retrieve()
                        if ret:
                            face = face_detect(frame)
                            # 如果找到人脸图片
                            if len(face) > 1:
                                # 目标文件夹路径
                                os.makedirs(targetFolder,exist_ok=True)
                                # 移动文件
                                shutil.copy(fullPath,targetFolder)
                                print(f'\n     {fullPath}   -------copy------->   {targetFile}')
                                break
                        else:
                            print("Error retrieving frame from movie!")
                            break

                '''
                #循环读取视频的帧
                while True:
                    flag, frame = cap.read()
                    if not flag:
                        break
                    face = face_detect(frame)
                    cv.imshow('read_img',frame)
                    # 如果找到人脸图片
                    if len(face) > 1:
                        cv.imshow('read_img',frame)
                        # 移动文件
                        shutil.copy(fullPath,fileFolder)
                        print(f'\n     {fullPath}   -------copy------->   {targetFile}')
                        break    
                '''


            else: # 如果目标文件夹已经存在某文件
                # 如果要对比两个文件的内容
                if content_compare == 'y':
                    # 先对比两个文件的修改时间（谁的时间越大，代表谁的内容越新）
                    baseTime = os.stat(fullPath).st_mtime
                    targetTime = os.stat(targetFile).st_mtime
                    if baseTime-targetTime > 0:
                        # 比了时间，再比一下MD5。如果MD5也不同，就将这个时间最新的文件复制过去
                        baseMD5 = getMD5(fullPath)
                        targetMD5 = getMD5(targetFile)
                        if baseMD5 != targetMD5:
                            os.unlink(targetFile)
                            shutil.copy(fullPath,fileFolder)
                            print(f'\n     {fullPath}   -------------->   {targetFile}')
                        else:
                            # MD5相同，而目标文件夹中的时间又小，就将修改时间改大，防止下次运行此脚本时再对比一遍MD5，浪费时间
                            # 修改文件的访问和修改时间，改成当前系统时间
                            os.utime(targetFile)
            n+=1
            print("\r%s:  Has scanned %s files. "%(baseFolder,n),end='' )
 
if __name__ == '__main__':
    print('----------Copying the files of BaseFolder to TargetFolder.----------\n')
    #base = input('Input base folder:')
    #target = input('Input target folder:')
    base = '/Volumes/Yellow/Test/'
    target = '/Volumes/Yellow/qvrpro'
    # 当有相同文件时，是否对比文件内容，把最新的同步过去，适用于经常变动的文件，如脚本，文档
    #content_compare = input('Compare content? y/n: ').lower()
    content_compare = 'y'
    if base == '' or target == '':
        print('base folder or target folder is empty')
    else:
        compare(base,target,content_compare)



 

