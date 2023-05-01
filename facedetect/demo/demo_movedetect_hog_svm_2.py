'''
参考资料：https://zhuanlan.zhihu.com/p/495765262
'''
#from pygame import mixer
import time
#mixer.init()
#mixer.music.load('warn.mp3')
#mixer.music.play()
#time.sleep(5)
#mixer.music.stop()



import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import imutils

def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox+ow < ix+iw and oy+oh < iy+ih


def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 1)


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
        # 检测代码
        roi=frame[50:450,350:600]
        (rects,weights)=hog.detectMultiScale(roi,winStride=(4,4),padding=(8,8),scale=1.05)
        rects=np.array([[x,y,x+w,y+h] for (x,y,w,h) in rects])
        pick=non_max_suppression(rects,probs=None,overlapThresh=0.65)
        for (x,y,w,h) in pick:
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,0,255),2)
            cv2.rectangle(frame,(50,50),(600,450),(0,255,0),2)
            cv2.rectangle(frame,(350,50),(600,450),(255,0,0),2)
            print("检测到危险区行人数{}".format(len(pick)))
            cv2.imshow("HOG+SVM+NMS",frame)
            if ord('q') == cv2.waitKey(1):
                break

# 释放内存
cv2.destroyAllWindows()
# 释放摄像头
cap.release()
