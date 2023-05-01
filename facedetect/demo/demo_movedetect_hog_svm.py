'''
参考资料：https://blog.csdn.net/weixin_45971950/article/details/122331273
'''
import cv2

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
        found, w = hog.detectMultiScale(frame)
        #print(found, w)
        foundList = []
        for ri, r in enumerate(found):
            flag = 0
            for qi, q in enumerate(found):
                if ri != qi and is_inside(r, q):
                    flag = 1
            if (flag == 0):
                foundList.append(r)
        
        for person in foundList:
            draw_person(frame, person)
        cv2.imshow("face", frame)
        if ord('q') == cv2.waitKey(1):
            break

# 释放内存
cv2.destroyAllWindows()
# 释放摄像头
cap.release()
