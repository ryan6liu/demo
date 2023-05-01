import numpy as np
import cv2

# 输入测试视频
cap = cv2.VideoCapture('20220827-093000-100000.mp4')
# 形态学操作需要使用
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# 创建混合高斯模型用于背景建模
fgbg = cv2.createBackgroundSubtractorMOG2()
# 视屏读取速度，1为1倍速，8为8倍速
speed = 8
i=0

while (True):
    if i % (int(24)*speed) != 0:
        i=i+1
        continue
    i=i+1

    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    # 形态学开运算去噪点
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    # 寻找视频中的轮廓
    #im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours , hierarchy = cv2.findContours(fgmask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # 计算各轮廓的周长
        perimeter = cv2.arcLength(c, True)
        # 周长是多少才能检测出人，有待验证
        if perimeter > 8000:
            # 找到一个直矩形（不会旋转）
            x, y, w, h = cv2.boundingRect(c)
            # 画出这个矩形
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    cv2.imshow('fgmask', fgmask)
    if ord('q') == cv2.waitKey(1):
        break

cap.release()
cv2.destroyAllWindows()
