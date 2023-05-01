# 导入cv模块
import cv2 as cv
# 检测函数


def face_detect_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')
    # 人脸检测
    # 参数1.表示的是要检测的灰度图像
    # 参数2.表示图像缩放系数
    # 参数3.目标大小，人脸最小不得小于5peix
    face = face_detect.detectMultiScale(gray, 1.3, 10)
    #face = face_detect.detectMultiScale(gray)
    # 绘制方框
    # 参数1：需要绘制的数据
    # 参数2，绘制的起始坐标
    # 参数3，绘制的高度和宽度
    # 参数4，颜色
    for x, y, w, h in face:
        cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)

    return face


# 读取摄像头
cap = cv.VideoCapture('20220827-093000-100000.mp4')
# 视频的帧率FPS
fps = cap.get(cv.CAP_PROP_FPS)
# 视频的总帧数
total_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
# 视屏读取速度，1为1倍速，8为8倍速
speed = 24

# 循环
while True:
    flag, frame = cap.read()
    # 最后一帧读完饭回false
    if not flag:
        break
    face_detect_demo(frame)
    cv.imshow('result', frame)

    # 倍速等待
    key = cv.waitKey(int(1000//fps))
    if ord('q') == key:
        break

# 释放内存
cv.destroyAllWindows()
# 释放摄像头
cap.release()
