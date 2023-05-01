# 导入cv模块
import cv2 as cv
# 检测函数


def face_detect_demo(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face_detect = cv.CascadeClassifier(
        'haarcascades/haarcascade_frontalface_default.xml')
    eye_detect = cv.CascadeClassifier(
        "haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    # 人脸检测
    # 参数1.表示的是要检测的灰度图像
    # 参数2.表示图像缩放系数
    # 参数3.目标大小，人脸最小不得小于5peix
    #face = face_detect.detectMultiScale(gray)
    face = face_detect.detectMultiScale(gray, 1.3, 10)
    #face = face_detect.detectMultiScale(gray, 1.1, 5, cv.CASCADE_SCALE_IMAGE, (50, 50), (100, 100))

    if len(face) > 0:
        for x, y, w, h in face:
            # 绘制方框
            # 参数1：需要绘制的数据
            # 参数2，绘制的起始坐标
            # 参数3，绘制的高度和宽度
            # 参数4，颜色
            cv.rectangle(img, (x, y), (x+w, y+h), color=(0, 0, 255), thickness=2)
            # cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2,8,0)

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_detect.detectMultiScale(roi_gray, 1.1, 1, cv.CASCADE_SCALE_IMAGE, (2, 2))
        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    return face


# 读取摄像头
cap = cv.VideoCapture('20220827-093000-100000.mp4')
# 视频的帧率FPS
fps = cap.get(cv.CAP_PROP_FPS)
# 视频的总帧数
total_frame = cap.get(cv.CAP_PROP_FRAME_COUNT)
# 视屏读取速度，1为1倍速，8为8倍速
speed = 8

for i in range(int(total_frame)):
    ret = cap.grab()
    if not ret:
        break
    if i % (int(fps)*speed) == 0:
        ret, frame = cap.retrieve()
        if ret:
            face_detect_demo(frame)
            cv.imshow('result', frame)
            if ord('q') == cv.waitKey(1):
                break
        else:
            print("Error retrieving frame from video!")
            break

# 释放内存
cv.destroyAllWindows()
# 释放摄像头
cap.release()
