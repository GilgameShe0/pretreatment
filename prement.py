import os, shutil
import cv2
import numpy as np

def get_GaryImage(img_path):
    # 获取图片
    original_img = cv2.imread(img_path)
    
    # 读取图片的高、宽
    height, width = original_img.shape[:2]
    # 切掉图片左边残留的人体影像，强调主体部分
    img = original_img[0:height, int(0.1*width):width]
    # 将图片缩小50%便于观察
    # img = cv2.resize(img, (int(0.5*width), int(0.5*height)), interpolation=cv.INTER_AREA)

    # 转化为灰度图
    gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return original_img,img, gary

def Gaussian_Blur(gray):

    # 使用高斯滤波去噪
    blurred = cv2.GaussianBlur(gray, (9,9), 0)

    return blurred

def Sobel_Gradient(blurred):

    # 边缘检测，使用索贝尔算子获取x，y方向上的梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    # x方向梯度减去y方向梯度，获得高水平梯度和低垂直梯度的图像区域
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    return gradient

def Blur_Thresh(gradient):

    # 再次通过高斯去噪
    blurred = cv2.GaussianBlur(gradient, (9,9), 0)

    # 二值化,选择阈值为5
    _, thresh = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)

    return thresh

def Image_Morphology(thresh):

    # 填充空白区域
    # 建立椭圆核函数
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    #腐蚀与膨胀
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    
    return closed

def FindCnts(closed):
    
    # 找出物体边缘
    cnts, _ = cv2.findContours(closed.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))

    return box

def Draw_Cut(original_img, box):

    draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0,0,255))
    
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    height = y2 -y1
    width = x2 - x1
    crop_img = original_img[y1:y1+height, x1:x1+width]

    return draw_img, crop_img

def JudgePre(img_path):
    # 判断图片是否经过预处理
    img = cv2.imread(img_path,0)

    # 强调左边黑色背景，按比例裁出左半边区域
    height, width = img.shape[:2]
    img = img[0:height,0:int(0.5*width)]
    
    # 对灰度图作垂直镜面翻转
    flipped_img = cv2.flip(img,1)
    
    img1 = cv2.resize(img,(8,8))
    img2 = cv2.resize(flipped_img,(8,8))
    
    # 将两者的缩略图灰度值相减，小于0的为0
    sub = cv2.subtract(img1,img2)
    
    # 计算第一列灰度值之和，小于2的可以认为是没有经过预处理
    summary = 0
    for i in range(8):
        summary += sub[i][0]

    return summary
    
def pretreatment(path, new_path):

    for filename in os.listdir(path):
        # 原图文件路径
        img_path = path + filename
        if os.path.splitext(filename)[1] == '.png':
            # 选择文件格式为png
            summary = JudgePre(img_path)
            original_img, img, gray = get_GaryImage(img_path)
            if summary > 2:
                # 灰度值判断大于2（数值可根据实际图片再调整），则为已经过预处理，直接复制到新路径下
                cv2.imwrite(new_path + filename, original_img)
                print(img_path, summary)
            else:
                # 小于2则进行图片预处理
                blurred = Gaussian_Blur(gray)
                gradient = Sobel_Gradient(blurred)
                thresh = Blur_Thresh(gradient)
                closed = Image_Morphology(thresh)
                box = FindCnts(closed)
                draw_img, crop_img = Draw_Cut(img,box)
                cv2.imwrite(new_path + filename, crop_img)
                print(new_path + filename，‘ok’)
        else:
            # 信息文件复制到新路径
            shutil.copyfile(img_path,new_path + filename)

# 文件路径，原文件路径和新文件路径
path = ''
new_path = ''
    
pretreatment(path,new_path)