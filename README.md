# pretreatment

## 图片预处理

### 运行环境

- os: 18.04.1-Ubuntu
- python: Python 3.7.3
- opencv: 4.1.0

### 目的

使用python opencv提取图中研究感兴趣的部分，去除不感兴趣的背景部分。

### 主要步骤

##### opencv处理图像

1. 读取原图，转为灰度图，并使用高斯滤波去噪

    ```python
        img = cv2.imread(img_path)
        gary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9,9), 0)
    ```
2. 获得高水平梯度和低垂直梯度的图像区域
    
    ```python
        # 边缘检测，使用索贝尔算子获取x，y方向上的梯度
        gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
        gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

        # x方向梯度减去y方向梯度，获得高水平梯度和低垂直梯度的图像区域
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)
    ```
3. 再次使用高斯去除噪声，将图片二值化，选择阈值为5

    ```python
        blurred = cv2.GaussianBlur(gradient, (9,9), 0)
        _, thresh = cv2.threshold(blurred, 5, 255, cv2.THRESH_BINARY)
    ```
4. 填充空白区域，腐蚀与膨胀

    ```python
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
    ```
5. 绘制物体边框

    ```python
        cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
        draw_img = cv2.drawContours(original_img.copy(), [box], -1, (0,0,255))
    ```
6. 裁剪
    cropImg = img_gray[y1_:y2_, x1_:x2_]

##### 判断图像是否经过预处理
    
    未强调左边背景部分，裁剪出左半边图并转为灰度图。将图片做镜面翻转，并使用8*8的缩略图，将原图与镜面图灰度值相减（使用镜面图是为了减少误差），观察相减后左边第一列灰度值。
    

    ```python
        
        import cv2
        img = cv2.imread("pretreatment/09002298.png",0)

        height, width = img.shape[:2]
        img = img[0:height,0:int(0.5*width)]
        
        flipped_img = cv2.flip(img,1)
        
        img1 = cv2.resize(img,(8,8))
        img2 = cv2.resize(flipped_img,(8,8))
        
        # 将两者的缩略图灰度值相减，小于0的为0
        sub = cv2.subtract(img1,img2)
        
        # 计算第一列灰度值之和，小于某个值的可以认为是没有经过预处理
        summary = 0
        for i in range(8):
            summary += sub[i][0]
        
        print('summary:',summary)
        print('未预处理图片:')
        print('原图灰度值:')
        print(img1)
        print('垂直镜像图灰度值:')
        print(img2)
        print('相减后灰度值：')
        print(sub)


    ```
    ```bash
        summary:0
        未预处理图片:
        原图灰度值:
        [[  0   0   0 200 165 153 146   0]
        [  0   0 182 203 173  66  65 134]
        [  0   0 205 189  75   0  21 180]
        [  0   0 210 186  46  16  21 141]
        [  0   0 214 223 105  66  50 157]
        [  0   0 199 216 201  78 107 201]
        [  0   0 215 212 159  61 146 195]
        [  0   0   0 150 162 188 185   0]]
        垂直镜像图灰度值:
        [[  0 146 153 165 200   0   0   0]
        [134  65  66 173 203 182   0   0]
        [180  21   0  75 189 205   0   0]
        [141  21  16  46 186 210   0   0]
        [157  50  66 105 223 214   0   0]
        [201 107  78 201 216 199   0   0]
        [195 146  61 159 212 215   0   0]
        [  0 185 188 162 150   0   0   0]]
        相减后灰度值：
        [[  0   0   0  35   0 153 146   0]
        [  0   0 116  30   0   0  65 134]
        [  0   0 205 114   0   0  21 180]
        [  0   0 194 140   0   0  21 141]
        [  0   0 148 118   0   0  50 157]
        [  0   0 121  15   0   0 107 201]
        [  0   0 154  53   0   0 146 195]
        [  0   0   0   0  12 188 185   0]]
    ```

    


##### 批量处理多张图片

    ```python
        import os
        import cv2
        path = ''
        for filename in os.listdir(path):    
            if os.path.splitext(filename)[1] == '.png':
                img_path = path + filename
                cv2.imread('img_path')
            else:
                # 复制文件
    ```


