import os
import cv2 as cv
import numpy as np


# 转到对应操作中处理图像
def pre_process(data_path, num, ext):

    file_name = os.path.split(data_path)[1].split('.')[0]
    img = cv.imdecode(np.fromfile(data_path, dtype=np.uint8), -1)

    print("num process=", num)

    if num == 0:  # 椒盐噪声
        img = AddSaltPepperNoise(img, 0.5)

    elif num == 1:  # 均值平滑
        img = cv.blur(img, (3, 3))
    elif num == 2:  # 中值平滑
        img = cv.medianBlur(img, 3)
    elif num == 3:  # 高斯平滑
        img = cv.GaussianBlur(img,(11,11),0)

    elif num == 4: # 图像锐化-拉普拉斯算子
        img=cv.Laplacian(img,cv.CV_64F)
    elif num==5: #图像锐化-Sobel算子水平方向
        img=cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    elif num==6: #图像锐化-Sobel算子垂直方向
        img=cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
    elif num==7: #将图像用双线性插值法扩大图像
        img = cv.resize(img, (0, 0), fx=2, fy=2, interpolation=cv.INTER_NEAREST)
    elif num==8: #左移30个像素，下移50个像素
        M=np.float32([[1,0,30],[0,1,60]])
        height, width, channel = img.shape
        img = cv.warpAffine(img, M, (width, height))
    elif num==9: #旋转45度，缩放因子为1
        height, width, channel = img.shape
        M = cv.getRotationMatrix2D(((width/2),(height/2)),45,1)
        img = cv.warpAffine(img, M, (width, height))

    elif num==10: #转灰度图
        img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    elif num==11:#转灰度后二值化-全局阈值法
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

    elif num==12: #直方图均衡化
        img= cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        src = cv.resize(img, (256,256))
        img=cv.equalizeHist(src)
    elif num==13: #灰度直方图
        img=cv.calcHist([img],[0],None,[256],[0,255])
    elif num==14: #仿射变换
        src = cv.resize(img, (256, 256))
        rows, cols = src.shape[: 2]
        post1=np.float32([[50,50],[200,50],[50,200]])
        post2=np.float32([[10,100],[200,50],[100,250]])
        M=cv.getAffineTransform(post1,post2)
        img=cv.warpAffine(src,M,(rows,cols))
    elif num==15: #透视变换
        src = cv.resize(img, (256, 256))
        rows, cols = src.shape[: 2]
        post1 = np.float32([[55, 65], [288, 49], [28, 237], [239,240]])
        post2 = np.float32([[0, 0], [200, 0], [0, 200], [200, 200]])
        M = cv.getPerspectiveTransform(post1, post2)
        img=cv.warpPerspective(src,M,(200,200))
    elif num==16: #图像翻转
        img=255-img
    elif num==17: #rgb转hsv
        img=cv.cvtColor(img,cv.COLOR_RGB2HSV)
    elif num==18: #hsv获取h
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        img = hsv[:,:,0]
    elif num==19: #hsv获取s
        hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        img = hsv[:,:,1]
    elif num==20: #hsv获取v
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img = hsv[:,:,2]
    elif num==21: #rgb获取b
        img=img[:,:,0]
    elif num==22: #rgb获取g
        img=img[:,:,1]
    elif num==23: #rgb获取r
        img=img[:,:,2]
    elif num==24: #水平翻转
        img=cv.flip(img,1,dst=None)
    elif num==25: #垂直翻转
        img=cv.flip(img,0,dst=None)
    elif num==26: #对角镜像
        img=cv.flip(img,-1,dst=None)
    elif num==27: #图像开运算
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.morphologyEx(img,cv.MORPH_OPEN,kernel)
    elif num==28: #图像闭运算
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.morphologyEx(img,cv.MORPH_CLOSE,kernel)
    elif num==29: #腐蚀
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.erode(img,kernel)
    elif num==30: #膨胀
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.dilate(img,kernel)
    elif num==31: #顶帽运算
        kernel=cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img=cv.morphologyEx(img,cv.MORPH_TOPHAT,kernel)
    elif num == 32:  # 底帽运算
        kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5))
        img = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)
    elif num == 33:  # houghLinesP实现线条检测
        img = cv.GaussianBlur(img, (3, 3), 0)
        edges = cv.Canny(img, 50, 150, apertureSize=3)
        minLineLength = 200
        maxLineGap = 15
        linesP = cv.HoughLinesP(edges,1,np.pi/180,80,minLineLength,maxLineGap)
        result_P = img.copy()
        for i_P in linesP:
            for x1, y1, x2, y2 in i_P:
                cv.line(result_P, (x1, y1), (x2, y2), (0, 255, 0), 3)
        img = result_P
    elif num == 34:  # canny边缘检测
        blur = cv.GaussianBlur(img, (3, 3), 0)
        image = cv.cvtColor(blur, cv.COLOR_BGR2GRAY)
        gradx = cv.Sobel(image, cv.CV_16SC1, 1, 0)
        grady = cv.Sobel(image, cv.CV_16SC1, 0, 1)
        img = cv.Canny(gradx, grady, 50, 150)
    elif num == 35:  # 图像增强
        CRH = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        CRH = CRH.astype('float')
        row, column = CRH.shape
        gradient = np.zeros((row,column))
        for x in range(row - 1):
            for y in range(column - 1):
                gx = abs(CRH[x + 1, y] - CRH[x, y])
                gy = abs(CRH[x, y + 1] - CRH[x, y])
                gradient[x, y] = gx + gy
        sharp = CRH+gradient
        sharp = np.where(sharp > 255, 255, sharp)
        sharp = np.where(sharp < 0, 0, sharp)
        gradient = gradient.astype('uint8')
        img = sharp.astype('uint8')

    elif num == 36:  # Roberts算子提取图像边缘
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
        y = cv.filter2D(grayImage, cv.CV_16S, kernely)
        absX = cv.convertScaleAbs(x)
        absY = cv.convertScaleAbs(y)
        img = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
    elif num == 37:  # Prewitt 算子提取图像边缘
        grayImage = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        x=cv.Sobel(grayImage, cv.CV_16S, 1, 0)
        y=cv.Sobel(grayImage, cv.CV_16S, 0, 1)
        absX=cv.convertScaleAbs(x)
        absY=cv.convertScaleAbs(y)
        img=cv.addWeighted(absX,0.5,absY,0.5,0)
    elif num == 38:  # Laplacian算子提取图像边缘
        grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        grayImage = cv.GaussianBlur(grayImage, (5, 5), 0, 0)
        dst = cv.Laplacian(grayImage, cv.CV_16S,ksize=3)
        img == cv.convertScaleAbs(dst)
    elif num == 39:  # LoG边缘提取
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image = cv.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv.BORDER_REPLICATE)
        image = cv.GaussianBlur(image, (3, 3), 0, 0)
        m1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
        rows=image.shape[0]
        cols=image.shape[1]
        image1=np.zeros(image.shape)
        for k in range(0, 2):
            for i in range(2, rows-2):
                for j in range(2, cols-2):
                    image1[i, j] = np.sum((m1*image[i - 2:i + 3, j - 2:j + 3, k]))
        img = cv.convertScaleAbs(image1)

    cv.imencode(".png", img)[1].tofile(r'./tmp/draw/{}.{}'.format(file_name, ext))

    return data_path, file_name


def AddSaltPepperNoise(src, rate):
    srcCopy = src.copy()
    height, width = srcCopy.shape[0:2]
    noiseCount = int(rate*height*width/2)
    # add salt noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    srcCopy[Y, X] = 255
    # add black peper noise
    X = np.random.randint(width, size=(noiseCount,))
    Y = np.random.randint(height, size=(noiseCount,))
    srcCopy[Y, X] = 0
    return srcCopy




