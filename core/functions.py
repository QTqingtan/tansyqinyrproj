import math
import sys

import numpy as np
import cv2
import pytesseract
sys.path.append("../")
from config import *
pytesseract.pytesseract.tesseract_cmd = TESSERACT_OCR


# 二值化算法大全
class myThreshold():

    def getMinimumThreshold(self, imgSrc):
        # 谷底最小值的阈值
        Y = Iter = 0
        HistGramC = []
        HistGramCC = []

        # 获取直方数组
        hist_cv = self.__getHisGram(imgSrc)
        for Y in range(256):
            HistGramC.append(hist_cv[Y])
            HistGramCC.append(hist_cv[Y])

        # 通过三点求均值来平滑直方图
        while( self.__IsDimodal(HistGramCC) == False):
            HistGramCC[0] = (HistGramC[0] + HistGramC[0] + HistGramC[1]) / 3.0  # 第一点
            for Y in range(1, 255):
                HistGramCC[Y] = (HistGramC[Y - 1] + HistGramC[Y] + HistGramC[Y + 1]) / 3  # 中间的点

            HistGramCC[255] = (HistGramC[254] + HistGramC[255] + HistGramC[255]) / 3  # 最后一点
            HistGramC = HistGramCC
            Iter += 1
            if (Iter >= 1000):
                return -1

        # 阈值极为两峰之间的最小值
        Peakfound = False
        for Y in range(1, 255):
            if (HistGramCC[Y - 1] < HistGramCC[Y] and HistGramCC[Y + 1] < HistGramCC[Y]):
                Peakfound = True
            if (Peakfound == True and HistGramCC[Y - 1] >= HistGramCC[Y] and HistGramCC[Y + 1] >= HistGramCC[Y]):
                return Y - 1
        return -1

    def __IsDimodal(self, HistGram):
        # 对直方图的峰进行计数，只有峰数位2才为双峰
        Count = 0

        for Y in range(1, 255):
            if HistGram[Y - 1] < HistGram[Y] and HistGram[Y + 1] < HistGram[Y]:
                Count += 1
                if(Count > 2):
                    return False

        if Count == 2:
            return True
        else:
            return False

    def __getHisGram(self, imgSrc):
        hist_cv = cv2.calcHist([imgSrc], [0], None, [256], [0, 256])
        return hist_cv

    # 一维最大熵
    def get1DMaxEntropyThreshold(self, imgSrc):
        X = Y = Amount = 0
        HistGramD = {}
        MinValue = 0
        MaxValue = 255
        Threshold = 0

        HistGram = self.__getHisGram(imgSrc)

        for i in range(256):
            if HistGram[MinValue] == 0:
                MinValue += 1
            else:
                break

        while MaxValue > MinValue and HistGram[MinValue] == 0:
            MaxValue -= 1

        if (MaxValue == MinValue):
            return MaxValue     # 图像只有一个颜色
        if (MinValue + 1 == MaxValue):
            return MinValue     # 图像只有二个颜色

        for Y in range(MinValue, MaxValue + 1):
            Amount += HistGram[Y]  # 像素总数

        for Y in range(MinValue, MaxValue + 1):
            HistGramD[Y] = HistGram[Y] / Amount + 1e-17

        MaxEntropy = 0.0
        for Y in range(MinValue + 1, MaxValue):
            SumIntegral = 0
            for X in range(MinValue, Y + 1):
                SumIntegral += HistGramD[X]

            EntropyBack = 0
            for X in range(MinValue, Y + 1):
                EntropyBack += (- HistGramD[X] / SumIntegral * math.log(HistGramD[X] / SumIntegral))

            EntropyFore = 0
            for X in range(Y + 1, MaxValue + 1):
                SumI = 1 - SumIntegral
                if SumI < 0:
                    SumI = abs(SumI)
                elif SumI == 0:
                    continue

                EntropyFore += (- HistGramD[X] / (1 - SumIntegral) * math.log(HistGramD[X] / SumI))

            if MaxEntropy < (EntropyBack + EntropyFore):
                Threshold = Y
                MaxEntropy = EntropyBack + EntropyFore

        if Threshold > 5:
            return Threshold - 5  # 存在误差
        return Threshold

    # ISODATA (intermeans)  阈值算法
    def getIsoDataThreshold(self, imgSrc):

        HistGram = self.__getHisGram(imgSrc)
        g = 0
        for i in range(1, len(HistGram)):
            if HistGram[i] > 0:
                g = i + 1
                break

        while True:
            l = 0
            totl = 0
            for i in range(0, g):
                totl = totl + HistGram[i]
                l = l + (HistGram[i] * i)

            h = 0
            toth = 0
            for i in range(g+1, len(HistGram)):
                toth += HistGram[i]
                h += (HistGram[i] * i)

            if totl > 0 and toth > 0:
                l = l/totl
                h = h/toth
                if g == int((l + h / 2.0)):
                    break
            g += 1
            if g > len(HistGram) - 2:
                return 0

        return g

    def getIntermodesThreshold(self, imgSrc):
        HistGram = self.__getHisGram(imgSrc)
        return 126

    # 获取阈值
    def getAlgos(self):
        algos = {
            0: 'getMinimumThreshold',  # 谷底最小值
            1: 'get1DMaxEntropyThreshold', # 一维最大熵
            2: 'getIsoDataThreshold', # intermeans
            # 3: 'getKittlerMinError', # kittler 最小错误
            4: 'getIntermodesThreshold', # 双峰平均值的阈值
        }
        return algos


# 灰度图像二值化
def gray_to_binary(gray, method=1):
    j = method  # 选择阈值获取算法0,1,2,3,4,5
    thr = myThreshold()
    # 选择阈值获取算法0,1,2,3,4,5
    algos = thr.getAlgos()[j]
    threshold = getattr(thr, algos)(gray)
    # 输出：阈值、二值化数据
    ret, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return ret, binary


# 根据图片大小粗略计算腐蚀 或膨胀所需核的大小
def cal_element_size(img):
    sp = img.shape
    width = sp[1]  # width(colums) of image
    kenaly = math.ceil((width / 400.0) * 12)
    kenalx = math.ceil((kenaly / 5.0) * 4)
    a = (int(kenalx), int(kenaly))
    return a


# 查找身份证号码可能的区域列表
def find_id_regions(img):
    regions = []
    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)
        # 面积小的都筛选掉
        if(area < 1000):
            continue
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # 计算高和宽
        width = rect[1][0]
        height = rect[1][1]
        # 筛选那些太细的矩形 留下扁的
        if height > width:
            if height < width * 5:
                continue
        else:
            if width < height * 5:
                continue
        regions.append(rect)
    return regions


# 检查身份证号 有效性
def is_identi_number(num):
    if (len(num) != 18):
        return False
    Wi = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    Ti = ['1', '0', 'x', '9', '8', '7', '6', '5', '4', '3', '2']
    sum = 0
    for i in range(17):
        sum += int(num[i]) * Wi[i]
    if Ti[sum % 11] == num[-1].lower():
        return True
    else:
        return False


# 通过顶点矩阵 裁剪图片
def crop_img_by_box(img, box):
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    height = y2 - y1
    width = x2 - x1
    # 裁剪
    crop_img = img[y1:y1 + height, x1:x1 + width]
    return crop_img, (x1, y1), width, height


# 根据身份证号码的位置推断姓名、性别、名族、出生年月、住址的位置
# :param cardNumPoint1: tuple 身份证号 所在 矩形的左上角坐标
# :param width: int 身份证号码所处的矩形的宽
# :param height: int 身份证号码所处的矩形的高
def find_chinese_regions(gray_img, id_rect):
    # 获取身份证号 坐标
    box = cv2.boxPoints(id_rect)
    box = np.int64(box)
    # 通过顶点获得身份证号码坐标
    _, point, width, height = crop_img_by_box(gray_img, box)

    new_x = point[0] - (width / 18) * 5.5
    new_width = int(width / 5 * 4)

    box = []  # 通过身份证号 位置  推断其他区域位置
    # new_y = cardNumPoint1[1] - height * 6.5
    # 身份证高度
    card_height = height / (0.9044 - 0.7976)
    # 粗略算出图像中身份证上边界的y坐标
    card_y_start = point[1] - card_height  # * 0.8

    # 为了保证不丢失文字区域，姓名的相对位置保留 以身份证上边界作为起始切割点
    new_y = card_y_start + card_height * 0.1  # 0.0967

    # 容错因子，防止矩形存在倾斜导致区域重叠
    factor = 25
    new_y = card_y_start if card_y_start > factor else factor
    new_height = card_height * (0.7616 - 0.0967) + card_height * 0.0967

    # 文字下边界坐标
    new_y_low = (new_y + new_height) if (new_y + new_height) <= point[1] - factor else point[1] - factor
    box.append([new_x, new_y])
    box.append([new_x + new_width, new_y])
    box.append([new_x + new_width, new_y_low])
    box.append([new_x, new_y_low])

    box = np.int0(box)
    # 获取汉字区域坐标信息，并剪切该区域
    return crop_img_by_box(gray_img, box)


# 水平投影边界坐标
def horizontal_projection(binary_img):
    # 水平行边界坐标
    boundaryCoors = []
    (x, y) = binary_img.shape
    a = [0 for z in range(0, x)]
    for i in range(0, x):
        for j in range(0, y):
            if binary_img[i, j] == 0:
                a[i] = a[i] + 1
    # 连续区域标识
    continuouStartFlag = False
    up = down = 0
    # 行高不足总高1/20 临时保存 考虑与下一个行合并
    tempUp = 0
    # 主要解决汉字中上下结构的单子行像素点不连续的问题
    for i in range(0, x):
        if a[i] > 1:
            if not continuouStartFlag:
                continuouStartFlag = True
                up = i
        else:
            if continuouStartFlag:
                continuouStartFlag = False
                down = i+20  # - 1
                if down - up >= x // 20 and down-up <= x//5:  # // 10
                    # 行高小于总高1/20的抛弃
                    boundaryCoors.append([up, down])
                    # print("if boundaryCoors", boundaryCoors)
                else:
                    if tempUp > 0:
                        if down - tempUp >= x // 20 and down - tempUp <= x//10:
                            # 行高小于总高1/20的抛弃
                            boundaryCoors.append([tempUp, down])
                            tempUp = 0
                    else:
                        tempUp = up
    # 姓名 年 月 日 性别-民族 地址
    if len(boundaryCoors) < 4:
        return False
    # print("boundaryCoors", boundaryCoors)
    return boundaryCoors


def get_id_nums(regions, gray_img):
    # 二值化处理
    ret, binary = gray_to_binary(gray_img, method=1)
    # 获得身份证号码
    cardNum=''
    angle = 0
    for rect in regions:
        angle = rect[2]
        # 高、宽、角度标定
        a, b = rect[1]
        if a > b:
            width = a
            height = b
            pts2 = np.float32([[0, height], [0, 0], [width, 0], [width, height]])
        else:
            width = b
            height = a
            angle = 90 + angle
            pts2 = np.float32([[width, height], [0, height], [0, 0], [width, 0]])

        # 透视变换（将图像投影到一个新的视平面）
        box = cv2.boxPoints(rect)
        pts1 = np.float32(box)  # 透视变换前位置
        # 变换矩阵（根据透视前位置和透视后位置计算）
        M = cv2.getPerspectiveTransform(pts1, pts2)
        cropImg = cv2.warpPerspective(binary, M, (int(width), int(height)))  # 输出身份证号码区域透视后的图

        # 计算腐蚀和膨胀的核大小
        kenalx = kenaly = int(math.ceil((height / 100.0)))
        # 膨胀和腐蚀操作
        kenal = cv2.getStructuringElement(cv2.MORPH_RECT, (kenalx, kenaly))
        dilation = cv2.dilate(cropImg, kenal, iterations=1)
        erosion = cv2.erode(dilation, kenal, iterations=1)

        # OCR识别
        cardNum = pytesseract.image_to_string(erosion)

        cardNum.replace(" ", "")
        cardNum.strip()
        cardNum = ''.join(char for char in cardNum if char.isalnum())
        if not cardNum:
            continue

        if is_identi_number(cardNum):
            print('身份证有效')
            return cardNum, angle, rect
        else:
            print('无效身份证=', cardNum)
            continue
    raise('身份证识别失败！！！')


# 分析汉字区域 and 识别提取
def get_chinese_char(gray_char_area_img):
    # 二值化处理
    ret, binary = gray_to_binary(gray_char_area_img, method=1)

    # 2. 膨胀和腐蚀操作，得到可以查找矩形的图片
    # kenalx = kenaly = int(math.ceil((binary.shape[1] / 100.0))) # 计算腐蚀和膨胀的核大小
    # a= (kenalx , kenaly)
    a = cal_element_size(binary)  # 获取核大小
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, a)

    # 微处理去掉小的噪点
    dilation_1 = cv2.dilate(binary, element1, iterations=1)
    erosion_1 = cv2.erode(dilation_1, element1, iterations=1)
    # 文字膨胀与腐蚀使其连成一个整体
    erosion_2 = cv2.erode(erosion_1, element2, iterations=1)
    dilation_2 = cv2.dilate(erosion_2, element1, iterations=1)

    # 获取各个文字行起始坐标
    boundaryCoors = horizontal_projection(dilation_2)
    if not boundaryCoors:
        raise('获取各个文字行起始坐标失败！')

    # 垂直投影对行内字符进行切割
    textLine = 0  # 有效文本行序号
    CARD_NAME = CARD_ETHNIC = CARD_ADDR = ''
    for textLine, boundaryCoor in enumerate(boundaryCoors):
        if textLine == 0:
            vertiCoors, text = get_name(binary, boundaryCoor, gray_char_area_img)  # 获得垂直字符切割坐标 和 字符
            # print("457name:", text)
            CARD_NAME = text
        elif textLine == 1:
            vertiCoors, text = get_gender_ethic(binary, boundaryCoor, gray_char_area_img)  # 获得垂直字符切割坐标 和 字符
            # CARD_GENDER = text[0]
            CARD_ETHNIC = text
            # print("439enthic", CARD_ETHNIC)
        else:
            # print("438")
            vertiCoors, text = get_address(binary, boundaryCoor, gray_char_area_img)  # 获得垂直字符切割坐标 和 字符
            CARD_ADDR += text

    return {'CARD_NAME': CARD_NAME, 'CARD_ETHNIC': CARD_ETHNIC, 'CARD_ADDR':CARD_ADDR}


# 文字通用切割处理
def chars_cut(BinaryImg, horiBoundaryCoor):

    # 列边界坐标
    vertiBoundaryCoors = []
    up, down = horiBoundaryCoor
    lineheight = down - up

    (x, y) = BinaryImg.shape
    a = [0 for z in range(0, y)]

    for j in range(0, y):
        for i in range(up, down):
            if BinaryImg[i, j] == 0:
                a[j] = a[j] + 1

    # 连续区域标识
    continuouStartFlag = False
    left = right = 0

    pixelNum = 0  # 统计每个列的像素数量
    maxWidth = 0  # 最宽的字符长度
    for i in range(0, y):
        # for i in range((down - a[j]), down):
        #     BinaryImg[i, j] = 0
        pixelNum += a[i]  # 统计像素
        if a[i] > 0:
            if not continuouStartFlag:
                continuouStartFlag = True
                left = i
        else:
            if continuouStartFlag:
                continuouStartFlag = False
                right = i
                if right - left > 0:
                    if pixelNum > lineheight * (right - left) // 10:
                        curW = right - left
                        maxWidth = curW if curW > maxWidth else maxWidth
                        vertiBoundaryCoors.append([left, right])
                    pixelNum = 0  # 遇到边界，归零

    return vertiBoundaryCoors, maxWidth


def _chineseCharHandle(BinaryImg, horiBoundaryCoor):
    # 获得该行字符边界坐标
    fator = 0.9
    vertiBoundaryCoors, maxWidth = chars_cut(BinaryImg, horiBoundaryCoor)
    newVertiBoundaryCoors = []  # 字符合并后的纵向坐标

    charNum = len(vertiBoundaryCoors)

    i = 0
    while i < charNum:
        if i + 1 >= charNum:
            newVertiBoundaryCoors.append(vertiBoundaryCoors[i])
            break

        curCharWidth = vertiBoundaryCoors[i][1] - vertiBoundaryCoors[i][0]
        if curCharWidth < maxWidth * fator:
            if vertiBoundaryCoors[i + 1][1] - vertiBoundaryCoors[i][0] <= maxWidth*(2 - fator):
                newVertiBoundaryCoors.append([vertiBoundaryCoors[i][0], vertiBoundaryCoors[i + 1][1]])
                i += 1
            elif curCharWidth > maxWidth / 4:
                newVertiBoundaryCoors.append(vertiBoundaryCoors[i])
        else:
            newVertiBoundaryCoors.append(vertiBoundaryCoors[i])

        i += 1
    return newVertiBoundaryCoors


# 身份证姓名
def get_name(BinaryImg, horiBoundaryCoor, origin=None):

    coors = _chineseCharHandle(BinaryImg, horiBoundaryCoor)
    if len(coors) == 0:
        return coors, '没有检测出名字'

    up, down = horiBoundaryCoor
    # box = np.int64([[coors[0][0], up], [coors[-1][1], up], [coors[-1][1], down], [coors[0][0], down]])
    box = np.int0([[coors[0][0], up], [coors[-1][1], up], [coors[-1][1], down], [coors[0][0], down]])

    text = ''
    if type(origin) == np.ndarray:
        cropImg, _, _, _ = crop_img_by_box(origin, box)

        text = pytesseract.image_to_string(cropImg, 'chi_sim', '7')
        text = text.replace(' ', '')
        text = text.replace('\n', '')
    return coors, text


# 身份证性别 民族
def get_gender_ethic(BinaryImg, horiBoundaryCoor, origin=None):
    text = ''
    coors = _chineseCharHandle(BinaryImg, horiBoundaryCoor)
    up, down = horiBoundaryCoor

    maxW = 0
    for coo in coors:
        curW = coo[1] - coo[0]
        maxW = curW if curW > maxW else maxW

    textIndex = 0
    # if type(origin) == np.ndarray:
    #     for i in range(len(coors)):
    #         box = np.int0([[coors[i][0], up], [coors[i][1], up], [coors[i][1], down], [coors[i][0], down]])
    #         if (coors[i][1] - coors[i][0]) < maxW * 0.88:
    #             continue
    #
    #         cropImg, _, _, _ = crop_img_by_box(origin, box)
    #         # OCR识别
    #         cv2.imshow("hanzu", cropImg)
    #         cv2.waitKey(0)
    #         en_char = pytesseract.image_to_string(cropImg, 'chi_sim', '7')
    #         print("55888888:", en_char)
    #         if en_char == '又' or en_char == '汊' or en_char == '汊':
    #             text = '汉'
    #             print("565", text)
    #         elif all(u'\u4e00' <= ch and ch <= u'\u9fff' for ch in en_char):
    #             text = en_char
    #         textIndex += 1

    # 默认汉族
    if text == '':
        # print("default")
        text = '汉'

    return coors, text


# 身份证地址处理
def get_address(BinaryImg, horiBoundaryCoor, origin=None):

    coors = _chineseCharHandle(BinaryImg, horiBoundaryCoor)
    up, down = horiBoundaryCoor
    box = np.int0([[coors[0][0], up], [coors[-1][1], up], [coors[-1][1], down], [coors[0][0], down]])

    text = ''
    addr_str = ''
    if type(origin) == np.ndarray:
        cropImg, _, _, _ = crop_img_by_box(origin, box)
        # OCR识别
        text = pytesseract.image_to_string(cropImg, 'chi_sim', '7')
        if '月' in text or '年' in text:
            return coors, ''

        addr_str = ''.join(str(x) for x in text)
        # print("addr_str", addr_str)

        addr_str = addr_str.replace("\n", "")
        addr_str = addr_str.replace(" ", "")
        # print("22222:", addr_str)
    return coors, addr_str

