import cv2
import functions as func


pathtoimg = r'E:\repository\tansyqinyrproj\uploads\id01.jpg'
img = cv2.imread(pathtoimg)

# 1.  转化成灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 二值化处理
ret, binary = func.gray_to_binary(gray, method=1)

# 3. 形态学处理 膨胀和腐蚀
a = func.cal_element_size(gray)
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 定义结构元素,MORPH_RECT矩形结构，尺寸2x2，较小尺寸用于去除噪声
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, a)  # 尺寸较大

dilation_1 = cv2.dilate(binary, element1, iterations=1)  # 膨胀（即白色像素扩张）去除较小黑点噪声
erosion_1 = cv2.erode(dilation_1, element1, iterations=1)  # 腐蚀（即黑色像素扩张）
erosion_2 = cv2.erode(erosion_1, element2, iterations=1)  # 较大腐蚀
dilation_2 = cv2.dilate(erosion_2, element1, iterations=1)  # 小范围膨胀


# 4.  查找身份证可能的区域
regions = func.find_id_regions(dilation_2)
# 5.  识别身份证号码
id_num, angle, id_rect = func.get_id_nums(regions, gray)
# print(id_num)
if(int(id_num[16]) % 2 == 0):
    CARD_GENDER = "女"
else:
    CARD_GENDER = "男"
CARD_YEAR = id_num[6:10]
CARD_MON = id_num[10:12]
CARD_DAY = id_num[12:14]
print("CARD_YEAR", CARD_YEAR)
print("CARD_MON", CARD_MON)
print("CARD_DAY", CARD_DAY)
print("gender", CARD_GENDER)

# 6.  寻找汉字区域
gray_char_area_img, point, width, height = func.find_chinese_regions(gray, id_rect)  # (gray, id_rect)

# 7.  识别汉字区域字符
text_dict = func.get_chinese_char(gray_char_area_img)
print(text_dict)

CARD_NAME = text_dict['CARD_NAME']
CARD_ETHNIC = text_dict['CARD_ETHNIC']
CARD_ADDR = text_dict['CARD_ADDR']

ocr_text = f'''
CARD_NAME = '{CARD_NAME}'
CARD_GENDER = '{CARD_GENDER}'
CARD_ETHNIC = '{CARD_ETHNIC}'
CARD_YEAR = '{CARD_YEAR}'
CARD_MON = '{CARD_MON}'
CARD_DAY = '{CARD_DAY}'
CARD_ADDR = '{CARD_ADDR}'
'''
print(ocr_text)
# show image
cv2.imshow('origin image', img)
cv2.imshow('binary image', binary)
cv2.imshow('dilation_1 image', dilation_1)
cv2.imshow('erosion_1 image', erosion_1)
cv2.imshow('erosion_2 image', erosion_2)
cv2.imshow('dilate_2 image', dilation_2)
cv2.waitKey(0)
cv2.destroyAllWindows()
