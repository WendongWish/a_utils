# Tool: PyCharm
# coding: utf-8
"""=========================================
# Project_Name: daily
# Author: WenDong
# Date: 2022/4/25 15:25
# Function:
# Description:
========================================="""
import cv2
from PIL import Image


def mask_to_image(save_root, img, mask=None, predict=None, mask_line_thickness=1, prediction_line_thickness=1):
    """
    :param save_root: 输出图像的保存路径
    :param img: 原图的输入路径
    :param mask: 原图所对应的标签路径
    :param predict: 预测结果的路径
    :param mask_line_thickness: 标签的轮廓在原图上所画线的线宽，默认为1
    :param prediction_line_thickness: 预测结果的轮廓在原图上所画线的线宽，默认为1
    :return: None(默认保存到save_root)
    """
    if img is None:
        raise Exception("img is None!")
    else:
        img = cv2.imread(img, cv2.IMREAD_COLOR)

    if save_root is None:
        raise Exception("save_root is None!")

    if (mask is None) and (predict is None):
        raise Exception("mask and predict cannot be None at the same time!")

    elif (mask is not None) and (predict is None):
        ret_mask, thresh_mask = cv2.threshold(cv2.cvtColor(cv2.imread(mask, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY), 127, 255, 0)
        contours_mask, im_mask = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=img, contours=contours_mask, contourIdx=-1, color=(255, 0, 0),
                         thickness=mask_line_thickness)
    elif (predict is not None) and (mask is None):
        ret_prediction, thresh_prediction = cv2.threshold(cv2.cvtColor(cv2.imread(predict, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY), 127, 255, 0)
        contours_prediction, im_prediction = cv2.findContours(thresh_prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=img, contours=contours_prediction, contourIdx=-1, color=(0, 255, 0),
                         thickness=prediction_line_thickness)
    else:
        ret_prediction, thresh_prediction = cv2.threshold(cv2.cvtColor(cv2.imread(predict, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY), 127, 255, 0)
        contours_prediction, im_prediction = cv2.findContours(thresh_prediction, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=img, contours=contours_prediction, contourIdx=-1, color=(0, 255, 0),
                         thickness=prediction_line_thickness)

        ret_mask, thresh_mask = cv2.threshold(cv2.cvtColor(cv2.imread(mask, cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY), 127, 255, 0)
        contours_mask, im_mask = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image=img, contours=contours_mask, contourIdx=-1, color=(255, 0, 0),
                         thickness=mask_line_thickness)

    cv2.imwrite(save_root, img)


if __name__ == '__main__':
    img = '../seg/img.jpg'
    mask = '../seg/mask.png'
    predict = '../seg/predict.png'


    save_root = '../seg/result.png'
    mask_to_image(save_root=save_root, img=img, mask=mask, predict=None)

    image = Image.open(save_root)
    image.show()
