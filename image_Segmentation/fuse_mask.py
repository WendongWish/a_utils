#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time    : 21-09-09 0009 15:54
# @Author  : Hy Wang
# @Software: PyCharm


import cv2
from os.path import join


def mask_to_image(save_root, img, mask, pred, mask_line_thickness, pred_line_thickness):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)  # mask
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)  # pred

    # 获取mask的轮廓
    ret_mask, thresh_mask = cv2.threshold(mask, 127, 255, 0)
    contours_mask, im_mask = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=img, contours=contours_mask, contourIdx=-1, color=(255, 0, 0), thickness=mask_line_thickness)

    # 获取predic的轮廓
    ret_pred, thresh_pred = cv2.threshold(pred, 127, 255, 0)
    contours_pred, im_pred = cv2.findContours(thresh_pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image=img, contours=contours_pred, contourIdx=-1, color=(0, 255, 0), thickness=pred_line_thickness)

    cv2.imwrite(save_root, img)
    # cv2.namedWindow('a')
    # cv2.imshow('a', img)
    # cv2.waitKey(0)


if __name__ == '__main__':
    root = 'G:\Datasets\ISIC2018\ISIC2018_Task1_Test_Input_GT\Iresult\Ia_3smallaim\\None_csSE_1361218_56\\'
    pic_num = str(19)
    save_root = join(root, 'result.png')
    img = cv2.imread(join(root, pic_num + '_image.png'))  #
    mask = cv2.imread(join(root, pic_num + '_GT.png'))  #
    pred = cv2.imread(join(root, pic_num + '_SR.png'))  #
    mask_to_image(save_root, img, mask, pred, 2, 2)