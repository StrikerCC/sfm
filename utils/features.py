# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/14/21 2:55 PM
"""

import cv2
import numpy as np


def corners(img):

    return []


def sift_features(img, flag_debug=False):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)

    if flag_debug:
        img_kp = cv2.drawKeypoints(img, kp, None)
        print(len(kp))
        print(kp[0])
        print(des.shape)
        cv2.namedWindow('kp', cv2.WINDOW_NORMAL)
        cv2.imshow('kp', img_kp)
        cv2.waitKey(0)

    return kp, des


def match_pts(img1, img2, flag_debug=False):
    if img1 is None or img2 is None:
        return None
    kp1, des1 = sift_features(img1, flag_debug)
    kp2, des2 = sift_features(img2, flag_debug)
    bf = cv2.BFMatcher_create()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [first for first, second in matches if first.distance < 0.5 * second.distance]

    if flag_debug:
        print('Get ', len(good_matches), ' good matches')
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.imshow('match', img3)
        cv2.waitKey(0)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    return pts1, pts2


def main():
    # img1 = cv2.imread('../data/graf/img1.ppm')
    # img2 = cv2.imread('../data/graf/img2.ppm')
    # pts1, pts2 = match_pts(img1, img2)
    # print(pts1)
    # print(pts2)
    # print(pts2.shape)
    # img_1, img_2 = img1, img2
    #
    # H, status = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=4)
    # imgOutput = cv2.warpPerspective(img1, H, (img_1.shape[1] + img_2.shape[1], img_1.shape[0]))
    # imgOutput[0:img_2.shape[0], 0:img_2.shape[1]] = img_2
    #
    # cv2.namedWindow('warp', cv2.WINDOW_NORMAL)
    # cv2.imshow('warp', imgOutput)
    # cv2.waitKey(0)
    return


if __name__ == '__main__':
    main()
