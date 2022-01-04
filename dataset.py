# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/13/21 2:45 PM
"""
import os
import sys

import cv2
import numpy as np
import yaml
import transforms3d as tf3

from utils.vis import PointsAndCamera


def get_baidu_dataset_file_paths(dataset_dir_path, travel_id):
    # camera intrinsic, probably undistorted already
    camera1_intrinsic_parameters_file_path = dataset_dir_path + '/camera_params/Camera_1.cam'
    camera2_intrinsic_parameters_file_path = dataset_dir_path + '/camera_params/Camera_2.cam'

    # travel id: each travel is a sequence of photo
    travel_id_prefix = 'Record' + '000'[len(str(travel_id)):]
    travel_id = travel_id_prefix + str(travel_id) + '/'

    # each line of pose file: image name, roll, pitch, yal, x, y, z
    pose_cam1_paths = dataset_dir_path + '/pose/' + travel_id + '/Camera_1.txt'
    pose_cam2_paths = dataset_dir_path + '/pose/' + travel_id + '/Camera_2.txt'

    # image folder
    img_cam1_dir_paths = dataset_dir_path + '/image/' + travel_id + '/Camera_1/'
    img_cam2_dir_paths = dataset_dir_path + '/image/' + travel_id + '/Camera_2/'

    # image paths
    img1_paths = [img_cam1_dir_paths + img_name for img_name in os.listdir(img_cam1_dir_paths)]
    img2_paths = [img_cam2_dir_paths + img_name for img_name in os.listdir(img_cam2_dir_paths)]

    assert os.path.exists(camera1_intrinsic_parameters_file_path)
    assert os.path.exists(camera2_intrinsic_parameters_file_path)
    assert os.path.exists(pose_cam1_paths)
    assert os.path.exists(pose_cam2_paths)
    assert os.path.exists(img_cam1_dir_paths)
    assert os.path.exists(img_cam2_dir_paths)
    assert len(img1_paths) > 0
    assert len(img2_paths) > 0
    return camera1_intrinsic_parameters_file_path, img_cam1_dir_paths, pose_cam1_paths, img1_paths, \
           camera2_intrinsic_parameters_file_path, img_cam2_dir_paths, pose_cam2_paths, img2_paths


def get_cam_intrinsic_from_cam_file(file_path):
    f = open(file_path, 'r')
    fx, fy, cx, cy = None, None, None, None
    for line in f.readlines():
        if 'fx' in line or 'fy' in line or 'Cx' in line or 'Cy' in line:
            index_equal_sign = line.find('=')
            index_new_line = line.find('\n')
            num = float(line[index_equal_sign + 1:index_new_line])
            if 'fx' in line:
                fx = num
            elif 'fy' in line:
                fy = num
            elif 'Cx' in line:
                cx = num
            elif 'Cy' in line:
                cy = num
    return fx, fy, cx, cy


def get_pose_matrix_from_pose_file(img_name, file_path):
    assert img_name[:-4] == '.jpg', img_name + ' is not a valid image format'
    f = open(file_path, 'r')
    for line in f.readlines():
        if img_name in line:
            nums = []
            index_img_name = line.find(img_name)
            index_start = index_img_name + len(img_name) + 1
            index_end = line.find(',')
            for i in range(6):
                nums.append(float(line[index_start:index_end]))
                index_start = index_end + 1
                index_end = line.find(line[index_start])
    return


class ApolloScapesDataSet:
    def __init__(self):
        self.len = 0
        self.cam_para_path, self.pose_path, self.img_dir, self.img_paths = None, None, None, None
        self.cam_intrinsic_matrix = None
        self.distortion_coefficients = None
        self.data = None

    def load(self, dataset_dir_path='./data/self-localization-sample/self-localization-sample/zpark/', travel_id=1):
        # file paths, image paths
        self.cam_para_path, self.img_dir, self.pose_path, self.img_paths, *_ = \
            get_baidu_dataset_file_paths(dataset_dir_path, travel_id)

        # camera intrinsic
        fu, fv, cu, cv = get_cam_intrinsic_from_cam_file(self.cam_para_path)
        self.cam_intrinsic_matrix = np.array([[fu, 0, cu],
                                              [0, fv, cv],
                                              [0, 0, 1]])
        self.distortion_coefficients = np.array([0.0, 0.0, 0.0, 0.0])

        # camera pose, img_name, roll, pitch, yal, x, y, z
        self.data = {}
        f = open(self.pose_path, 'r')
        for line in f.readlines():
            img_name, roll_pitch_yal_x_y_z = line.split()
            roll, pitch, yal, x, y, z = roll_pitch_yal_x_y_z.split(',')
            roll, pitch, yal, x, y, z = float(roll), float(pitch), float(yal), float(x), float(y), float(z)
            self.data[img_name] = np.array([roll, pitch, yal, x, y, z])
        f.close()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        assert isinstance(item, int), 'Expect ' + str(item) + ' to be a integer'
        img_name = list(self.data.keys())[item]
        img_name_first = list(self.data.keys())[0]

        # time
        time = img_name[:13]

        # image
        img_path = self.img_dir + '/' + img_name
        assert os.path.exists(img_path)
        img = cv2.imread(img_path)

        # pose gt, relative pose refer to the first pose
        rpyxyz_first = self.data[img_name_first]
        rpyxyz = self.data[img_name]

        pose_first = np.eye(4)
        # pose[:3, :3] = tf3.taitbryan.euler2mat(*rpyxyz_first[:3])
        pose_first[:3, :3] = tf3.euler.euler2mat(*rpyxyz_first[:3])
        pose_first[:3, -1] = rpyxyz_first[3:]

        pose = np.eye(4)
        # pose[:3, :3] = tf3.taitbryan.euler2mat(*rpyxyz[:3])
        pose[:3, :3] = tf3.euler.euler2mat(*rpyxyz[:3])
        pose[:3, -1] = rpyxyz[3:]

        pose_relative = np.matmul(np.linalg.inv(pose_first), pose)
        return time, img, pose_relative


class EurocDataset:
    def __init__(self):
        self.times_path, self.cam_para_path, self.pose_path = '', '', ''
        self.times, self.img_paths = [], []
        self.cam_intrinsic_matrix = None
        self.distortion_coefficients = None

    def load(self, dataset_dir_path='./data/EuRoC/MH_01_easy/mav0/cam0'):
        self.cam_para_path = dataset_dir_path + '/sensor.yaml'
        self.times_path = dataset_dir_path + '/data.csv'
        img_dir_path = dataset_dir_path + '/data/'

        """camera parameters"""
        f = open(self.cam_para_path, 'r')
        camera_info = yaml.safe_load(f)
        fu, fv, cu, cv = camera_info['intrinsics']
        self.cam_intrinsic_matrix = np.array([[fu, 0, cu],
                                              [0, fv, cv],
                                              [0, 0, 1]])
        self.distortion_coefficients = np.asarray(camera_info['distortion_coefficients'])

        """time stamp and images"""
        f = open(self.times_path, 'r')
        for line in f.readlines():
            if len(line) == 0 or line[0] == '#':
                continue
            time, img_name = line.split(',')
            time = float(time)
            self.times.append(time / 1e9)
            self.img_paths.append(img_dir_path + img_name[:-1])

        """"""
        return True

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        assert isinstance(item, int), 'Expect ' + str(item) + ' to be a integer'
        return self.times[item], cv2.imread(self.img_paths[item]) if item < self.__len__() else None


def main():
    # dataset_dir_path = './data/self-localization-sample/self-localization-sample/zpark/'
    # cam_para_path, pose_path, img_paths = get_baidu_dataset_file_paths(dataset_dir_path)
    # print(cam_para_path)
    # print(pose_path)
    # print(img_paths)
    vis = PointsAndCamera()

    dataset = ApolloScapesDataSet()
    dataset.load(travel_id=11)
    print('camera matrix', dataset.cam_intrinsic_matrix, type(dataset.cam_intrinsic_matrix))
    print('distortion coefficient', dataset.distortion_coefficients, type(dataset.distortion_coefficients))
    print('first five images', dataset.img_paths[:5])

    poses = []
    for data in dataset:
        # time, img = data
        time, img, pose = data
        print('Sampling time:', time, 'shape:', img.shape)
        print('pose', pose)

        # tf
        if len(poses) > 0:
            tf_current_2_last_gt = np.matmul(pose, np.linalg.inv(poses[-1]))
            print(tf_current_2_last_gt[:3, -1])

        # update
        poses.append(pose)
        print()

        '''vis'''
        # 3d camera pose
        vis.add(frame_pose=pose)
        vis.show()

        # camera view
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', img)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
