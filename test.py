# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/14/21 3:03 PM
"""
import cv2
import numpy as np
import open3d.visualization
import transforms3d as tf3
import open3d

import utils.features
import utils.geometry
import utils.vis
from utils.vis import PointsAndCamera

from dataset import ApolloScapesDataSet, EurocDataset


def opencv_direct_recover_pose(dataset):
    pc_trian = PointsAndCamera()
    pc_gt = PointsAndCamera()

    poses_gt = []
    poses = []
    for i_frame in range(len(dataset)):
        '''read data'''
        time_frame, frame, pose_gt = dataset[i_frame]
        # time_frame, frame = dataset[i_frame]

        '''print header'''
        print('Sampling time:', time_frame, ' frame shape:', frame.shape)

        '''use current frame as reference frame'''
        i_last_frame = i_frame - 1
        if i_last_frame < 0:
            # record init pose
            # pose gt
            time_frame, frame, pose_gt = dataset[0]
            poses_gt.append(pose_gt)
            # pose est
            poses.append(np.eye(4))
            continue

        '''read last data'''
        time_last_frame, last_frame, pose_last_gt = dataset[i_last_frame]
        # time_last_frame, last_frame = dataset[i_last_frame]

        # fake last frame
        # M = np.eye(3)
        # M[:2, -1] = [100, 0]
        # last_frame = cv2.warpPerspective(frame, M, dsize=frame.shape[:2][::-1])

        '''build gt: read gt and related info'''
        # pose of current = tf from current to world
        tf_current_2_last_gt = np.matmul(np.linalg.inv(pose_last_gt), pose_gt)

        '''start odometer between current and next frame'''
        tf_current_2_last = np.eye(4)

        '''feature matching'''
        pts_last, pts = utils.features.match_pts(last_frame, frame)
        if len(pts_last) < 5:
            print('SIFT: # of features', len(pts_last), ' less than 5')
            continue
        print('SIFT: # of features', len(pts_last))

        '''fundamental matrix'''
        '''essential matrix'''
        '''recover pose'''
        # param R Output rotation matrix with the translation vector, this matrix makes up a tuple that performs a
        # change of basis from the first camera's coordinate system (current frame) to the second camera's coordinate
        # system (last frame)
        results = cv2.recoverPose(points1=pts, points2=pts_last, cameraMatrix1=dataset.cam_intrinsic_matrix,
                                  distCoeffs1=dataset.distortion_coefficients,
                                  cameraMatrix2=dataset.cam_intrinsic_matrix,
                                  distCoeffs2=dataset.distortion_coefficients)

        num_inliers, essential_matrix, rotation, translation, mask = results
        mask = mask.astype(bool).squeeze(axis=-1)
        translation = translation.squeeze(axis=-1)
        tf_current_2_last[:3, :3] = rotation
        tf_current_2_last[:3, -1] = translation
        pts_last, pts = pts_last[mask], pts[mask]

        if num_inliers < 5:
            print('cv2 recoverPose: ', num_inliers, 'points not enough')
            continue
        print('cv2 recoverPose: ', num_inliers, ' points for depth')

        '''triangulation'''
        proj_last = np.matmul(dataset.cam_intrinsic_matrix, tf_current_2_last[:3, :])
        proj = np.matmul(dataset.cam_intrinsic_matrix, np.eye(4)[:3, :])
        pts_4d_homo = cv2.triangulatePoints(projMatr1=proj, projMatr2=proj_last,
                                            projPoints1=pts, projPoints2=pts_last)
        pts_4d_homo = pts_4d_homo / pts_4d_homo[-1, :]
        pts_3d = pts_4d_homo[:3, :].T
        # pts_3d_in_world_frame = pts_3d

        '''print odometer result'''

        '''estimate camera frame transformation between frame'''
        pose_last_frame = poses[-1]
        _, rvec, tvec, mask = cv2.solvePnPRansac(pts_3d, pts_last, cameraMatrix=dataset.cam_intrinsic_matrix, distCoeffs=np.asarray(dataset.distortion_coefficients))
        mask = mask.squeeze(axis=-1)
        pts_3d_pnp_ransac, pts_pnp_ransac = pts_3d[mask], pts[mask]

        # result = cv2.solvePnPRefineLM(pts_3d_pnp_ransac, pts_pnp_ransac, cameraMatrix=dataset.cam_intrinsic_matrix, distCoeffs=np.asarray(dataset.distortion_coefficients), rvec=rvec, tvec=tvec)
        # rvec, tvec = result
        tf_current_2_last_pnp = np.eye(4)
        tf_current_2_last_pnp[:3, :3] = cv2.Rodrigues(rvec)[0]
        tf_current_2_last_pnp[:3, -1:] = tvec

        rvec = rvec.squeeze(axis=-1)
        tvec = tvec.squeeze(axis=-1)

        print()
        print('rvec', rvec)
        print('tvec', tvec, 'norm', np.linalg.norm(tvec))
        print()

        '''camera pose'''
        pose = np.matmul(pose_last_frame, tf_current_2_last_pnp)
        pts_3d_in_world_frame = np.matmul(pose, pts_4d_homo)[:3, :].T

        '''epipolar error'''
        pts_2d_homo, pts_2d_next_homo = pts_last.squeeze(axis=1), pts_last.squeeze(axis=1)
        pts_2d_homo, pts_2d_next_homo = np.hstack((pts_2d_homo, np.ones((len(pts_2d_homo), 1)))), np.hstack((pts_2d_next_homo, np.ones((len(pts_2d_next_homo), 1))))
        fundamental_matrix = np.matmul(np.linalg.inv(dataset.cam_intrinsic_matrix.T),
                                       np.matmul(essential_matrix, np.linalg.inv(dataset.cam_intrinsic_matrix)))
        error_epipolar = 0
        for i in range(len(pts_last)):
            pt_homo, pt_next_homo = pts_2d_homo[i], pts_2d_next_homo[i]
            error_epipolar += np.matmul(pt_next_homo, np.matmul(fundamental_matrix, pt_homo.T))

        '''statistic'''
        pre_fix = '     '
        print('Feature Matching')
        print(pre_fix, '# of match', len(pts_last))
        print(pre_fix, '# of inliers:', num_inliers)
        print(pre_fix, 'mask shape:', mask.shape)

        print('Epipolar Geometry')
        print(pre_fix, 'essential matrix\n', utils.vis.indent_array(essential_matrix, level=2), 'rank:', np.linalg.matrix_rank(essential_matrix))
        print(pre_fix, 'epipolar error / pixel:', error_epipolar, 'mean:', error_epipolar / num_inliers)
        print(pre_fix, 'rx, ry, rz, x, y, z refer to last : ', utils.geometry.tf_2_rpyxyz_degree(tf_current_2_last))
        print(pre_fix, 'rx, ry, rz, x, y, z refer to l gt : ', utils.geometry.tf_2_rpyxyz_degree(tf_current_2_last_gt)[:3], np.asarray(tf_current_2_last_gt[:3, -1])/np.linalg.norm(np.asarray(tf_current_2_last_gt[:3, -1])))

        print('Triangulation')
        num_triangulate_to_show = 3
        print(pre_fix, 'last frame points: ', pts_last.shape, '\n', utils.vis.indent_array(pts_last[0:num_triangulate_to_show], level=2))
        print(pre_fix, 'curr frame points: ', pts.shape, '\n', utils.vis.indent_array(pts[0:num_triangulate_to_show], level=2))
        print(pre_fix, 'homo trian points: ', pts_4d_homo.shape, '\n', utils.vis.indent_array(pts_4d_homo[:, 0:num_triangulate_to_show], level=2), 'norm is\n',
              np.linalg.norm(pts_4d_homo[:, 0:num_triangulate_to_show], axis=0))
        print(pre_fix, '3d compute points: ', pts_3d.shape, '\n', utils.vis.indent_array(pts_3d[0:num_triangulate_to_show], level=2), 'norm is\n ', np.linalg.norm(pts_3d[0:num_triangulate_to_show], axis=-1))

        print('PnP')
        print(pre_fix, 'rx, ry, rz, x, y, z refer to last', utils.geometry.tf_2_rpyxyz(tf_current_2_last_pnp))
        print(pre_fix, 'rx, ry, rz, x, y, z refer to l gt', utils.geometry.tf_2_rpyxyz(tf_current_2_last_gt))
        print(pre_fix, 'rx, ry, rz, x, y, z refer to init', utils.geometry.tf_2_rpyxyz(pose))
        print(pre_fix, 'rx, ry, rz, x, y, z refer to i gt', utils.geometry.tf_2_rpyxyz(pose_gt))

        print()

        '''vis'''
        pts_last_filter, pts_filter = pts_last.squeeze(axis=1), pts.squeeze(axis=1)
        # pts_last_filter, pts_filter = pts_last_filter[mask], pts_filter[mask]

        img_match = utils.vis.draw_matches(last_frame, pts_last_filter, last_frame, pts_filter)

        cv2.namedWindow('match', cv2.WINDOW_NORMAL)
        cv2.imshow('match', img_match)
        cv2.waitKey(1)

        # visualizer.add_cam_frame(pose_last)
        # visualizer.show()
        # cv2.waitKey(0)

        pc_trian.add(pts_3d_in_world_frame, pose)
        pc_trian.show()

        # pc_gt.add(pts_3d, pose_gt)
        # pc_gt.show()

        # if i_frame % 20 == 0:
        #     open3d.visualization.draw_geometries([pc_trian.pc, pc_trian.frame])

        '''update'''
        poses_gt.append(pose_gt)
        poses.append(pose)


def main():
    dataset = ApolloScapesDataSet()
    dataset.load(travel_id=11)

    print('camera matrix', dataset.cam_intrinsic_matrix, type(dataset.cam_intrinsic_matrix))
    print('distortion coefficient', dataset.distortion_coefficients, type(dataset.distortion_coefficients))
    print('first five images', dataset.img_paths[:5])

    # opencv_find_fundamental_first()
    opencv_direct_recover_pose(dataset)


if __name__ == '__main__':
    main()
