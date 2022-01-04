# # -*- coding: utf-8 -*-
#
# """
# @author: Cheng Chen
# @email: chengc0611@gmail.com
# @time: 12/14/21 3:03 PM
# """
# import cv2
# import numpy as np
# import open3d.visualization
# import transforms3d as tf3
# import open3d
#
# import utils.features
# import utils.vis
# from utils.vis import CameraPosesVis, PointsAndCamera
#
# from dataset import EurocDataset
#
#
# def opencv_direct_recover_pose(dataset, visualizer):
#     last_frame, frame = None, None
#     pose_last, pose = None, None
#     pc_trian = PointsAndCamera()
#
#     for i_frame, (time, img) in enumerate(dataset):
#         print('Sampling time:', time, ' img shape:', img.shape)
#
#         if last_frame is None:
#             last_frame = img
#             pose_last = np.eye(4)
#             continue
#
#         frame = img
#         tf = np.eye(4)
#
#         '''feature matching'''
#         pts_last, pts = utils.features.match_pts(last_frame, frame)
#         # pts_last, pts = pts_last.squeeze(axis=1), pts.squeeze(axis=1)
#         if len(pts) < 5:
#             continue
#         print('SIFT: # of features', len(pts))
#
#         '''fundamental matrix'''
#         '''essential matrix'''
#         '''recover pose'''
#         # param R Output rotation matrix with the translation vector, this matrix makes up a tuple that performs a
#         # change of basis from the first camera 's coordinate system to the second camera' s coordinate system
#         results = cv2.recoverPose(points1=pts, points2=pts_last, cameraMatrix1=dataset.cam_intrinsic_matrix,
#                                   distCoeffs1=np.asarray(dataset.distortion_coefficients),
#                                   cameraMatrix2=dataset.cam_intrinsic_matrix,
#                                   distCoeffs2=np.asarray(dataset.distortion_coefficients))
#
#         num_inliers, essential_matrix, rotation, translation, mask = results
#         mask = mask.astype(bool).squeeze(axis=-1)
#         translation = translation.squeeze(axis=-1)
#         tf[:3, :3] = rotation
#         tf[:3, -1] = translation
#         pts_last, pts = pts_last[mask], pts[mask]
#
#         if num_inliers < 5:
#             continue
#         print('cv2 recoverPose: ', num_inliers, ' points for depth')
#
#         '''triangulation'''
#         proj = np.matmul(dataset.cam_intrinsic_matrix, tf[:3, :])
#         proj_last = np.matmul(dataset.cam_intrinsic_matrix, np.eye(4)[:3, :])
#         pts_4d_homo = cv2.triangulatePoints(projMatr1=proj_last, projMatr2=proj,
#                                             projPoints1=pts_last, projPoints2=pts)
#         pts_4d_homo = pts_4d_homo / pts_4d_homo[-1, :]
#         pts_3d_in_last_frame = pts_4d_homo[:3, :].T
#         pts_3d_in_world_frame = np.matmul(pose_last, pts_4d_homo)[:3, :].T
#
#         print('last frame points: \n', pts_last.shape, '\n', pts_last[0:5])
#         print('curr frame points: \n', pts.shape, '\n', pts[0:5])
#         print('homogenous triangulated points: \n', pts_4d_homo.shape, '\n', pts_4d_homo[:, 0:5], 'norm is\n', np.linalg.norm(pts_4d_homo[:, 0:5], axis=0))
#         print('3d points \n', pts_3d_in_last_frame.shape, '\n', pts_3d_in_last_frame[0:5], 'norm is\n ', np.linalg.norm(pts_3d_in_last_frame[0:5], axis=-1))
#
#         '''vis 3d points'''
#         # pc_trian = PointsAndCamera()
#         # pc_trian.load(pts_3d_in_world_frame)
#         # pc_trian.show()
#
#         '''pose estimation'''
#
#         # if pts_3d.shape[1] >= 6:
#         # _, rvec, tvec, mask = cv2.solvePnPRansac(pts_3d, pts, cameraMatrix=dataset.cam_intrinsic_matrix, distCoeffs=np.asarray(dataset.distortion_coefficients))
#         # mask = mask.squeeze(axis=-1)
#         # pts_3d_pnp_ransac, pts_pnp_ransac = pts_3d[mask], pts[mask]
#         #
#         # result = cv2.solvePnPRefineLM(pts_3d_pnp_ransac, pts_pnp_ransac, cameraMatrix=dataset.cam_intrinsic_matrix, distCoeffs=np.asarray(dataset.distortion_coefficients), rvec=rvec, tvec=tvec)
#         # rvec, tvec = result
#         # tf = np.eye(4)
#         # tf[:3, :3] = cv2.Rodrigues(rvec)[0]
#         # tf[:3, -1:] = tvec
#
#         pose = np.matmul(pose_last, tf)
#         # else:   # skip this step
#         #     print('No enough points to pnp')
#         #     continue
#
#         '''epipolar error'''
#         pts_2d_last_homo, pts_2d_homo = pts_last.squeeze(axis=1), pts.squeeze(axis=1)
#         pts_2d_last_homo, pts_2d_homo = np.hstack((pts_2d_last_homo, np.ones((len(pts_2d_last_homo), 1)))), np.hstack((pts_2d_homo, np.ones((len(pts_2d_homo), 1))))
#         fundamental_matrix = np.matmul(np.linalg.inv(dataset.cam_intrinsic_matrix.T),
#                                        np.matmul(essential_matrix, np.linalg.inv(dataset.cam_intrinsic_matrix)))
#         error_epipolar = 0
#         for i in range(len(pts)):
#             pt_last_homo, pt_homo = pts_2d_last_homo[i], pts_2d_homo[i]
#             error_epipolar += np.matmul(pt_last_homo, np.matmul(fundamental_matrix, pt_homo.T))
#
#         '''statistic'''
#         print('# of match', len(pts))
#         print('# of inliers:', num_inliers)
#         print('mask shape:', mask.shape)
#         print('essential matrix\n', essential_matrix, 'rank:', np.linalg.matrix_rank(essential_matrix))
#         print('epipolar error / pixel:', error_epipolar, 'mean:', error_epipolar / num_inliers)
#         print('rotation matrix square:\n', np.matmul(rotation.T, rotation))
#         print('translation:\n', translation, 'norm:', np.linalg.norm(translation))
#         print('translation wrt initial', pose[:3, -1])
#         print()
#
#         '''vis'''
#         pts_last_filter, pts_filter = pts.squeeze(axis=1), pts_last.squeeze(axis=1)
#         # pts_last_filter, pts_filter = pts_last_filter[mask], pts_filter[mask]
#
#         img_match = utils.vis.draw_matches(frame, pts_filter, last_frame, pts_last_filter)
#
#         cv2.namedWindow('match', cv2.WINDOW_NORMAL)
#         cv2.imshow('match', img_match)
#         cv2.waitKey(1)
#
#         # visualizer.add_cam_frame(pose_last)
#         # visualizer.show()
#         # cv2.waitKey(0)
#
#         if i_frame == 1:
#             pc_trian.load(pts_3d_in_last_frame)
#         else:
#             pc_trian.add(pts_3d_in_last_frame, pose_last)
#
#         pc_trian.show()
#         # if i_frame % 20 == 0:
#         #     open3d.visualization.draw_geometries([pc_trian.pc, pc_trian.frame])
#
#         '''update'''
#         last_frame = img
#         pose_last = pose
#
#
# def opencv_find_fundamental_first(dataset):
#     last_frame, frame = None, None
#
#     for time, img in dataset:
#         print('Sampling time:', time, ' img shape:', img.shape)
#
#         if last_frame is None:
#             last_frame = img
#             continue
#
#         frame = img
#
#         '''feature matching'''
#         pts_last, pts = utils.features.match_pts(last_frame, frame)
#
#         '''fundamental matrix'''
#         fundamental_matrix, inliers_mask_f = cv2.findFundamentalMat(pts_last, pts, method=cv2.FM_RANSAC,
#                                                                                    ransacReprojThreshold=4,
#                                                                                    confidence=0.8, maxIters=10000)
#         # essential_matrix_last_2_current, inliers_mask_e = cv2.findEssentialMat(pts_last, pts,
#         #                                                                        dataset.cam_intrinsic_matrix,
#         #                                                                        method=cv2.FM_RANSAC)
#
#         if fundamental_matrix is None:
#             print('Cannot find fundamental matrix for ', time)
#             continue
#
#         inliers_mask_f = inliers_mask_f.astype(bool)
#         pts_last_match, pts_match = pts_last[inliers_mask_f], pts[inliers_mask_f]
#
#         '''essential matrix'''
#         essential_matrix = cv2.findEssentialMat(pts_last, pts, dataset.cam_intrinsic_matrix)
#
#         '''recover pose'''
#         rotation_1, rotation_2, translation = cv2.decomposeEssentialMat(essential_matrix)
#
#         '''epipolar error'''
#         error_epipolar = 0
#
#         '''statistics'''
#         print('# of match', len(pts))
#         print('# of inliers:', len(inliers_mask_f))
#         print('mask shape:', inliers_mask_f.shape)
#         print('fundamental matrix\n', fundamental_matrix, '\nrank is',
#               np.linalg.matrix_rank(fundamental_matrix))
#         print('essential matrix\n', essential_matrix, 'rank:', np.linalg.matrix_rank(essential_matrix))
#         pts_last, pts = pts_last.squeeze(axis=1), pts.squeeze(axis=1)
#         pts_last, pts = np.hstack((pts_last, np.ones((len(pts_last), 1)))), np.hstack((pts, np.ones((len(pts), 1))))
#
#         for i in range(len(pts)):
#             pt_last, pt = pts_last[i], pts[i]
#             error_epipolar += np.matmul(pt, np.matmul(fundamental_matrix, pt_last.T))
#         print('epipolar error / pixel:', error_epipolar, 'mean:', error_epipolar / len(inliers_mask_f))
#         # print('rotation matrix square:\n', np.matmul(rotation.T, rotation))
#         # print('translation:\n', translation)
#
#
#         print()
#
#         '''vis'''
#         img_match = utils.vis.draw_matches(frame, pts[inliers_mask_f], last_frame, pts_last[inliers_mask_f])
#
#         cv2.namedWindow('match', cv2.WINDOW_NORMAL)
#         cv2.imshow('match', img_match)
#         cv2.waitKey(0)
#
#         '''update'''
#         last_frame = img
#
#
# def opencv_find_essential_first(dataset):
#     last_frame, frame = None, None
#
#     for time, img in dataset:
#         print('Sampling time:', time, ' img shape:', img.shape)
#
#         if last_frame is None:
#             last_frame = img
#             continue
#
#         frame = img
#
#         '''feature matching'''
#         pts_last, pts = utils.features.match_pts(last_frame, frame)
#
#         '''fundamental matrix'''
#         '''essential matrix'''
#         essential_matrix, inliers_mask = cv2.findEssentialMat(pts_last, pts, dataset.cam_intrinsic_matrix,
#                                                               method=cv2.FM_RANSAC)
#         '''recover pose'''
#
#         if essential_matrix is None:
#             print('Cannot find essential matrix for ', time)
#             continue
#
#         inliers_mask = inliers_mask.astype(bool)
#         pts_last_match, pts_match = pts_last[inliers_mask], pts[inliers_mask]
#
#         rotation_1, rotation_2, translation = cv2.decomposeEssentialMat(essential_matrix)
#
#         '''epipolar error'''
#         fundamental_matrix = np.matmul(dataset.cam_intrinsic_matrix.T,
#                                        np.matmul(essential_matrix, dataset.cam_intrinsic_matrix))
#         error_epipolar = 0
#
#         '''statistics'''
#         print('# of match', len(pts))
#         print('# of inliers:', len(inliers_mask))
#         print('mask shape:', inliers_mask.shape)
#         print('fundamental matrix\n', fundamental_matrix, '\nrank is',
#               np.linalg.matrix_rank(fundamental_matrix))
#         print('essential matrix\n', essential_matrix, 'rank:', np.linalg.matrix_rank(essential_matrix))
#
#         pts_last_match, pts_match = pts_last_match.squeeze(axis=1), pts_match.squeeze(axis=1)
#         pts_last_match, pts_match = np.hstack((pts_last_match, np.ones((len(pts_last_match), 1)))), np.hstack((pts_match, np.ones((len(pts_match), 1))))
#
#         for i in range(len(pts)):
#             pt_last, pt = pts_last[i], pts[i]
#             error_epipolar += np.matmul(pt, np.matmul(fundamental_matrix, pt_last.T))
#         print('epipolar error / pixel:', error_epipolar, 'mean:', error_epipolar / len(inliers_mask))
#         # print('rotation matrix square:\n', np.matmul(rotation.T, rotation))
#         # print('translation:\n', translation)
#         print()
#
#         '''vis'''
#         img_match = utils.vis.draw_matches(frame, pts[inliers_mask], last_frame, pts_last[inliers_mask])
#
#
#         cv2.namedWindow('match', cv2.WINDOW_NORMAL)
#         cv2.imshow('match', img_match)
#         cv2.waitKey(0)
#
#         '''update'''
#         last_frame = img
#
#
# def main():
#     dataset = EurocDataset()
#     dataset.load()
#     visualizer = CameraPosesVis()
#     # visualizer = None
#
#     print('camera matrix', dataset.cam_intrinsic_matrix, type(dataset.cam_intrinsic_matrix))
#     print('distortion coefficient', dataset.distortion_coefficients, type(dataset.distortion_coefficients))
#     print('first five images', dataset.img_paths[:5])
#
#     # opencv_find_fundamental_first()
#     opencv_direct_recover_pose(dataset, visualizer)
#
#
# if __name__ == '__main__':
#     main()
#     # a = np.asarray([[-3.16641733e-03, 2.93345722e-01, -6.42577449e-01],
#     #                 [-2.90305464e-01, -3.04329840e-04, 3.34897771e-02],
#     #                 [6.43905142e-01, -3.17769790e-02, -3.15511139e-03]])
#     #
#     # b = np.asarray([[9.99988916e-01, 5.51286887e-04, -4.67595709e-03],
#     #                 [-5.30290490e-04, 9.99989778e-01, 4.49034052e-03],
#     #                 [4.67838476e-03, -4.48781114e-03, 9.99978986e-01]])
#     #
#     # print(np.linalg.matrix_rank(a))
#     # print(np.matmul(b, b.T))
