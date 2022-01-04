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
# from dataset import ApolloScapesDataSet
#
#
# def opencv_direct_recover_pose(dataset, visualizer):
#     pc_trian = PointsAndCamera()
#
#     pose_last_frame = np.eye(4)
#     for i_frame in range(len(dataset)):
#         '''use current frame as reference frame'''
#         i_last_frame = i_frame - 1
#         if i_last_frame < 0:
#             continue
#
#         '''read data'''
#         time_last_frame, last_frame, roll_pitch_yal_x_y_z_last_frame = dataset[i_last_frame]
#
#         time_frame, frame, roll_pitch_yal_x_y_z_frame = dataset[i_frame]
#
#         # fake last frame
#         M = np.eye(3)
#         M[:2, -1] = [100, 0]
#         last_frame = cv2.warpPerspective(frame, M, dsize=frame.shape[:2][::-1])
#
#         '''print header'''
#         print('Sampling time:', time_frame, ' frame shape:', frame.shape)
#
#         '''build gt: read gt and related info'''
#         pose_gt = np.eye(4)
#         pose_gt[:3, :3] = tf3.euler.euler2mat(*roll_pitch_yal_x_y_z_last_frame[:3])
#         pose_gt[:3, 3] = roll_pitch_yal_x_y_z_last_frame[3:]
#
#         '''start odometer between current and next frame'''
#         tf_current_2_last = np.eye(4)
#
#         '''feature matching'''
#         pts_last, pts = utils.features.match_pts(last_frame, frame)
#         if len(pts_last) < 5:
#             continue
#         print('SIFT: # of features', len(pts_last))
#
#         '''fundamental matrix'''
#         '''essential matrix'''
#         '''recover pose'''
#         # param R Output rotation matrix with the translation vector, this matrix makes up a tuple that performs a
#         # change of basis from the first camera's coordinate system (current frame) to the second camera's coordinate
#         # system (last frame)
#         results = cv2.recoverPose(points1=pts, points2=pts_last, cameraMatrix1=dataset.cam_intrinsic_matrix,
#                                   distCoeffs1=dataset.distortion_coefficients,
#                                   cameraMatrix2=dataset.cam_intrinsic_matrix,
#                                   distCoeffs2=dataset.distortion_coefficients)
#
#         num_inliers, essential_matrix, rotation, translation, mask = results
#         mask = mask.astype(bool).squeeze(axis=-1)
#         translation = translation.squeeze(axis=-1)
#         tf_current_2_last[:3, :3] = rotation
#         tf_current_2_last[:3, -1] = translation
#         pts_last, pts = pts_last[mask], pts[mask]
#
#         if num_inliers < 5:
#             continue
#         print('cv2 recoverPose: ', num_inliers, ' points for depth')
#
#         '''triangulation'''
#         proj_last = np.matmul(dataset.cam_intrinsic_matrix, np.eye(4)[:3, :])
#         proj = np.matmul(dataset.cam_intrinsic_matrix, tf_current_2_last[:3, :])
#         pts_4d_homo = cv2.triangulatePoints(projMatr1=proj, projMatr2=proj_last,
#                                             projPoints1=pts, projPoints2=pts_last)
#         pts_4d_homo = pts_4d_homo / pts_4d_homo[-1, :]
#         pts_3d_in_last_frame = pts_4d_homo[:3, :].T
#         pts_3d_in_world_frame = np.matmul(pose_last_frame, pts_4d_homo)[:3, :].T
#
#         '''print odometer result'''
#         print('curr frame points: \n', pts_last.shape, '\n', pts_last[0:5])
#         print('next frame points: \n', pts.shape, '\n', pts[0:5])
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
#         pose = np.matmul(pose_last_frame, tf_current_2_last)
#         # else:   # skip this step
#         #     print('No enough points to pnp')
#         #     continue
#
#         '''epipolar error'''
#         pts_2d_homo, pts_2d_next_homo = pts_last.squeeze(axis=1), pts_last.squeeze(axis=1)
#         pts_2d_homo, pts_2d_next_homo = np.hstack((pts_2d_homo, np.ones((len(pts_2d_homo), 1)))), np.hstack((pts_2d_next_homo, np.ones((len(pts_2d_next_homo), 1))))
#         fundamental_matrix = np.matmul(np.linalg.inv(dataset.cam_intrinsic_matrix.T),
#                                        np.matmul(essential_matrix, np.linalg.inv(dataset.cam_intrinsic_matrix)))
#         error_epipolar = 0
#         for i in range(len(pts_last)):
#             pt_homo, pt_next_homo = pts_2d_homo[i], pts_2d_next_homo[i]
#             error_epipolar += np.matmul(pt_next_homo, np.matmul(fundamental_matrix, pt_homo.T))
#
#         '''statistic'''
#         print('# of match', len(pts_last))
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
#         pts_last_filter, pts_filter = pts_last.squeeze(axis=1), pts.squeeze(axis=1)
#         # pts_last_filter, pts_filter = pts_last_filter[mask], pts_filter[mask]
#
#         img_match = utils.vis.draw_matches(last_frame, pts_last_filter, last_frame, pts_filter)
#
#         cv2.namedWindow('match', cv2.WINDOW_NORMAL)
#         cv2.imshow('match', img_match)
#         cv2.waitKey(1)
#
#         # visualizer.add_cam_frame(pose_last)
#         # visualizer.show()
#         # cv2.waitKey(0)
#
#         pc_trian.add(pts_3d_in_last_frame, pose_last_frame)
#
#         pc_trian.show()
#         # if i_frame % 20 == 0:
#         #     open3d.visualization.draw_geometries([pc_trian.pc, pc_trian.frame])
#
#         '''update'''
#         pose_last_frame = pose
#
#
# def main():
#     dataset = ApolloScapesDataSet()
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
