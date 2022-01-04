# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/16/21 10:53 AM
"""
import numpy as np
import transforms3d as tf3
import cv2


def rotation_between_3dv_frame(v_src, v_tgt):
    """v_tgt = R * v_src"""
    assert len(v_src.shape) < 3 and len(v_tgt.shape) < 3, 'v1 shape: ' + str(v_src.shape) + 'v2 shape: ' + str(v_tgt.shape)
    if len(v_src.shape) > 1:
        assert 1 in v_src.shape, 'invalid shape ' + str(v_src.shape)
        v_src = v_src.squeeze(axis=v_src.shape.index(1))
    if len(v_tgt.shape) > 1:
        assert 1 in v_src.shape, 'invalid shape ' + str(v_src.shape)
        v_tgt = v_tgt.squeeze(axis=v_tgt.shape.index(1))
    v_src = v_src / np.linalg.norm(v_src)
    v_tgt = v_tgt / np.linalg.norm(v_tgt)
    # assume v1 is parallel to (1, 0, 0) in it's local frame
    R_src_2_world = np.eye(len(v_src))
    r_x = v_src    # v1 frame (1,0,0) in world frame
    r_z = np.cross(R_src_2_world[:, 0], r_x) if r_x[0] != 1 else np.array([0, 0, 1])    # v1 frame (0,1,0) in world frame
    r_z = r_z / np.linalg.norm(r_z)
    r_y = np.cross(r_z, r_x)    # v1 frame (0,1,0) in world frame
    r_y = r_y / np.linalg.norm(r_y)
    R_src_2_world[:, 0] = r_x
    R_src_2_world[:, 1] = r_y
    R_src_2_world[:, 2] = r_z

    R_tgt_2_world = np.eye(len(v_tgt))
    r_x = v_tgt    # v1 frame (1,0,0) in world frame
    r_z = np.cross(R_tgt_2_world[:, 0], r_x) if r_x[0] != 1 else np.array([0, 0, 1])    # v1 frame (0,1,0) in world frame
    r_z = r_z / np.linalg.norm(r_z)
    r_y = np.cross(r_z, r_x)    # v1 frame (0,1,0) in world frame
    r_y = r_y / np.linalg.norm(r_y)
    R_tgt_2_world[:, 0] = r_x
    R_tgt_2_world[:, 1] = r_y
    R_tgt_2_world[:, 2] = r_z
    return np.matmul(R_tgt_2_world.T, R_src_2_world).T


def rotation_between_3dv_rotate_axis(v_src, v_tgt):
    v_src, v_tgt = v_src / np.linalg.norm(v_src), v_tgt / np.linalg.norm(v_tgt)
    v_rotate = np.cross(v_src, v_tgt)
    v_rotate /= np.linalg.norm(v_rotate)
    angle = np.arccos(np.dot(v_src, v_tgt) / np.linalg.norm(v_src) / np.linalg.norm(v_tgt))
    return tf3.euler.axangle2mat(v_rotate, angle)


def tf_2_rpyxyz(tf):
    assert tf.shape == (4, 4)
    assert np.allclose(np.matmul(tf[:3, :3].T, tf[:3, :3]), np.eye(3))

    roll, pitch, yal = tf3.euler.mat2euler(tf[:3, :3])
    x, y, z = tf[:3, -1]
    return roll, pitch, yal, x, y, z


def tf_2_rpyxyz_degree(tf):
    assert tf.shape == (4, 4)
    assert np.allclose(np.matmul(tf[:3, :3].T, tf[:3, :3]), np.eye(3))
    roll, pitch, yal, x, y, z = tf_2_rpyxyz(tf)
    return roll/np.pi*180.0, pitch/np.pi*180.0, yal/np.pi*180.0, x, y, z


def pnp_DLT(pts_3d, pts_img, cam_intrinsic_matrix):
    assert pts_3d.shape[1] == 3
    assert pts_img.shape[1] == 2
    assert cam_intrinsic_matrix.shape == (3, 3)

    return


def pnp_opencv_test():
    # 3d points
    pts_3d = np.array([[0.0, 0.0, 0.0],
                       [100.0, 0.0, 0.0],
                       [100.0, 100.0, 0.0],
                       [0.0, 100.0, 0.0]])
    pts_3d = pts_3d.T
    pts_3d_homo = np.vstack([pts_3d, np.ones((1, len(pts_3d[0])))])

    # camera pose
    cam_intrinsic_matrix = np.array([[10, 0, 0],
                                     [0, 10, 0],
                                     [0, 0, 1]]).astype(float)
    cam_dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0])

    tsfm = np.eye(4)
    # tsfm[:3, :3] = tf3.euler.axangle2mat((1, 1, 0), 3)
    tsfm[:3, :3] = tf3.euler.euler2mat(1, 3.2, 1)
    tsfm[:3, -1] = [0, 0, 200]
    tsfm = tsfm[:3, :4]

    pts = np.matmul(cam_intrinsic_matrix, np.matmul(tsfm, pts_3d_homo))
    pts = pts[:2, :] / pts[2, :]

    print('3D\n', pts_3d)
    print('2D\n', pts)
    print('Ro\n', cv2.Rodrigues(tsfm[:3, :3])[0])

    _, rvec, tvec = cv2.solvePnP(pts_3d.T, pts.T, cameraMatrix=cam_intrinsic_matrix, distCoeffs=cam_dist_coeffs)
    print('Pnp')
    print(rvec)
    print(tvec)

    _, rvec, tvec, mask = cv2.solvePnPRansac(pts_3d.T, pts.T, cameraMatrix=cam_intrinsic_matrix, distCoeffs=cam_dist_coeffs)
    print('Pnp Ransac')
    print(rvec)
    print(tvec)
    print(mask)

    cv2.solvePnPRefineLM(pts_3d.T, pts.T, cameraMatrix=cam_intrinsic_matrix, distCoeffs=cam_dist_coeffs, rvec=rvec, tvec=tvec)
    print('LM Ransac')
    print(rvec)
    print(tvec)
    print(mask)
    return


def main():
    # v1 = np.array([0, 0, 1])
    # for i in range(100):
    #     v2 = np.random.random((3, 1))
    #     v2 = np.matmul(tf3.euler.euler2mat(*v2), v1)
    #     v2 = np.array([10, 10, 10])
    #     r = rotation_between_3dv_rotate_axis(v1, v2)
    #
    #     est = np.matmul(r, v1)
    #     # if not np.alltrue(np.abs(v2 - est) < 0.0000000000001):
    #     print(r, '\n * \n', v1)
    #     print(v2, est)
    pnp_opencv_test()


if __name__ == '__main__':
    main()
