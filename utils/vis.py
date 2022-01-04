# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 12/14/21 5:22 PM
"""
import cv2
import numpy as np
import open3d as o3
# import geometry
# import transforms3d as tf3


class PointsAndCamera:
    def __init__(self):
        self.pc = o3.geometry.PointCloud()
        self.mesh_class = o3.geometry.TriangleMesh()
        self.frames = []
        self.frame_coords = None
        self.lineSet = o3.geometry.LineSet()

        # self.vis = o3.visualization.Visualizer()
        # self.vis.create_window(window_name="pc")
        # self.vis.get_render_option().point_size = 2

    def add(self, points=None, frame_pose=None):
        if points is not None:
            assert len(points.shape) == 2
            assert points.shape[1] == 3
            assert np.sum(np.matmul(frame_pose[:3, :3], frame_pose[:3, :3].T) - np.eye(3)) < 0.001, \
                'Expect rotation matrix to be orthonormal but R * R^t is ' + str(np.matmul(frame_pose[:3, :3], frame_pose[:3, :3].T))

            pts = np.asarray(self.pc.points)

            self.pc.points = o3.utility.Vector3dVector(np.vstack([pts, points]))
            # self.vis.add_geometry(self.pc)

        if frame_pose is not None:
            assert frame_pose.shape == (4, 4)
            # frame update
            frame = self.mesh_class.create_coordinate_frame(size=1.0) if len(self.frames) > 0 else \
                self.mesh_class.create_coordinate_frame(size=3.0)
            frame.transform(frame_pose)
            self.frames.append(frame)

            # line segment connect frames update
            self.frame_coords = frame_pose[:3, -1].reshape(1, 3) if self.frame_coords is None else \
                np.vstack((self.frame_coords, frame_pose[:3, -1]))
            if len(self.frame_coords) > 2:
                connection = np.arange(0, len(self.frame_coords) - 1)[:, None]
                connection = np.hstack((connection, connection + 1))
                self.lineSet.points = o3.utility.Vector3dVector(self.frame_coords)
                self.lineSet.lines = o3.utility.Vector2iVector(connection)

            # self.vis.add_geometry(self.frames[-1])

    def show(self):
        o3.visualization.draw_geometries([self.pc, self.lineSet] + self.frames)
        # o3.visualization.draw_geometries([self.lineSet] + self.frames)

        # self.vis.poll_events()
        # self.vis.update_renderer()
        # self.vis.run()
#
#
# class CameraPosesVis:
#     def __init__(self):
#         self.camera_frames = []
#
#         self.mesh_class = o3.geometry.TriangleMesh()
#         self.points = np.array([[0, 0, 0]])
#         self.lineSet = o3.geometry.LineSet()
#
#         # self.vis = o3.visualization.Visualizer()
#         # self.vis.create_window(window_name="trajectory")
#         # self.vis.get_render_option().point_size = 3
#         self.last_pose = None
#
#     def add_cam_frame(self, pose):
#         assert pose.shape == (4, 4)
#         assert np.sum(np.matmul(pose[:3, :3], pose[:3, :3].T) - np.eye(3)) < 0.001, \
#             'Expect rotation matrix to be orthonormal but R * R^t is ' + str(np.matmul(pose[:3, :3], pose[:3, :3].T))
#
#         '''make mesh object'''
#         # new camera frame
#         cam_frame = self.mesh_class.create_coordinate_frame(size=1.0)
#         cam_frame.transform(pose)
#
#         # line from last frame to new frame
#         self.points = np.vstack((self.points, pose[:3, -1]))
#
#         if len(self.points) > 2:
#             connection = np.arange(0, len(self.points)-1)[:, None]
#             connection = np.hstack((connection, connection+1))
#             self.lineSet.points = o3.utility.Vector3dVector(self.points)
#             self.lineSet.lines = o3.utility.Vector2iVector(connection)
#
#         # if self.last_pose is not None:
#         #     tf = np.matmul(pose, np.linalg.inv(self.last_pose))
#         #     translation = tf[:3, -1]
#         #     dis_tf = np.linalg.norm(tf[:3, -1])
#         #
#         #     trajectory_segment = self.mesh_class.create_arrow(cylinder_radius=dis_tf / 10.0, cone_radius=dis_tf / 8.0,
#         #                                                       cylinder_height=dis_tf * 0.9, cone_height=dis_tf * 0.1)
#         #     trajectory_segment.transform(self.last_pose)
#         #     # trajectory_segment.rotate(np.array([[],
#         #     #                                     [],
#         #     #                                     []]))
#         #     # trajectory_segment.rotate(tf[:3, :3])
#         #
#         # # self.camera_frames.append(cam_frame)
#
#         # using last 10 points only
#         # if len(self.points) > 50:
#         #     self.points = np.array([[0, 0, 0]])
#         #     self.vis.clear_geometries()
#
#         # self.vis.add_geometry(cam_frame)
#         # self.vis.add_geometry(self.lineSet)
#         # if trajectory_segment:
#         #     self.vis.add_geometry(trajectory_segment)
#         o3.visualization.draw_geometries([self.lineSet, cam_frame])
#
#         self.last_pose = pose
#         return
#
#     def show(self):
#         # self.vis.update_geometry()
#         self.vis.poll_events()
#         self.vis.update_renderer()
#         # o3.visualization.draw_geometries(self.camera_frames)


def random_color():
    color = np.random.random(3)
    color *= 255
    color = color.astype(int)
    return tuple(color.tolist())


def draw_matches(img1, pts1, img2, pts2):
    assert len(pts1) == len(pts2)
    m, n = img1.shape[:2]
    offset_1_2_2 = np.array([n, 0])
    img3 = np.hstack((img1, img2))
    for i in range(len(pts1)):
        line_color, circle_color = random_color(), random_color()
        pt1, pt2 = pts1[i], pts2[i] + offset_1_2_2
        cv2.circle(img3, pt1.astype(int), radius=3, color=circle_color)
        cv2.circle(img3, pt2.astype(int), radius=3, color=circle_color)
        cv2.line(img3, pt1.astype(int), pt2.astype(int), color=line_color)
    return img3


def indent_array(s, level=1):
    return '\t'*level + str(s).replace('\n', '\n'+'\t'*level)


def main():
    """"""
    '''random color'''
    # color = random_color()
    # print(color, type(color))
    # print(color[0], type(color[0]))

    '''transformation and open3d arrow'''
    tf = np.eye(4)
    tf[:3, -1] = (10, 10, 10)
    translation = tf[:3, -1]
    dis_tf = np.linalg.norm(translation)

    '''line'''
    line_start, line_end = np.array([0, 0, 0]), translation

    '''frames'''
    mesh_class = o3.geometry.TriangleMesh()
    world_frame = mesh_class.create_coordinate_frame(size=10)
    next_frame = mesh_class.create_coordinate_frame()
    next_frame.transform(tf)

    '''arrows'''
    line = o3.geometry.LineSet()
    line.lines = o3.utility.Vector2iVector(np.array([[0, 1]]))
    line.points = o3.utility.Vector3dVector(np.vstack([line_start, line_end]))

    # trajectory_segment.transform(tf)
    print(line_start)
    # trajectory_segment.translate(np.matmul(rotation_arrow, dir_arrow/2))

    o3.visualization.draw_geometries([world_frame, next_frame, line])


if __name__ == '__main__':
    main()
