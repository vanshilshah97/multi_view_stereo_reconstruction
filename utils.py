import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import imageio
from tqdm import tqdm
from transforms3d.euler import mat2euler, euler2mat
import pyrender
import trimesh
import cv2
from dataloader import load_middlebury_data

EPS = 1e-8


def add_coordinate(scene, R, T, axis_len=0.05, sections=6, ratio=20):
    T_base = np.eye(4)
    T_base[:3, :3] = R
    T_base[:3, 3] = T

    _trans = np.eye(4)
    _trans[:3, :3] = euler2mat(0, np.pi / 2, 0, "szyz")
    _trans[:3, 3] = np.array([axis_len / 2, 0.0, 0.0])
    mx = trimesh.creation.cylinder(
        axis_len / ratio,
        axis_len,
        transform=T_base @ _trans,
        sections=sections,
    )
    mx.visual.vertex_colors = np.ones(mx.vertices.shape) * np.array([[1.0, 0.0, 0.0]])
    scene.add(pyrender.Mesh.from_trimesh(mx))

    _trans = np.eye(4)
    _trans[:3, :3] = euler2mat(0, np.pi / 2, 0, "szxz")
    _trans[:3, 3] = np.array([0.0, axis_len / 2, 0.0])
    my = trimesh.creation.cylinder(
        axis_len / ratio, axis_len, transform=T_base @ _trans, sections=sections
    )
    my.visual.vertex_colors = np.ones(my.vertices.shape) * np.array([[0.0, 1.0, 0.0]])
    scene.add(pyrender.Mesh.from_trimesh(my))

    _trans = np.eye(4)
    _trans[:3, 3] = np.array([0.0, 0.0, axis_len / 2])
    mz = trimesh.creation.cylinder(
        axis_len / ratio, axis_len, transform=T_base @ _trans, sections=sections
    )
    mz.visual.vertex_colors = np.ones(mz.vertices.shape) * np.array([[0.0, 0.0, 1.0]])
    scene.add(pyrender.Mesh.from_trimesh(mz))

    return scene


def viz_camera_poses(DATA):
    scene = pyrender.Scene()

    for data in DATA:
        scene = add_coordinate(scene, data["R"].T, -(data["R"].T @ data["T"][:, None])[:, 0])
    pyrender.Viewer(scene, use_raymond_lighting=True)
    return
