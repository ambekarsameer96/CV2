from audioop import add
import numpy as np
import open3d as o3d
import cv2
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import sampling_matching_methods as sm
from scipy.stats import multivariate_normal


# globals.
DATA_DIR = 'Data'  # This depends on where this file is located. Change for your needs.

# == Load data ==
def vis_open3d(data):
    # visualization from ndarray
    vis_pcd = o3d.geometry.PointCloud()
    vis_pcd.points = o3d.utility.Vector3dVector(data)
    o3d.visualization.draw_geometries([vis_pcd])

def open3d(path, vis=False):
    pcd = o3d.io.read_point_cloud(path)

    # convert into ndarray
    pcd_arr = np.asarray(pcd.points)

    # clean the point cloud using a threshold
    # distances to origin (0,0,0)
    distances = np.sqrt(np.sum(pcd_arr ** 2, axis = 1))
    pcd_arr_cleaned = pcd_arr[distances < 2]

    if vis:
        vis_open3d(pcd_arr_cleaned)

    return pcd_arr_cleaned

def open_wave_data():
    target = np.load(os.path.join(DATA_DIR, 'wave_target.npy')).T
    source = np.load(os.path.join(DATA_DIR, 'wave_source.npy')).T
    return source, target

def open_bunny_data():
    target = np.load(os.path.join(DATA_DIR, 'bunny_target.npy')).T
    source = np.load(os.path.join(DATA_DIR, 'bunny_source.npy')).T
    return source, target

def file_picker(file_path):
    files1 = os.listdir(file_path)
    files1 = sorted(files1)
    l1 = []
    for file1 in files1:
        if '.pcd' in str(file1):
            if not 'normal' in str(file1):
                file1 = os.path.join(file_path, file1)
                l1.append(file1)
    return l1

def add_noise(pcd):
    mean = np.mean(pcd)
    std = np.std(pcd)
    noise = multivariate_normal.pdf(pcd[0], mean=mean, cov=std)
    return noise

def calc_R_t(source, target):
    """
    compute R and t for a given translated source and target point cloud.
    """
    source_mean = np.mean(source, axis=0)
    target_mean = np.mean(target, axis=0)

    source_centered = (source - source_mean).T
    target_centered = (target - target_mean).T

    cov = source_centered @ target_centered.T
    U, S, VT = np.linalg.svd(cov)
    mid = np.eye(3)
    mid[-1][-1] = np.linalg.det(VT.T @ U.T)

    R = VT.T @ mid @ U.T
    t = target_mean - (R @ source_mean)
    return R, t

def rototranslation(Rt, R, t):
    """
    update the rototranslation matrix to
    include iterative R and t update
    """
    Rt_add = np.eye(4)
    Rt_add[:3, :3] = R
    Rt_add[:3, 3] = t
    return Rt @ Rt_add

def compute_translation(points, Rt):
    R = Rt[:3, :3]
    t = Rt[:3, 3]
    points = (R @ points.T).T + t.T
    return points


def ICP(source, target, th=0.001, matching_fn=sm.min_dist, sampling_fn=None, iterative_sampling_fn=None):
    """
    Perform ICP algorithm and return rotation matrix and translation vector.
    """
    if sampling_fn is None:
        sampling_fn = sm.no_sampling

    if iterative_sampling_fn is None:
        iterative_sampling_fn = sm.no_sampling

    R = np.eye(3)
    t = np.zeros(3)
    Rt = np.eye(4)

    RMS_li = []
    RMS_prev = 100
    iterations = 500

    source_sampled = sampling_fn(source)
    target = sampling_fn(target)

    for _ in range(iterations):
        if iterative_sampling_fn is not None:
            source_sampled = iterative_sampling_fn(source)

        # match points from target to points from source
        target_BF = matching_fn(source_sampled, target)

        # compute RMS
        RMS = np.sqrt(np.mean(np.linalg.norm(target_BF - source_sampled, axis=1)))
        print(RMS)

        # compute R and T and update rototranslation matrix
        R, t = calc_R_t(source_sampled, target_BF)
        Rt = rototranslation(Rt, R, t)

        # rotate and translate source pointcloud
        source_sampled = (R @ source_sampled.T).T + t.T

        # for iterative sampling
        source = (R @ source.T).T + t.T

        # convergence condition
        if np.abs(RMS - RMS_prev) < 1e-5:
            break

        RMS_prev = RMS
        RMS_li.append(RMS)

    print(f"Mean RMS: {np.mean(RMS_li)}")
    return Rt, RMS_li

def run_ICP(source, target, args=dict(), o3dvis=False, matplotvis=True):
    Rt, RMS_li = ICP(source, target, **args)
    R = Rt[:3, :3]
    t = Rt[:3, 3]

    source_tr = (R @ source.T).T + t.T

    if o3dvis:
        final = np.append(source, source_tr, axis=0)
        vis_open3d(final)

    if matplotvis:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(target[:,0], target[:,1], target[:,2], s=1)
        ax.scatter3D(source_tr[:,0], source_tr[:,1], source_tr[:,2], s=1)
        plt.show()

    return RMS_li

############################
#       Experiments        #
############################

source, target = open_wave_data()
# source, target = open_bunny_data()

# Using all points
# RMS_li = run_ICP(source, target)
#
# # Uniform sub-sampling
# unif_args = {'sampling_fn': sm.unif_sampling}
# RMS_li = run_ICP(source, target, args=unif_args)
#
# # Random sub-sampling
# rand_args = {'iterative_sampling_fn': sm.unif_sampling}
# RMS_li = run_ICP(source, target, args=rand_args)

# Multi-resolution sub-sampling (TODO)
# ratio = 5
# multires_args = {'iterative_sampling_fn': sm.multires_sampling}
# R, t = run_ICP(source_multires, target_multires, args=multires_args)

# Informative region sub-sampling
# IR_args = {'sampling_fn': sm.density_sampling}
# RMS_li = run_ICP(source, target, args=IR_args)
#
# # KDTree
# KD_args = {'matching_fn': sm.kd_method}
# RMS_li = run_ICP(source, target, args=KD_args)

# z-buffer
ZB_args = {'matching_fn': sm.z_buffer}
RMS_li = run_ICP(source, target, args=ZB_args)

############################
#   Merge Scene            #
############################
# file_path = './Data/data/'
# file_list = file_picker(file_path)
#
# # 3.1
# #pick 2,4,10th files and send to main_ICP
# Rts = []
# steps = 4
# source = open3d(file_list[0])
#
# # Compute RT values for each pair of frames
# for i in range(steps, len(file_list), steps):
#     print(file_list[i])
#     target = open3d(file_list[i])
#     Rt, RMS = ICP(source, target, matching_fn=sm.kd_method, sampling_fn=sm.multires_sampling)
#     Rts.append(Rt)
#     source = target
#
# # Construct final point cloud at the end
# final = open3d(file_list[0])
# for en, i in enumerate(range(10, len(file_list), steps)):
#     print(f"{i=}")
#     target = open3d(file_list[i])
#     final = compute_translation(final, Rts[en])
#     final = np.vstack((final, target))
#
# final = np.delete(final, 0, axis=0)
# vis_open3d(final)
#
# # 3.2
# # pick 2,4,10th files and send to main_ICP
# final = open3d(file_list[0])
#
# # Compute RT values for each pair of frames
# for i in range(steps, len(file_list), steps):
#     print(file_list[i])
#     target = open3d(file_list[i])
#     Rt, RMS = ICP(final, target, matching_fn=sm.kd_method, sampling_fn=sm.multires_sampling)
#     final = compute_translation(final, Rt)
#     final = np.vstack((final, target))
#
# vis_open3d(final)
